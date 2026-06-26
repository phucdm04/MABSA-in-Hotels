from __future__ import annotations

import argparse
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model_similarity_guided import SimilarityGuidedMABSAModel
from train import PTDataset, collate_fn, evaluate, get_score_summary, move_to_device


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Expected true or false.")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_quad_task(task: str, quad_impl: str) -> str:
    names = {x.strip().lower() for x in task.split(",") if x.strip()}
    if not ({"quad", "quadprediction", "asqp", "quadra"} & names):
        return task
    names.discard("quad")
    names.discard("quadprediction")
    names.discard("asqp")
    names.discard("quadra")
    names.discard("quad_cls")
    names.discard("quadclass")
    names.discard("quad_classification")
    if quad_impl == "cls":
        names.add("quad_cls")
    else:
        names.add("quad")
    return ",".join(sorted(names))


def mean_enabled_score(score: dict, task: str) -> float:
    task_set = {x.strip().lower() for x in task.split(",") if x.strip()}
    if "asqp" in task_set or "quadra" in task_set or "quadprediction" in task_set or "quad" in task_set:
        task_set.update({"maope", "aope", "mabsc", "macsa", "quad"})
    if "quad_cls" in task_set or "quadclass" in task_set or "quad_classification" in task_set:
        task_set.update({"maope", "aope", "macc", "masc", "quad_cls"})
    if "maope" in task_set:
        task_set.add("aope")
    metric_values = []
    if "mate" in task_set:
        metric_values.append(score["mate_span_f1"])
    if "mote" in task_set:
        metric_values.append(score["mote_span_f1"])
    if "macc" in task_set:
        metric_values.append(score["macc_macro_f1"])
    if "masc" in task_set:
        metric_values.append(score["masc_macro_f1"])
    if "aope" in task_set:
        metric_values.append(score["aope_macro_f1"])
    if "mabsc" in task_set:
        metric_values.append(score["mabsc_span_f1"])
    if "macsa" in task_set:
        metric_values.append(score["macsa_span_f1"])
    if "quad" in task_set:
        metric_values.append(score["quad_f1"])
    if "quad_cls" in task_set:
        metric_values.append(score["quad_cls_f1"])
    return sum(metric_values) / max(len(metric_values), 1)


def train_one_epoch_similarity_guided(
    model: SimilarityGuidedMABSAModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    log_fn=None,
    epoch_idx: int = 1,
    task: str = "mate,mote,macc,masc",
    tb_writer: SummaryWriter | None = None,
    global_step_start: int = 0,
    tb_log_interval: int = 100,
    profile_steps: int = 0,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0
    steps = 0
    global_step = global_step_start

    for step_idx, batch in enumerate(tqdm(dataloader, desc="Training", leave=False), start=1):
        do_profile = profile_steps > 0 and step_idx <= profile_steps
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        step_t0 = time.perf_counter()
        batch = move_to_device(batch, device)
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        move_t1 = time.perf_counter()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            image=batch["image"],
            mate_labels=batch["mate_labels"],
            mote_labels=batch["mote_labels"],
            mabsc_labels=batch["mabsc_labels"],
            macsa_labels=batch["macsa_labels"],
            categories=batch["categories"],
            aspect_spans=batch["aspect_spans"],
            opinion_spans=batch["opinion_spans"],
            aope_relations=batch["aope_relations"],
            sentiments=batch["sentiments"],
            task=task,
            profile_forward=do_profile,
        )
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        forward_t1 = time.perf_counter()
        loss = outputs["loss"]

        optimizer.zero_grad()
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        zero_t1 = time.perf_counter()
        loss.backward()
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        backward_t1 = time.perf_counter()
        optimizer.step()
        if do_profile and device.type == "cuda":
            torch.cuda.synchronize()
        optim_t1 = time.perf_counter()

        batch_loss = float(loss.detach().cpu().item())
        running_loss += batch_loss
        steps += 1
        global_step += 1

        should_log_tb = tb_writer is not None and (step_idx % tb_log_interval == 0 or step_idx == len(dataloader))
        if should_log_tb:
            tb_writer.add_scalar("Loss/batch_total", batch_loss, global_step)
            tb_writer.add_scalar("Loss/running", running_loss / steps, global_step)
            for loss_name, loss_value in outputs.get("losses", {}).items():
                if loss_name == "total_loss":
                    continue
                tb_writer.add_scalar(f"Loss/head/{loss_name}", float(loss_value.detach().cpu().item()), global_step)

            tb_writer.flush()

        if log_fn is not None and (step_idx % 100 == 0 or step_idx == len(dataloader)):
            loss_parts = []
            weighted_loss_parts = []
            for loss_name, loss_value in outputs.get("losses", {}).items():
                if loss_name == "total_loss":
                    continue
                raw_value = float(loss_value.detach().cpu().item())
                loss_parts.append(f"{loss_name}={raw_value:.6f}")
                if loss_name in {"aope_contrastive_loss", "mabsc_boundary_loss"}:
                    continue
                if loss_name == "guidance_loss":
                    weight = float(getattr(model, "guidance_loss_weight", 1.0))
                else:
                    weight = float(getattr(model, "loss_weights", {}).get(loss_name, 1.0))
                weighted_loss_parts.append(f"weighted_{loss_name}={raw_value * weight:.6f}")
            suffix = " | " + " | ".join(loss_parts + weighted_loss_parts) if loss_parts else ""
            log_fn(
                f"[train][epoch {epoch_idx}][iter {step_idx}] "
                f"running_loss={running_loss / steps:.6f} | batch_total_loss={batch_loss:.6f}{suffix}"
            )

        if do_profile:
            end_t = time.perf_counter()
            if log_fn is not None:
                log_fn(f"[profile_steps][epoch {epoch_idx}][iter {step_idx}] timing seconds")
                log_fn(f"  move_to_device: {move_t1 - step_t0:.6f}")
                log_fn(f"  forward:        {forward_t1 - move_t1:.6f}")
                log_fn(f"  zero_grad:      {zero_t1 - forward_t1:.6f}")
                log_fn(f"  backward:       {backward_t1 - zero_t1:.6f}")
                log_fn(f"  optimizer_step: {optim_t1 - backward_t1:.6f}")
                log_fn(f"  post_step:      {end_t - optim_t1:.6f}")
                log_fn(f"  total:          {end_t - step_t0:.6f}")
                log_fn(f"  loss:           {batch_loss:.6f}")
                forward_timings = outputs.get("forward_timings", {})
                if forward_timings:
                    log_fn("  forward_detail:")
                    for name, value in forward_timings.items():
                        log_fn(f"    {name}: {value:.6f}")
            if step_idx >= profile_steps:
                return running_loss / max(steps, 1), global_step

    if tb_writer is not None:
        tb_writer.flush()
    return running_loss / max(steps, 1), global_step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimilarityGuidedMABSAModel")
    parser.add_argument("--train_dir", type=str, default="formatted_data/train")
    parser.add_argument("--val_dir", type=str, default="formatted_data/val")
    parser.add_argument("--save_dir", type=str, default="checkpoints_similarity_guided")
    parser.add_argument("--log_dir", type=str, default="log_similarity_guided")
    parser.add_argument("--text_model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--vision_model_name", type=str, default="google/vit-base-patch16-224")
    parser.add_argument("--num_categories", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--dropout_p", type=float, default=0.5)
    parser.add_argument("--guidance_mode", type=str, default="learned", choices=["learned", "cosine"])
    parser.add_argument("--guidance_loss_weight", type=float, default=0.5)
    parser.add_argument("--use_visual_branch", type=str2bool, default=True)
    parser.add_argument("--use_image_cross_attention", type=str2bool, default=True)
    parser.add_argument("--use_visual_gate", type=str2bool, default=True)
    parser.add_argument("--use_visual_guidance", type=str2bool, default=True)
    parser.add_argument("--use_guidance_loss", type=str2bool, default=True)
    parser.add_argument("--early_stopping_patience", type=int, default=3)
    parser.add_argument("--mate_loss_weight", type=float, default=1.0)
    parser.add_argument("--mote_loss_weight", type=float, default=1.0)
    parser.add_argument("--macc_loss_weight", type=float, default=1.0)
    parser.add_argument("--masc_loss_weight", type=float, default=1.0)
    parser.add_argument("--mabsc_loss_weight", type=float, default=1.0)
    parser.add_argument("--macsa_loss_weight", type=float, default=1.0)
    parser.add_argument("--aope_loss_weight", type=float, default=1.0)
    parser.add_argument("--maope_impl", type=str, default="ce", choices=["ce", "contrastive"])
    parser.add_argument("--maope_contrastive_weight", type=float, default=0.2)
    parser.add_argument("--task", type=str, default="mate,mote,macc,masc")
    parser.add_argument("--quad_impl", type=str, default="seq", choices=["seq", "cls"])
    parser.add_argument("--tensorboard", type=str2bool, default=False)
    parser.add_argument("--tb_log_interval", type=int, default=100)
    parser.add_argument("--profile_one_step", type=str2bool, default=False)
    parser.add_argument("--profile_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    args = parser.parse_args()
    if getattr(args, "aope_loss_weight", None) is not None:
        args.aope_loss_weight = args.aope_loss_weight
    effective_task = resolve_quad_task(args.task, args.quad_impl)
    set_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    device = torch.device(args.device)
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(args.log_dir, f"{run_time}_similarity_guided_{args.epochs}_{args.batch_size}_{args.lr}.log")
    tb_log_dir = os.path.join(args.log_dir, "tensorboard", run_time)
    tb_writer = SummaryWriter(tb_log_dir) if args.tensorboard else None

    def log(msg: str) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    train_ds = PTDataset(args.train_dir)
    val_ds = PTDataset(args.val_dir)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = SimilarityGuidedMABSAModel(
        text_model_name=args.text_model_name,
        vision_model_name=args.vision_model_name,
        num_categories=args.num_categories,
        mate_loss_weight=args.mate_loss_weight,
        mote_loss_weight=args.mote_loss_weight,
        macc_loss_weight=args.macc_loss_weight,
        masc_loss_weight=args.masc_loss_weight,
        aope_loss_weight=args.aope_loss_weight,
        dropout_p=args.dropout_p,
        guidance_mode=args.guidance_mode,
        guidance_loss_weight=args.guidance_loss_weight,
        use_visual_branch=args.use_visual_branch,
        use_image_cross_attention=args.use_image_cross_attention,
        use_visual_gate=args.use_visual_gate,
        use_visual_guidance=args.use_visual_guidance,
        use_guidance_loss=args.use_guidance_loss,
        mabsc_loss_weight=args.mabsc_loss_weight,
        macsa_loss_weight=args.macsa_loss_weight,
        maope_impl=args.maope_impl,
        maope_contrastive_weight=args.maope_contrastive_weight,
    ).to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    best_val = -1.0
    no_improve_epochs = 0
    global_step = 0
    log(f"Log file: {log_path}")
    if tb_writer is not None:
        log(f"TensorBoard log dir: {tb_log_dir}")
        tb_writer.add_scalar("Run/started", 1, 0)
        tb_writer.flush()
    log(f"Device: {args.device}")
    log(f"Seed: {args.seed}")
    log(f"TASK: {args.task}")
    log(f"QUAD_IMPL: {args.quad_impl}")
    log(f"MAOPE_IMPL: {args.maope_impl}")
    log(f"MAOPE_CONTRASTIVE_WEIGHT: {args.maope_contrastive_weight}")
    log(f"USE_VISUAL_BRANCH: {args.use_visual_branch}")
    log(f"USE_IMAGE_CROSS_ATTENTION: {args.use_image_cross_attention}")
    log(f"USE_VISUAL_GATE: {args.use_visual_gate}")
    log(f"USE_VISUAL_GUIDANCE: {args.use_visual_guidance}")
    log(f"USE_GUIDANCE_LOSS: {args.use_guidance_loss}")
    log(f"EFFECTIVE_TASK: {effective_task}")
    log(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")

    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_one_epoch_similarity_guided(
            model,
            train_loader,
            optimizer,
            device,
            log_fn=log,
            epoch_idx=epoch,
            task=effective_task,
            tb_writer=tb_writer,
            global_step_start=global_step,
            tb_log_interval=args.tb_log_interval,
            profile_steps=(1 if args.profile_one_step else args.profile_steps),
        )
        if args.profile_one_step or args.profile_steps > 0:
            log(f"profile stopped after {1 if args.profile_one_step else args.profile_steps} training step(s).")
            break
        val_metrics = evaluate(model, val_loader, device)
        val_score = get_score_summary(val_metrics)
        mean_score = mean_enabled_score(val_score, effective_task)

        log(f"\nEpoch {epoch}/{args.epochs}")
        log(f"train_loss: {train_loss:.6f}")
        log(
            f"val_metric - MATE_SPAN_F1: {val_score['mate_span_f1']:.4f} | "
            f"MOTE_SPAN_F1: {val_score['mote_span_f1']:.4f} | "
            f"MACC_MACRO_F1: {val_score['macc_macro_f1']:.4f} | "
            f"MASC_ACCURACY: {val_score['masc_accuracy']:.4f} | "
            f"MASC_MICRO_F1: {val_score['masc_micro_f1']:.4f} | "
            f"MASC_COUNT: {int(val_score['masc_correct'])}/{int(val_score['masc_total'])} | "
            f"MASC_MACRO_F1: {val_score['masc_macro_f1']:.4f} | "
            f"MABSC_SPAN_F1: {val_score['mabsc_span_f1']:.4f} | "
            f"MACSA_SPAN_F1: {val_score['macsa_span_f1']:.4f} | "
            f"MAOPE_F1: {val_score['aope_macro_f1']:.4f} | "
            f"QUAD_F1: {val_score['quad_f1']:.4f} | "
            f"QUAD_CLS_F1: {val_score['quad_cls_f1']:.4f}"
        )
        if tb_writer is not None:
            tb_writer.add_scalar("Loss/train_epoch", train_loss, epoch)
            tb_writer.add_scalar("Metric/val_mean_score", mean_score, epoch)
            tb_writer.add_scalar("Metric/val_mate_span_f1", val_score["mate_span_f1"], epoch)
            tb_writer.add_scalar("Metric/val_mote_span_f1", val_score["mote_span_f1"], epoch)
            tb_writer.add_scalar("Metric/val_macc_macro_f1", val_score["macc_macro_f1"], epoch)
            tb_writer.add_scalar("Metric/val_masc_accuracy", val_score["masc_accuracy"], epoch)
            tb_writer.add_scalar("Metric/val_masc_micro_f1", val_score["masc_micro_f1"], epoch)
            tb_writer.add_scalar("Metric/val_masc_macro_f1", val_score["masc_macro_f1"], epoch)
            tb_writer.add_scalar("Metric/val_mabsc_span_f1", val_score["mabsc_span_f1"], epoch)
            tb_writer.add_scalar("Metric/val_macsa_span_f1", val_score["macsa_span_f1"], epoch)
            tb_writer.add_scalar("Metric/val_aope_f1", val_score["aope_macro_f1"], epoch)
            tb_writer.add_scalar("Metric/val_quad_f1", val_score["quad_f1"], epoch)
            tb_writer.add_scalar("Metric/val_quad_cls_f1", val_score["quad_cls_f1"], epoch)
            tb_writer.flush()

        if mean_score > best_val:
            best_val = mean_score
            no_improve_epochs = 0
            ckpt_path = os.path.join(args.save_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            log(f"Saved best model to: {ckpt_path}")
        else:
            no_improve_epochs += 1
            log(f"No improvement for {no_improve_epochs} epoch(s).")
            if no_improve_epochs >= args.early_stopping_patience:
                log(f"Early stopping triggered at epoch {epoch}.")
                break

    if tb_writer is not None:
        tb_writer.close()


