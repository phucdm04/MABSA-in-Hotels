from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, CLIPVisionModel


Span = Tuple[int, int]


class MABSABaselineModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        vision_model_name: str = "openai/clip-vit-base-patch32",
        num_categories: int = 6,
        num_sentiments: int = 3,
        mate_loss_weight: float = 1.0,
        mote_loss_weight: float = 1.0,
        macc_loss_weight: float = 1.0,
        masc_loss_weight: float = 1.0,
        aope_loss_weight: float = 1.0,
        dropout_p: float = 0.1,
        cross_attn_heads: int = 8,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        lower_name = vision_model_name.lower()
        if "clip" in lower_name:
            self.image_encoder = CLIPVisionModel.from_pretrained(vision_model_name)
        else:
            self.image_encoder = AutoModel.from_pretrained(vision_model_name)

        self.loss_weights = {
            "mate_loss": mate_loss_weight,
            "mote_loss": mote_loss_weight,
            "macc_loss": macc_loss_weight,
            "masc_loss": masc_loss_weight,
            "aope_loss": aope_loss_weight,
            "mabsc_loss": 1.0,
            "macsa_loss": 1.0,
        }
        self.dropout = nn.Dropout(dropout_p)

        text_dim = self.text_encoder.config.hidden_size
        img_dim = self.image_encoder.config.hidden_size
        fusion_dim = text_dim + img_dim
        token_fusion_dim = 2 * text_dim
        mote_input_dim = token_fusion_dim + text_dim

        self.image_null = nn.Parameter(torch.randn(img_dim))
        self.text_norm = nn.LayerNorm(text_dim)
        self.image_norm = nn.LayerNorm(img_dim)
        self.span_norm = nn.LayerNorm(text_dim)
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.masc_input_norm = nn.LayerNorm(text_dim + fusion_dim)
        self.text_after_cross_norm = nn.LayerNorm(text_dim)
        self.image_after_cross_norm = nn.LayerNorm(text_dim)
        self.image_proj = nn.Linear(img_dim, text_dim)
        self.text_to_image_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=cross_attn_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.image_to_text_attn = nn.MultiheadAttention(
            embed_dim=text_dim,
            num_heads=cross_attn_heads,
            dropout=dropout_p,
            batch_first=True,
        )
        self.gate_linear = nn.Linear(2 * text_dim, text_dim)

        self.mate_head = nn.Sequential(
            nn.Linear(token_fusion_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_fusion_dim, 3),
        )
        self.mabsc_head = nn.Sequential(
            nn.Linear(token_fusion_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_fusion_dim, 5),
        )
        self.macsa_head = nn.Sequential(
            nn.Linear(token_fusion_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_fusion_dim, 8),
        )
        self.mote_head = nn.Sequential(
            nn.Linear(mote_input_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(token_fusion_dim, 3),
        )
        self.macc_head = nn.Sequential(
            nn.Linear(text_dim + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, num_categories),
        )
        self.masc_head = nn.Sequential(
            nn.Linear(text_dim + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, num_sentiments),
        )
        self.aope_head = nn.Sequential(
            nn.Linear(2 * text_dim + fusion_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(fusion_dim, 2),
        )

    @staticmethod
    def _get_span_repr(token_embeddings: torch.Tensor, span: Span) -> torch.Tensor:
        start, end = span
        if end <= start:
            return token_embeddings[start]
        return token_embeddings[start:end].mean(dim=0)

    def _encode_image(
        self,
        image: Optional[torch.Tensor],
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if image is None:
            null = self.image_null.unsqueeze(0).expand(batch_size, -1)
            return null, null.unsqueeze(1)
        img_out = self.image_encoder(pixel_values=image)
        img_tokens = img_out.last_hidden_state
        return img_tokens[:, 0, :], img_tokens

    def _build_aspect_inputs(
        self,
        H: torch.Tensor,
        G: torch.Tensor,
        aspect_spans: Sequence[Sequence[Span]],
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        reps: List[torch.Tensor] = []
        index_map: List[Tuple[int, int]] = []
        for b_idx, spans in enumerate(aspect_spans):
            for s_idx, span in enumerate(spans):
                a_repr = self._get_span_repr(H[b_idx], span)
                a_repr = self.span_norm(a_repr)
                x = torch.cat([a_repr, G[b_idx]], dim=-1)
                x = self.masc_input_norm(x)
                x = self.dropout(x)
                reps.append(x)
                index_map.append((b_idx, s_idx))
        if not reps:
            return None, index_map
        return torch.stack(reps, dim=0), index_map

    def _build_aspect_context(
        self,
        H: torch.Tensor,
        aspect_spans: Optional[Sequence[Sequence[Span]]],
    ) -> torch.Tensor:
        batch_size, _, text_dim = H.shape
        ctx = H.new_zeros((batch_size, text_dim))
        if aspect_spans is None:
            return ctx
        for b_idx, spans in enumerate(aspect_spans):
            reps: List[torch.Tensor] = []
            for span in spans:
                reps.append(self._get_span_repr(H[b_idx], span))
            if reps:
                ctx[b_idx] = torch.stack(reps, dim=0).mean(dim=0)
        return self.span_norm(ctx)

    def _build_span_reprs(
        self,
        H: torch.Tensor,
        spans_batch: Optional[Sequence[Sequence[Span]]],
    ) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for b_idx in range(H.size(0)):
            spans = spans_batch[b_idx] if spans_batch is not None and b_idx < len(spans_batch) else []
            reps: List[torch.Tensor] = []
            for span in spans:
                reps.append(self.span_norm(self._get_span_repr(H[b_idx], span)))
            out.append(torch.stack(reps, dim=0) if reps else H.new_zeros((0, H.size(-1))))
        return out

    def _compute_aope_logits(
        self,
        aspect_reprs: List[torch.Tensor],
        opinion_reprs: List[torch.Tensor],
        G: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int, int]]]:
        aope_feats: List[torch.Tensor] = []
        aope_index_map: List[Tuple[int, int, int]] = []
        for b_idx, a_reprs in enumerate(aspect_reprs):
            o_reprs = opinion_reprs[b_idx]
            if a_reprs.size(0) == 0 or o_reprs.size(0) == 0:
                continue
            for a_idx in range(a_reprs.size(0)):
                for o_idx in range(o_reprs.size(0)):
                    aope_feats.append(torch.cat([a_reprs[a_idx], o_reprs[o_idx], G[b_idx]], dim=-1))
                    aope_index_map.append((b_idx, a_idx, o_idx))
        if not aope_feats:
            return None, aope_index_map
        return self.aope_head(torch.stack(aope_feats, dim=0)), aope_index_map

    @staticmethod
    def _soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-8) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        target_1hot = F.one_hot(target, num_classes=num_classes).float()
        inter = (probs * target_1hot).sum(dim=0)
        union = probs.sum(dim=0) + target_1hot.sum(dim=0) - inter
        iou_per_class = (inter + eps) / (union + eps)
        return 1.0 - iou_per_class.mean()

    @staticmethod
    def _mixed_ce_iou_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ce_weight: float = 1.0,
        iou_weight: float = 1.0,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        if ignore_index is not None:
            valid_mask = target != ignore_index
            logits = logits[valid_mask]
            target = target[valid_mask]
        if target.numel() == 0:
            return logits.new_tensor(0.0)
        ce = F.cross_entropy(logits, target)
        iou = MABSABaselineModel._soft_iou_loss(logits=logits, target=target, num_classes=num_classes)
        return ce_weight * ce + iou_weight * iou

    @staticmethod
    def _balanced_mixed_ce_iou_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ce_weight: float = 1.0,
        iou_weight: float = 1.0,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        if ignore_index is not None:
            valid_mask = target != ignore_index
            logits = logits[valid_mask]
            target = target[valid_mask]
        if target.numel() == 0:
            return logits.new_tensor(0.0)
        counts = torch.bincount(target.clamp_min(0), minlength=num_classes).float().clamp_min(1.0)
        weights = (target.numel() / (num_classes * counts)).to(device=logits.device, dtype=logits.dtype)
        ce = F.cross_entropy(logits, target, weight=weights)
        iou = MABSABaselineModel._soft_iou_loss(logits=logits, target=target, num_classes=num_classes)
        return ce_weight * ce + iou_weight * iou

    @staticmethod
    def _balanced_ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
        valid_target = target[target != ignore_index] if ignore_index is not None else target
        if valid_target.numel() == 0:
            return logits.new_tensor(0.0)
        num_classes = logits.size(-1)
        counts = torch.bincount(valid_target.clamp_min(0), minlength=num_classes).float().clamp_min(1.0)
        weights = (valid_target.numel() / (num_classes * counts)).to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits,
            target,
            weight=weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        mate_labels: Optional[torch.Tensor] = None,
        mote_labels: Optional[torch.Tensor] = None,
        mabsc_labels: Optional[torch.Tensor] = None,
        macsa_labels: Optional[torch.Tensor] = None,
        categories: Optional[Sequence[Sequence[int]]] = None,
        aspect_spans: Optional[Sequence[Sequence[Span]]] = None,
        opinion_spans: Optional[Sequence[Sequence[Span]]] = None,
        aope_relations: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        sentiments: Optional[Sequence[Sequence[int]]] = None,
        task: str = "mate,mote,macc,masc",
    ) -> Dict[str, Any]:
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Keep custom fusion/head path in fp32 for numerical stability.
        H = text_out.last_hidden_state.float()

        cls = self.text_norm(H[:, 0, :])

        I, I_tokens = self._encode_image(image=image, batch_size=input_ids.size(0))
        I = I.float()
        I_tokens = I_tokens.float()
        I = self.image_norm(I)
        G = torch.cat([cls, I], dim=-1)
        G = self.fusion_norm(G)
        G = self.dropout(G)

        i_tokens_proj = self.image_proj(I_tokens)
        i_cross, _ = self.image_to_text_attn(query=i_tokens_proj, key=H, value=H)

        H_co = self.text_after_cross_norm(H)
        I_tokens_co = self.image_after_cross_norm(i_tokens_proj + i_cross)
        I_co = I_tokens_co[:, 0, :]
        I_co = self.dropout(I_co)

        gate = torch.sigmoid(self.gate_linear(torch.cat([H_co, I_co.unsqueeze(1).expand(-1, H_co.size(1), -1)], dim=-1)))
        H_fused = gate * H_co + (1.0 - gate) * I_co.unsqueeze(1).expand(-1, H_co.size(1), -1)
        H_fused = self.text_after_cross_norm(H_fused)

        n_tokens = H_fused.size(1)
        i_tok = I_co.unsqueeze(1).expand(-1, n_tokens, -1)
        h_fused = torch.cat([H_fused, i_tok], dim=-1)
        mate_logits = self.mate_head(h_fused)
        mabsc_logits = self.mabsc_head(h_fused)
        macsa_logits = self.macsa_head(h_fused)
        aspect_ctx = self._build_aspect_context(H_fused, aspect_spans)
        aspect_ctx_tok = aspect_ctx.unsqueeze(1).expand(-1, n_tokens, -1)
        mote_in = torch.cat([h_fused, aspect_ctx_tok], dim=-1)
        mote_logits = self.mote_head(mote_in)

        out: Dict[str, Any] = {
            "mate_logits": mate_logits,
            "mote_logits": mote_logits,
            "mabsc_logits": mabsc_logits,
            "macsa_logits": macsa_logits,
            "global_repr": G,
        }
        mate_mask = attention_mask.bool()

        masc_logits = None
        macc_logits = None
        masc_index_map: List[Tuple[int, int]] = []
        if aspect_spans is not None:
            aspect_inputs, masc_index_map = self._build_aspect_inputs(H_fused, G, aspect_spans)
            if aspect_inputs is not None:
                masc_logits = self.masc_head(aspect_inputs)
                macc_logits = self.macc_head(aspect_inputs)
        out["masc_logits"] = masc_logits
        out["masc_index_map"] = masc_index_map
        out["macc_logits"] = macc_logits
        out["macc_index_map"] = masc_index_map

        aspect_reprs = self._build_span_reprs(H_fused, aspect_spans)
        opinion_reprs = self._build_span_reprs(H_fused, opinion_spans)
        aope_logits, aope_index_map = self._compute_aope_logits(aspect_reprs, opinion_reprs, G)
        out["aope_logits"] = aope_logits
        out["aope_index_map"] = aope_index_map

        losses: Dict[str, torch.Tensor] = {}
        task_set = {x.strip().lower() for x in task.split(",") if x.strip()}
        if "asqp" in task_set or "quadra" in task_set or "quadprediction" in task_set or "quad" in task_set:
            task_set.update({"maope", "aope", "mabsc", "macsa"})
        if "quad_cls" in task_set or "quadclass" in task_set or "quad_classification" in task_set:
            task_set.update({"maope", "aope", "macc", "masc"})
        if "maope" in task_set:
            task_set.add("aope")

        if mate_labels is not None and "mate" in task_set:
            losses["mate_loss"] = F.cross_entropy(
                mate_logits.view(-1, mate_logits.size(-1)),
                mate_labels.view(-1),
                ignore_index=-100,
            )
        if mote_labels is not None and "mote" in task_set:
            losses["mote_loss"] = self._mixed_ce_iou_loss(
                logits=mote_logits.view(-1, mote_logits.size(-1)),
                target=mote_labels.view(-1),
                num_classes=mote_logits.size(-1),
                ce_weight=1.0,
                iou_weight=1.0,
                ignore_index=-100,
            )
        if mabsc_labels is not None and "mabsc" in task_set:
            losses["mabsc_loss"] = self._balanced_mixed_ce_iou_loss(
                logits=mabsc_logits.view(-1, mabsc_logits.size(-1)),
                target=mabsc_labels.view(-1),
                num_classes=mabsc_logits.size(-1),
                ce_weight=1.0,
                iou_weight=1.0,
                ignore_index=-100,
            )
        if macsa_labels is not None and "macsa" in task_set:
            losses["macsa_loss"] = self._balanced_mixed_ce_iou_loss(
                logits=macsa_logits.view(-1, macsa_logits.size(-1)),
                target=macsa_labels.view(-1),
                num_classes=macsa_logits.size(-1),
                ce_weight=1.0,
                iou_weight=1.0,
                ignore_index=-100,
            )
        if macc_logits is not None and categories is not None and "macc" in task_set:
            target_list: List[int] = []
            for b_idx, s_idx in masc_index_map:
                if b_idx < len(categories) and s_idx < len(categories[b_idx]):
                    target_list.append(int(categories[b_idx][s_idx]))
            if target_list:
                target = torch.tensor(target_list, dtype=torch.long, device=macc_logits.device)
                losses["macc_loss"] = self._balanced_ce_loss(macc_logits, target, ignore_index=-100)
        if masc_logits is not None and sentiments is not None and "masc" in task_set:
            target_list: List[int] = []
            for b_idx, s_idx in masc_index_map:
                if b_idx < len(sentiments) and s_idx < len(sentiments[b_idx]):
                    target_list.append(int(sentiments[b_idx][s_idx]))
            if target_list:
                target = torch.tensor(target_list, dtype=torch.long, device=masc_logits.device)
                losses["masc_loss"] = self._balanced_ce_loss(masc_logits, target)
        if aope_logits is not None and aope_relations is not None and "aope" in task_set:
            aope_sets = {b_idx: set(items) for b_idx, items in enumerate(aope_relations)}
            target_list: List[int] = []
            for b_idx, a_idx, o_idx in aope_index_map:
                target_list.append(1 if (b_idx in aope_sets and (a_idx, o_idx) in aope_sets[b_idx]) else 0)
            if target_list:
                aope_target = torch.tensor(target_list, dtype=torch.long, device=aope_logits.device)
                relation_loss = F.cross_entropy(aope_logits, aope_target)
                aux_losses: List[torch.Tensor] = []
                if mate_labels is not None:
                    aux_losses.append(
                        losses.get(
                            "mate_loss",
                            F.cross_entropy(
                                mate_logits.view(-1, mate_logits.size(-1)),
                                mate_labels.view(-1),
                                ignore_index=-100,
                            ),
                        )
                    )
                if mote_labels is not None:
                    aux_losses.append(
                        losses.get(
                            "mote_loss",
                            self._mixed_ce_iou_loss(
                                mote_logits.view(-1, mote_logits.size(-1)),
                                mote_labels.view(-1),
                                mote_logits.size(-1),
                                1.0,
                                1.0,
                                -100,
                            ),
                        )
                    )
                aux_loss = sum(aux_losses) if aux_losses else relation_loss.new_tensor(0.0)
                losses["aope_loss"] = aux_loss + relation_loss
        if losses:
            total_loss = 0.0
            for k, v in losses.items():
                total_loss = total_loss + self.loss_weights.get(k, 1.0) * v
            losses["total_loss"] = total_loss
            out["losses"] = losses
            out["loss"] = total_loss
        return out


class ABSABaselineModel(nn.Module):
    def __init__(
        self,
        text_model_name: str = "bert-base-uncased",
        num_categories: int = 6,
        num_sentiments: int = 3,
        mate_loss_weight: float = 1.0,
        mote_loss_weight: float = 1.0,
        macc_loss_weight: float = 1.0,
        masc_loss_weight: float = 1.0,
        aope_loss_weight: float = 1.0,
        dropout_p: float = 0.1,
    ) -> None:
        super().__init__()
        self.text_encoder = AutoModel.from_pretrained(text_model_name)
        self.loss_weights = {
            "mate_loss": mate_loss_weight,
            "mote_loss": mote_loss_weight,
            "macc_loss": macc_loss_weight,
            "masc_loss": masc_loss_weight,
            "aope_loss": aope_loss_weight,
            "mabsc_loss": 1.0,
            "macsa_loss": 1.0,
        }
        self.dropout = nn.Dropout(dropout_p)

        text_dim = self.text_encoder.config.hidden_size
        self.text_norm = nn.LayerNorm(text_dim)
        self.span_norm = nn.LayerNorm(text_dim)

        self.mate_head = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, 3))
        self.mote_head = nn.Sequential(nn.Linear(2 * text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, 3))
        self.mabsc_head = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, 5))
        self.macsa_head = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, 8))
        self.macc_head = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, num_categories))
        self.masc_head = nn.Sequential(nn.Linear(text_dim, text_dim), nn.ReLU(), nn.Linear(text_dim, num_sentiments))
        self.aope_head = nn.Sequential(nn.Linear(2 * text_dim, text_dim), nn.ReLU(), nn.Dropout(dropout_p), nn.Linear(text_dim, 2))

    @staticmethod
    def _get_span_repr(token_embeddings: torch.Tensor, span: Span) -> torch.Tensor:
        start, end = span
        if end <= start:
            return token_embeddings[start]
        return token_embeddings[start:end].mean(dim=0)

    @staticmethod
    def _soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, num_classes: int, eps: float = 1e-8) -> torch.Tensor:
        probs = F.softmax(logits, dim=-1)
        target_1hot = F.one_hot(target, num_classes=num_classes).float()
        inter = (probs * target_1hot).sum(dim=0)
        union = probs.sum(dim=0) + target_1hot.sum(dim=0) - inter
        iou_per_class = (inter + eps) / (union + eps)
        return 1.0 - iou_per_class.mean()

    @staticmethod
    def _mixed_ce_iou_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        num_classes: int,
        ce_weight: float = 1.0,
        iou_weight: float = 1.0,
        ignore_index: Optional[int] = None,
    ) -> torch.Tensor:
        if ignore_index is not None:
            valid_mask = target != ignore_index
            logits = logits[valid_mask]
            target = target[valid_mask]
        if target.numel() == 0:
            return logits.new_tensor(0.0)
        ce = F.cross_entropy(logits, target)
        iou = ABSABaselineModel._soft_iou_loss(logits=logits, target=target, num_classes=num_classes)
        return ce_weight * ce + iou_weight * iou

    @staticmethod
    def _balanced_ce_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: Optional[int] = None) -> torch.Tensor:
        valid_target = target[target != ignore_index] if ignore_index is not None else target
        if valid_target.numel() == 0:
            return logits.new_tensor(0.0)
        num_classes = logits.size(-1)
        counts = torch.bincount(valid_target.clamp_min(0), minlength=num_classes).float().clamp_min(1.0)
        weights = (valid_target.numel() / (num_classes * counts)).to(device=logits.device, dtype=logits.dtype)
        return F.cross_entropy(
            logits,
            target,
            weight=weights,
            ignore_index=ignore_index if ignore_index is not None else -100,
        )

    def _build_aspect_context(self, H: torch.Tensor, aspect_spans: Optional[Sequence[Sequence[Span]]]) -> torch.Tensor:
        batch_size, _, text_dim = H.shape
        ctx = H.new_zeros((batch_size, text_dim))
        if aspect_spans is None:
            return ctx
        for b_idx, spans in enumerate(aspect_spans):
            reps = [self._get_span_repr(H[b_idx], span) for span in spans]
            if reps:
                ctx[b_idx] = torch.stack(reps, dim=0).mean(dim=0)
        return self.span_norm(ctx)

    def _build_aspect_inputs(self, H: torch.Tensor, aspect_spans: Sequence[Sequence[Span]]) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        reps: List[torch.Tensor] = []
        index_map: List[Tuple[int, int]] = []
        for b_idx, spans in enumerate(aspect_spans):
            for s_idx, span in enumerate(spans):
                reps.append(self.dropout(self.span_norm(self._get_span_repr(H[b_idx], span))))
                index_map.append((b_idx, s_idx))
        if not reps:
            return None, index_map
        return torch.stack(reps, dim=0), index_map

    def _build_span_reprs(self, H: torch.Tensor, spans_batch: Optional[Sequence[Sequence[Span]]]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        for b_idx in range(H.size(0)):
            spans = spans_batch[b_idx] if spans_batch is not None and b_idx < len(spans_batch) else []
            reps = [self.span_norm(self._get_span_repr(H[b_idx], span)) for span in spans]
            out.append(torch.stack(reps, dim=0) if reps else H.new_zeros((0, H.size(-1))))
        return out

    def _compute_aope_logits(self, aspect_reprs: List[torch.Tensor], opinion_reprs: List[torch.Tensor]) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int, int]]]:
        aope_feats: List[torch.Tensor] = []
        aope_index_map: List[Tuple[int, int, int]] = []
        for b_idx, a_reprs in enumerate(aspect_reprs):
            o_reprs = opinion_reprs[b_idx]
            if a_reprs.size(0) == 0 or o_reprs.size(0) == 0:
                continue
            for a_idx in range(a_reprs.size(0)):
                for o_idx in range(o_reprs.size(0)):
                    aope_feats.append(torch.cat([a_reprs[a_idx], o_reprs[o_idx]], dim=-1))
                    aope_index_map.append((b_idx, a_idx, o_idx))
        if not aope_feats:
            return None, aope_index_map
        return self.aope_head(torch.stack(aope_feats, dim=0)), aope_index_map

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        mate_labels: Optional[torch.Tensor] = None,
        mote_labels: Optional[torch.Tensor] = None,
        mabsc_labels: Optional[torch.Tensor] = None,
        macsa_labels: Optional[torch.Tensor] = None,
        categories: Optional[Sequence[Sequence[int]]] = None,
        aspect_spans: Optional[Sequence[Sequence[Span]]] = None,
        opinion_spans: Optional[Sequence[Sequence[Span]]] = None,
        aope_relations: Optional[Sequence[Sequence[Tuple[int, int]]]] = None,
        sentiments: Optional[Sequence[Sequence[int]]] = None,
        task: str = "mate,mote,macc,masc",
    ) -> Dict[str, Any]:
        _ = image
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = self.dropout(self.text_norm(text_out.last_hidden_state))

        mate_logits = self.mate_head(H)
        mabsc_logits = self.mabsc_head(H)
        macsa_logits = self.macsa_head(H)
        aspect_ctx = self._build_aspect_context(H, aspect_spans)
        aspect_ctx_tok = aspect_ctx.unsqueeze(1).expand(-1, H.size(1), -1)
        mote_logits = self.mote_head(torch.cat([H, aspect_ctx_tok], dim=-1))

        out: Dict[str, Any] = {
            "mate_logits": mate_logits,
            "mote_logits": mote_logits,
            "mabsc_logits": mabsc_logits,
            "macsa_logits": macsa_logits,
            "global_repr": H[:, 0, :],
        }
        mate_mask = attention_mask.bool()

        masc_logits = None
        macc_logits = None
        masc_index_map: List[Tuple[int, int]] = []
        if aspect_spans is not None:
            aspect_inputs, masc_index_map = self._build_aspect_inputs(H, aspect_spans)
            if aspect_inputs is not None:
                masc_logits = self.masc_head(aspect_inputs)
                macc_logits = self.macc_head(aspect_inputs)
        out["masc_logits"] = masc_logits
        out["masc_index_map"] = masc_index_map
        out["macc_logits"] = macc_logits
        out["macc_index_map"] = masc_index_map

        aspect_reprs = self._build_span_reprs(H, aspect_spans)
        opinion_reprs = self._build_span_reprs(H, opinion_spans)
        aope_logits, aope_index_map = self._compute_aope_logits(aspect_reprs, opinion_reprs)
        out["aope_logits"] = aope_logits
        out["aope_index_map"] = aope_index_map

        losses: Dict[str, torch.Tensor] = {}
        task_set = {x.strip().lower() for x in task.split(",") if x.strip()}
        if "asqp" in task_set or "quadra" in task_set or "quadprediction" in task_set or "quad" in task_set:
            task_set.update({"maope", "aope", "mabsc", "macsa"})
        if "quad_cls" in task_set or "quadclass" in task_set or "quad_classification" in task_set:
            task_set.update({"maope", "aope", "macc", "masc"})
        if "maope" in task_set:
            task_set.add("aope")
        if mate_labels is not None and "mate" in task_set:
            losses["mate_loss"] = F.cross_entropy(
                mate_logits.view(-1, mate_logits.size(-1)),
                mate_labels.view(-1),
                ignore_index=-100,
            )
        if mote_labels is not None and "mote" in task_set:
            losses["mote_loss"] = self._mixed_ce_iou_loss(mote_logits.view(-1, mote_logits.size(-1)), mote_labels.view(-1), mote_logits.size(-1), 1.0, 1.0, -100)
        if mabsc_labels is not None and "mabsc" in task_set:
            losses["mabsc_loss"] = self._mixed_ce_iou_loss(
                mabsc_logits.view(-1, mabsc_logits.size(-1)),
                mabsc_labels.view(-1),
                mabsc_logits.size(-1),
                1.0,
                1.0,
                -100,
            )
        if macsa_labels is not None and "macsa" in task_set:
            losses["macsa_loss"] = self._mixed_ce_iou_loss(
                macsa_logits.view(-1, macsa_logits.size(-1)),
                macsa_labels.view(-1),
                macsa_logits.size(-1),
                1.0,
                1.0,
                -100,
            )
        if macc_logits is not None and categories is not None and "macc" in task_set:
            target_list = [int(categories[b_idx][s_idx]) for b_idx, s_idx in masc_index_map if b_idx < len(categories) and s_idx < len(categories[b_idx])]
            if target_list:
                target = torch.tensor(target_list, dtype=torch.long, device=macc_logits.device)
                losses["macc_loss"] = self._balanced_ce_loss(macc_logits, target, ignore_index=-100)
        if masc_logits is not None and sentiments is not None and "masc" in task_set:
            target_list = [int(sentiments[b_idx][s_idx]) for b_idx, s_idx in masc_index_map if b_idx < len(sentiments) and s_idx < len(sentiments[b_idx])]
            if target_list:
                target = torch.tensor(target_list, dtype=torch.long, device=masc_logits.device)
                losses["masc_loss"] = self._balanced_ce_loss(masc_logits, target)
        if aope_logits is not None and aope_relations is not None and "aope" in task_set:
            aope_sets = {b_idx: set(items) for b_idx, items in enumerate(aope_relations)}
            aope_target = [1 if (b_idx in aope_sets and (a_idx, o_idx) in aope_sets[b_idx]) else 0 for b_idx, a_idx, o_idx in aope_index_map]
            if aope_target:
                aope_target_tensor = torch.tensor(aope_target, dtype=torch.long, device=aope_logits.device)
                relation_loss = F.cross_entropy(aope_logits, aope_target_tensor)
                aux_losses: List[torch.Tensor] = []
                if mate_labels is not None:
                    aux_losses.append(
                        losses.get(
                            "mate_loss",
                            F.cross_entropy(
                                mate_logits.view(-1, mate_logits.size(-1)),
                                mate_labels.view(-1),
                                ignore_index=-100,
                            ),
                        )
                    )
                if mote_labels is not None:
                    aux_losses.append(
                        losses.get(
                            "mote_loss",
                            self._mixed_ce_iou_loss(
                                mote_logits.view(-1, mote_logits.size(-1)),
                                mote_labels.view(-1),
                                mote_logits.size(-1),
                                1.0,
                                1.0,
                                -100,
                            ),
                        )
                    )
                aux_loss = sum(aux_losses) if aux_losses else relation_loss.new_tensor(0.0)
                losses["aope_loss"] = aux_loss + relation_loss

        if losses:
            total_loss = 0.0
            for k, v in losses.items():
                total_loss = total_loss + self.loss_weights.get(k, 1.0) * v
            losses["total_loss"] = total_loss
            out["losses"] = losses
            out["loss"] = total_loss
        return out



