from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import MABSABaselineModel


Span = Tuple[int, int]


class LinearChainCRF(nn.Module):
    def __init__(self, num_tags: int) -> None:
        super().__init__()
        self.start_transitions = nn.Parameter(torch.zeros(num_tags))
        self.end_transitions = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))

    def _log_partition_one(self, emissions: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions + emissions[0]
        for timestep in range(1, emissions.size(0)):
            score = torch.logsumexp(
                score.unsqueeze(1) + self.transitions + emissions[timestep].unsqueeze(0),
                dim=0,
            )
        return torch.logsumexp(score + self.end_transitions, dim=0)

    def _gold_score_one(self, emissions: torch.Tensor, tags: torch.Tensor) -> torch.Tensor:
        score = self.start_transitions[tags[0]] + emissions[0, tags[0]]
        for timestep in range(1, emissions.size(0)):
            score = score + self.transitions[tags[timestep - 1], tags[timestep]] + emissions[timestep, tags[timestep]]
        return score + self.end_transitions[tags[-1]]

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        for b_idx in range(emissions.size(0)):
            valid = mask[b_idx].bool()
            if valid.sum().item() == 0:
                continue
            seq_emissions = emissions[b_idx, valid]
            seq_tags = tags[b_idx, valid].long()
            losses.append(self._log_partition_one(seq_emissions) - self._gold_score_one(seq_emissions, seq_tags))
        if not losses:
            return emissions.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

    def _decode_one(self, emissions: torch.Tensor) -> List[int]:
        score = self.start_transitions + emissions[0]
        history: List[torch.Tensor] = []
        for timestep in range(1, emissions.size(0)):
            next_score = score.unsqueeze(1) + self.transitions + emissions[timestep].unsqueeze(0)
            best_score, best_path = next_score.max(dim=0)
            score = best_score
            history.append(best_path)
        score = score + self.end_transitions
        best_last = int(score.argmax().item())
        best_tags = [best_last]
        for best_path in reversed(history):
            best_last = int(best_path[best_last].item())
            best_tags.append(best_last)
        best_tags.reverse()
        return best_tags

    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        decoded = emissions.new_zeros((emissions.size(0), emissions.size(1)), dtype=torch.long)
        for b_idx in range(emissions.size(0)):
            valid = mask[b_idx].bool()
            if valid.sum().item() == 0:
                continue
            tags = self._decode_one(emissions[b_idx, valid])
            decoded[b_idx, valid] = torch.tensor(tags, dtype=torch.long, device=emissions.device)
        return decoded


class Guidance(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        lambda_global: float = 1.0,
        lambda_local: float = 1.0,
        lambda_aspect: float = 1.0,
    ) -> None:
        super().__init__()
        self.register_buffer("lambda_global_init", torch.tensor(float(lambda_global)))
        self.register_buffer("lambda_local_init", torch.tensor(float(lambda_local)))
        self.register_buffer("lambda_aspect_init", torch.tensor(float(lambda_aspect)))
        self.similarity_guide = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.text_proj = nn.Linear(hidden_size, hidden_size)
        self.image_proj = nn.Linear(hidden_size, hidden_size)
        self.log_temperature = nn.Parameter(torch.tensor(0.0))
        with torch.no_grad():
            self.similarity_guide[-1].bias.zero_()

    @staticmethod
    def _safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
        return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)

    @staticmethod
    def _get_span_repr(token_embeddings: torch.Tensor, span: Span) -> torch.Tensor:
        start, end = span
        if end <= start:
            return token_embeddings[start]
        return token_embeddings[start:end].mean(dim=0)

    def forward(
        self,
        text_tokens: torch.Tensor,
        image_tokens: torch.Tensor,
        aspect_spans: Optional[Sequence[Sequence[Span]]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = text_tokens.shape

        text_global_raw = text_tokens.mean(dim=1)
        image_global_raw = image_tokens.mean(dim=1)
        g_g = F.cosine_similarity(text_global_raw, image_global_raw, dim=-1)
        g_g_vec = g_g.unsqueeze(1).expand(B, L)

        image_patches = image_tokens[:, 1:, :] if image_tokens.size(1) > 1 else image_tokens
        temperature = self.log_temperature.exp().clamp_min(1e-6)

        text_token_proj = self.text_proj(text_tokens)
        image_patch_proj = self.image_proj(image_patches)
        text_token_norm = self._safe_normalize(text_token_proj / temperature, dim=-1)
        image_patch_norm = self._safe_normalize(image_patch_proj / temperature, dim=-1)

        local_sim = torch.bmm(text_token_norm, image_patch_norm.transpose(1, 2))
        top_k = max(1, int(math.sqrt(image_patch_norm.size(1))))
        g_l_vec = local_sim.topk(k=top_k, dim=-1).values.mean(dim=-1)
        local_alignment_loss = 1.0 - g_l_vec.mean()

        text_global = self._safe_normalize(self.text_proj(text_global_raw) / temperature, dim=-1)
        image_global = self._safe_normalize(self.image_proj(image_global_raw) / temperature, dim=-1)
        contrastive_logits = torch.matmul(text_global, image_global.transpose(0, 1))
        contrastive_targets = torch.arange(B, device=text_tokens.device)
        global_contrastive_loss = 0.5 * (
            F.cross_entropy(contrastive_logits, contrastive_targets)
            + F.cross_entropy(contrastive_logits.transpose(0, 1), contrastive_targets)
        )

        g_a_vec = text_tokens.new_zeros(B, L)
        aspect_alignment_losses = []

        if aspect_spans is not None:
            for b_idx, spans in enumerate(aspect_spans):
                if not spans:
                    continue

                aspect_scores: List[torch.Tensor] = []

                for span in spans:
                    aspect_repr = self._get_span_repr(
                        text_tokens[b_idx],
                        span,
                    )

                    aspect_norm = self._safe_normalize(
                        aspect_repr,
                        dim=-1,
                    )

                    aspect_sim = torch.matmul(
                        aspect_norm.unsqueeze(0),
                        image_patch_norm[b_idx].transpose(0, 1),
                    ).squeeze(0)

                    aspect_score = aspect_sim.topk(
                        k=top_k,
                        dim=-1,
                    ).values.mean()

                    aspect_scores.append(aspect_score)

                    # NEW
                    aspect_alignment_losses.append(
                        1.0 - aspect_score
                    )

                    start, end = span
                    start = max(0, min(int(start), L))
                    end = max(start, min(int(end), L))

                    if end > start:
                        g_a_vec[b_idx, start:end] = aspect_score

                if aspect_scores:
                    sample_aspect_score = torch.stack(
                        aspect_scores,
                        dim=0,
                    ).mean()

                    g_a_vec[b_idx] = torch.where(
                        g_a_vec[b_idx] == 0,
                        sample_aspect_score.expand_as(g_a_vec[b_idx]),
                        g_a_vec[b_idx],
                    )
        if aspect_alignment_losses:
            aspect_alignment_loss = torch.stack(
                aspect_alignment_losses
            ).mean()
        else:
            aspect_alignment_loss = text_tokens.new_tensor(0.0)


        scores = torch.stack([g_g_vec, g_l_vec, g_a_vec], dim=-1)
        init_weighted = (
            self.lambda_global_init * g_g_vec
            + self.lambda_local_init * g_l_vec
            + self.lambda_aspect_init * g_a_vec
        ).unsqueeze(-1)
        guidance = torch.sigmoid(self.similarity_guide(scores) + init_weighted)
        guidance_loss = local_alignment_loss + global_contrastive_loss + aspect_alignment_loss
        return guidance, guidance_loss


class SimilarityGuidedMABSAModel(MABSABaselineModel):
    """Baseline model with scalar semantic guidance from text-image similarity."""

    def __init__(
        self,
        *args,
        aope_loss_weight=None,
        guidance_mode: str = "semantic_scalar",
        lambda_global: float = 1.0,
        lambda_local: float = 1.0,
        lambda_aspect: float = 1.0,
        guidance_loss_weight: float = 0.5,
        use_visual_branch: bool = True,
        use_image_cross_attention: bool = True,
        use_visual_gate: bool = True,
        use_visual_guidance: bool = True,
        use_guidance_loss: bool = True,
        mabsc_loss_weight: float = 1.0,
        macsa_loss_weight: float = 1.0,
        masc_use_mote: bool = True,
        maope_impl: str = "ce",
        maope_contrastive_weight: float = 0.2,
        **kwargs,
    ) -> None:
        _ = guidance_mode
        if aope_loss_weight is not None and "aope_loss_weight" not in kwargs:
            kwargs["aope_loss_weight"] = aope_loss_weight
        super().__init__(*args, **kwargs)
        self.loss_weights["mabsc_loss"] = float(mabsc_loss_weight)
        self.loss_weights["macsa_loss"] = float(macsa_loss_weight)
        text_dim = self.text_encoder.config.hidden_size
        num_categories = kwargs.get("num_categories", 6)
        self.guidance = Guidance(
            hidden_size=text_dim,
            lambda_global=lambda_global,
            lambda_local=lambda_local,
            lambda_aspect=lambda_aspect,
        )
        self.maope_impl = str(maope_impl).lower()
        self.maope_contrastive_weight = float(maope_contrastive_weight)
        self.aope_aspect_proj = nn.Linear(text_dim, text_dim)
        self.aope_opinion_proj = nn.Linear(text_dim, text_dim)
        self.aope_log_temperature = nn.Parameter(torch.log(torch.tensor(0.07)))
        self.masc_use_mote = bool(masc_use_mote)
        self.macc_head = nn.Sequential(
            nn.Linear(text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(text_dim, num_categories),
        )
        if self.masc_use_mote:
            self.masc_mote_input_norm = nn.LayerNorm(2 * text_dim)
            self.masc_head = nn.Sequential(
                nn.Linear(2 * text_dim, text_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout.p),
                nn.Linear(text_dim, 3),
            )
        else:
            self.masc_head = nn.Sequential(
                nn.Linear(text_dim, text_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout.p),
                nn.Linear(text_dim, 3),
            )
        self.aope_head = nn.Sequential(
            nn.Linear(2 * text_dim, text_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(text_dim, 2),
        )
        token_fusion_dim = 2 * text_dim
        self.mabsc_head = nn.Sequential(
            nn.Linear(token_fusion_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(token_fusion_dim, 5),
        )
        self.mate_crf = LinearChainCRF(3)
        self.mote_crf = LinearChainCRF(3)
        self.mabsc_crf = LinearChainCRF(5)
        self.macsa_crf = LinearChainCRF(8)
        self.macsa_head = nn.Sequential(
            nn.Linear(token_fusion_dim, token_fusion_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout.p),
            nn.Linear(token_fusion_dim, 8),
        )
        self.guidance_loss_weight = float(guidance_loss_weight)
        self.use_visual_branch = bool(use_visual_branch)
        self.use_image_cross_attention = bool(use_image_cross_attention)
        self.use_visual_gate = bool(use_visual_gate)
        self.use_visual_guidance = bool(use_visual_guidance)
        self.use_guidance_loss = bool(use_guidance_loss)

    @staticmethod
    def _boundary_loss(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = -100) -> torch.Tensor:
        valid = target.ne(ignore_index)
        if valid.sum().item() == 0:
            return logits.new_tensor(0.0)
        logits = logits[valid]
        target = target[valid]
        probs = F.softmax(logits, dim=-1)
        start_target = torch.isin(target, torch.tensor([1, 2, 3], device=target.device)).to(dtype=logits.dtype)
        inside_target = target.ne(0).to(dtype=logits.dtype)
        start_prob = probs[:, 1:4].sum(dim=-1).clamp(1e-6, 1.0 - 1e-6)
        inside_prob = (1.0 - probs[:, 0]).clamp(1e-6, 1.0 - 1e-6)
        start_loss = F.binary_cross_entropy(start_prob, start_target)
        inside_loss = F.binary_cross_entropy(inside_prob, inside_target)
        return 0.5 * (start_loss + inside_loss)

    def _build_mote_context(self, H: torch.Tensor, mote_logits: torch.Tensor) -> torch.Tensor:
        opinion_probs = F.softmax(mote_logits, dim=-1)[..., 1:].sum(dim=-1)
        denom = opinion_probs.sum(dim=1, keepdim=True).clamp_min(1e-6)
        ctx = torch.bmm(opinion_probs.unsqueeze(1), H).squeeze(1) / denom
        return self.span_norm(ctx)

    def _build_masc_inputs_with_mote(
        self,
        H: torch.Tensor,
        aspect_spans: Sequence[Sequence[Span]],
        mote_logits: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        mote_ctx = self._build_mote_context(H, mote_logits)
        reps: List[torch.Tensor] = []
        index_map: List[Tuple[int, int]] = []
        for b_idx, spans in enumerate(aspect_spans):
            for s_idx, span in enumerate(spans):
                aspect_repr = self.span_norm(self._get_span_repr(H[b_idx], span))
                x = torch.cat([aspect_repr, mote_ctx[b_idx]], dim=-1)
                x = self.masc_mote_input_norm(x)
                x = self.dropout(x)
                reps.append(x)
                index_map.append((b_idx, s_idx))
        if not reps:
            return None, index_map
        return torch.stack(reps, dim=0), index_map

    def _build_aspect_inputs_without_global(
        self,
        H: torch.Tensor,
        aspect_spans: Sequence[Sequence[Span]],
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        reps: List[torch.Tensor] = []
        index_map: List[Tuple[int, int]] = []
        for b_idx, spans in enumerate(aspect_spans):
            for s_idx, span in enumerate(spans):
                aspect_repr = self.span_norm(self._get_span_repr(H[b_idx], span))
                reps.append(self.dropout(aspect_repr))
                index_map.append((b_idx, s_idx))
        if not reps:
            return None, index_map
        return torch.stack(reps, dim=0), index_map

    def _compute_aope_logits_without_global(
        self,
        aspect_reprs: List[torch.Tensor],
        opinion_reprs: List[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int, int]]]:
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

    def _aope_contrastive_loss(
        self,
        aspect_reprs: List[torch.Tensor],
        opinion_reprs: List[torch.Tensor],
        aope_relations: Sequence[Sequence[Tuple[int, int]]],
    ) -> torch.Tensor:
        losses: List[torch.Tensor] = []
        temperature = self.aope_log_temperature.exp().clamp_min(1e-6)
        for b_idx, relations in enumerate(aope_relations):
            if b_idx >= len(aspect_reprs) or b_idx >= len(opinion_reprs):
                continue
            a_reprs = aspect_reprs[b_idx]
            o_reprs = opinion_reprs[b_idx]
            if a_reprs.size(0) == 0 or o_reprs.size(0) == 0:
                continue
            a_proj = F.normalize(self.aope_aspect_proj(a_reprs), dim=-1)
            o_proj = F.normalize(self.aope_opinion_proj(o_reprs), dim=-1)
            logits = torch.matmul(a_proj, o_proj.transpose(0, 1)) / temperature
            valid_rows: List[torch.Tensor] = []
            targets: List[int] = []
            seen = set()
            for a_idx, o_idx in relations:
                a_idx = int(a_idx)
                o_idx = int(o_idx)
                if a_idx in seen:
                    continue
                if 0 <= a_idx < logits.size(0) and 0 <= o_idx < logits.size(1):
                    valid_rows.append(logits[a_idx])
                    targets.append(o_idx)
                    seen.add(a_idx)
            if valid_rows:
                losses.append(
                    F.cross_entropy(
                        torch.stack(valid_rows, dim=0),
                        torch.tensor(targets, dtype=torch.long, device=logits.device),
                    )
                )
        if not losses:
            return aspect_reprs[0].new_tensor(0.0) if aspect_reprs else self.aope_log_temperature.new_tensor(0.0)
        return torch.stack(losses, dim=0).mean()

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
        profile_forward: bool = False,
    ) -> Dict[str, Any]:
        _ = profile_forward
        text_out = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        H = text_out.last_hidden_state.float()

        if self.use_visual_branch:
            _, I_tokens = self._encode_image(image=image, batch_size=input_ids.size(0))
            I_tokens = I_tokens.float()
            S_v = self.image_proj(I_tokens)
            visual_guidance, guidance_loss = self.guidance(H, S_v, aspect_spans)
        else:
            S_v = H.new_zeros(input_ids.size(0), 1, H.size(-1))
            visual_guidance = H.new_zeros(input_ids.size(0), H.size(1), 1)
            guidance_loss = H.new_tensor(0.0)

        if not self.use_visual_guidance:
            visual_guidance = torch.zeros_like(visual_guidance)
        if not self.use_guidance_loss:
            guidance_loss = H.new_tensor(0.0)

        i_tokens_proj = S_v

        if self.use_visual_branch and self.use_image_cross_attention:
            h_cross, _ = self.text_to_image_attn(query=H, key=i_tokens_proj, value=i_tokens_proj)
            i_cross, _ = self.image_to_text_attn(query=i_tokens_proj, key=H, value=H)
        else:
            h_cross = torch.zeros_like(H)
            i_cross = torch.zeros_like(i_tokens_proj)

        H_cross = self.text_after_cross_norm(H + h_cross)
        image_tokens_cross = self.image_after_cross_norm(i_tokens_proj + i_cross)
        visual_ctx = image_tokens_cross[:, 1:, :].mean(dim=1) if image_tokens_cross.size(1) > 1 else image_tokens_cross[:, 0, :]
        visual_ctx = self.dropout(visual_ctx)
        visual_ctx_tok = visual_ctx.unsqueeze(1).expand(-1, H_cross.size(1), -1)

        if self.use_visual_branch and self.use_visual_gate:
            gate = torch.sigmoid(self.gate_linear(torch.cat([H_cross, visual_ctx_tok], dim=-1)))
            H_cross = gate * H_cross + (1.0 - gate) * visual_ctx_tok
            visual_ctx_for_heads = visual_ctx_tok
        else:
            visual_ctx_for_heads = torch.zeros_like(visual_ctx_tok)
        H_fused = H_cross + H_cross * visual_guidance
        H_fused = self.text_after_cross_norm(H_fused)

        n_tokens = H_fused.size(1)
        h_fused = torch.cat([H_fused, visual_ctx_for_heads[:, :n_tokens, :]], dim=-1)
        
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
            "global_repr": H_fused[:, 0, :],
            "visual_guidance": visual_guidance.squeeze(-1),
        }
        sequence_mask = attention_mask.bool()
        mate_valid_mask = sequence_mask
        if mate_labels is not None:
            mate_valid_mask = mate_valid_mask & mate_labels.ne(-100)
        mote_valid_mask = sequence_mask
        if mote_labels is not None:
            mote_valid_mask = mote_valid_mask & mote_labels.ne(-100)
        mabsc_valid_mask = sequence_mask
        if mabsc_labels is not None:
            mabsc_valid_mask = mabsc_valid_mask & mabsc_labels.ne(-100)
        macsa_valid_mask = sequence_mask
        if macsa_labels is not None:
            macsa_valid_mask = macsa_valid_mask & macsa_labels.ne(-100)
        out["mate_pred"] = self.mate_crf.decode(mate_logits, mate_valid_mask)
        out["mote_pred"] = self.mote_crf.decode(mote_logits, mote_valid_mask)
        out["mabsc_pred"] = self.mabsc_crf.decode(mabsc_logits, mabsc_valid_mask)
        out["macsa_pred"] = self.macsa_crf.decode(macsa_logits, macsa_valid_mask)

        masc_logits = None
        macc_logits = None
        masc_index_map: List[Tuple[int, int]] = []
        if aspect_spans is not None:
            aspect_inputs, masc_index_map = self._build_aspect_inputs_without_global(H_fused, aspect_spans)
            if aspect_inputs is not None:
                macc_logits = self.macc_head(aspect_inputs)
                if not self.masc_use_mote:
                    masc_logits = self.masc_head(aspect_inputs)
            if self.masc_use_mote:
                masc_inputs, masc_index_map = self._build_masc_inputs_with_mote(H_fused, aspect_spans, mote_logits)
                if masc_inputs is not None:
                    masc_logits = self.masc_head(masc_inputs)
        out["masc_logits"] = masc_logits
        out["masc_index_map"] = masc_index_map
        out["macc_logits"] = macc_logits
        out["macc_index_map"] = masc_index_map

        aspect_reprs = self._build_span_reprs(H_fused, aspect_spans)
        opinion_reprs = self._build_span_reprs(H_fused, opinion_spans)
        aope_logits, aope_index_map = self._compute_aope_logits_without_global(aspect_reprs, opinion_reprs)
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
            mate_mask = attention_mask.bool() & mate_labels.ne(-100)
            crf_loss = self.mate_crf.neg_log_likelihood(mate_logits, mate_labels.clamp_min(0), mate_mask)
            boundary_loss = self._boundary_loss(mate_logits.view(-1, mate_logits.size(-1)), mate_labels.view(-1), -100)
            losses["mate_boundary_loss"] = boundary_loss
            losses["mate_loss"] = crf_loss + boundary_loss
        if mote_labels is not None and "mote" in task_set:
            mote_mask = attention_mask.bool() & mote_labels.ne(-100)
            crf_loss = self.mote_crf.neg_log_likelihood(mote_logits, mote_labels.clamp_min(0), mote_mask)
            boundary_loss = self._boundary_loss(mote_logits.view(-1, mote_logits.size(-1)), mote_labels.view(-1), -100)
            losses["mote_boundary_loss"] = boundary_loss
            losses["mote_loss"] = crf_loss + boundary_loss
        if mabsc_labels is not None and "mabsc" in task_set:
            mabsc_mask = attention_mask.bool() & mabsc_labels.ne(-100)
            crf_loss = self.mabsc_crf.neg_log_likelihood(mabsc_logits, mabsc_labels.clamp_min(0), mabsc_mask)
            boundary_loss = self._boundary_loss(mabsc_logits.view(-1, mabsc_logits.size(-1)), mabsc_labels.view(-1), -100)
            losses["mabsc_boundary_loss"] = boundary_loss
            losses["mabsc_loss"] = crf_loss + boundary_loss
        if macsa_labels is not None and "macsa" in task_set:
            macsa_mask = attention_mask.bool() & macsa_labels.ne(-100)
            crf_loss = self.macsa_crf.neg_log_likelihood(macsa_logits, macsa_labels.clamp_min(0), macsa_mask)
            boundary_loss = self._boundary_loss(macsa_logits.view(-1, macsa_logits.size(-1)), macsa_labels.view(-1), -100)
            losses["macsa_boundary_loss"] = boundary_loss
            losses["macsa_loss"] = crf_loss + boundary_loss
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
                counts = torch.bincount(aope_target_tensor, minlength=2).float().clamp_min(1.0)
                aope_weight = (counts.sum() / (2.0 * counts)).to(device=aope_logits.device, dtype=aope_logits.dtype)
                relation_loss = F.cross_entropy(aope_logits, aope_target_tensor, weight=aope_weight)
                aux_losses: List[torch.Tensor] = []
                if mate_labels is not None:
                    mate_mask = attention_mask.bool() & mate_labels.ne(-100)
                    aux_losses.append(
                        losses.get(
                            "mate_loss",
                            self.mate_crf.neg_log_likelihood(mate_logits, mate_labels.clamp_min(0), mate_mask)
                            + self._boundary_loss(mate_logits.view(-1, mate_logits.size(-1)), mate_labels.view(-1), -100),
                        )
                    )
                if mote_labels is not None:
                    mote_mask = attention_mask.bool() & mote_labels.ne(-100)
                    aux_losses.append(
                        losses.get(
                            "mote_loss",
                            self.mote_crf.neg_log_likelihood(mote_logits, mote_labels.clamp_min(0), mote_mask)
                            + self._boundary_loss(mote_logits.view(-1, mote_logits.size(-1)), mote_labels.view(-1), -100),
                        )
                    )
                aux_loss = sum(aux_losses) if aux_losses else relation_loss.new_tensor(0.0)
                if self.maope_impl == "contrastive":
                    contrastive_loss = self._aope_contrastive_loss(aspect_reprs, opinion_reprs, aope_relations)
                    losses["aope_contrastive_loss"] = contrastive_loss
                    losses["aope_loss"] = aux_loss + relation_loss + self.maope_contrastive_weight * contrastive_loss
                else:
                    losses["aope_loss"] = aux_loss + relation_loss
        losses["guidance_loss"] = guidance_loss

        if losses:
            total_loss = 0.0
            for k, v in losses.items():
                if k in {
                    "aope_contrastive_loss",
                    "mate_boundary_loss",
                    "mote_boundary_loss",
                    "mabsc_boundary_loss",
                    "macsa_boundary_loss",
                }:
                    continue
                if k == "guidance_loss":
                    total_loss = total_loss + self.guidance_loss_weight * v
                else:
                    total_loss = total_loss + self.loss_weights.get(k, 1.0) * v
            losses["total_loss"] = total_loss
            out["losses"] = losses
            out["loss"] = total_loss
        return out
