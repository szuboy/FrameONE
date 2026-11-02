import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
from model import TransformerRegressionModel


class RegressionTrainer(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        model_cfg = cfg["model"]
        keys = (
            "in_channels",
            "feature_dim",
            "d_model",
            "nhead",
            "num_layers",
            "dim_feedforward",
            "dropout",
            "K_global",
            "K_local",
            "motion_scales",
        )
        self.encoder = TransformerRegressionModel(
            **{k: model_cfg[k] for k in keys if k in model_cfg}
        )
        self.weight_sigma = cfg.get("loss_weight", {}).get("gaussian_sigma", 2.0)
        if self.weight_sigma <= 0:
            raise ValueError("The Gaussian weighting sigma must be positive.")
        self.view_loss_weight = model_cfg.get("view_loss_weight", 0.1)
        self.motion_recon_weight = model_cfg.get("motion_recon_weight", 0.1)
        self.basis_ortho_weight = model_cfg.get("basis_ortho_weight", 0.01)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x, view_ids=None, return_view_logits=False):
        return self.encoder(x, view_ids=view_ids, return_view_logits=return_view_logits)

    def training_step(self, batch, batch_idx):
        video = batch["video"].to(self.device, dtype=torch.float32)
        es_mask = batch["label_ES"].to(self.device, dtype=torch.float32)
        ed_mask = batch["label_ED"].to(self.device, dtype=torch.float32)
        view_ids = batch["view_idx"].to(self.device, dtype=torch.long)
        output = self.encoder(
            video, view_ids=view_ids, return_view_logits=True, return_motion_basis=True
        )
        pred_scores = output["scores"]
        target = self.build_target_curve(es_mask, ed_mask, video.shape[1])
        frame_weights = self.build_frame_weights(es_mask, ed_mask)
        main_loss = (
            frame_weights * F.smooth_l1_loss(pred_scores, target, reduction="none")
        ).mean()
        motion_loss = F.mse_loss(output["motion_recon"], output["motion_gt"])
        ortho_loss = self.cross_basis_orthogonality_loss(view_ids)
        view_loss = F.cross_entropy(output["view_logits"], view_ids)
        return (
            main_loss
            + self.motion_recon_weight * motion_loss
            + self.basis_ortho_weight * ortho_loss
            + self.view_loss_weight * view_loss
        )

    def build_target_curve(self, es_mask, ed_mask, T):
        target = torch.empty_like(es_mask, dtype=torch.float32)
        frames = torch.arange(T, device=es_mask.device)
        for b in range(es_mask.shape[0]):
            es, ed = es_mask[b] > 0, ed_mask[b] > 0
            if not es.any() or not ed.any():
                raise ValueError("Training clips must contain both ES and ED labels.")
            if (es & ed).any():
                raise ValueError("A frame cannot be labeled both ES and ED.")
            indices = torch.nonzero(es | ed, as_tuple=False).squeeze(-1)
            values = ed[indices].float()
            right = torch.searchsorted(indices, frames, right=True)
            left = (right - 1).clamp(0, len(indices) - 1)
            right = right.clamp(0, len(indices) - 1)
            fraction = (frames - indices[left]).float() / (
                indices[right] - indices[left]
            ).clamp_min(1)
            target[b] = values[left] + fraction * (values[right] - values[left])
        return target

    def build_frame_weights(self, es_mask, ed_mask):
        frames = torch.arange(es_mask.shape[1], device=es_mask.device).float()
        weights = torch.ones_like(es_mask, dtype=torch.float32)
        for b in range(es_mask.shape[0]):
            indices = torch.nonzero(
                (es_mask[b] > 0) | (ed_mask[b] > 0), as_tuple=False
            ).squeeze(-1)
            distances = frames[:, None] - indices[None, :]
            weights[b] += torch.exp(
                -0.5 * (distances / self.weight_sigma).square()
            ).sum(dim=1)
        return weights

    def cross_basis_orthogonality_loss(self, view_ids):
        bases = self.encoder.motion_basis
        local_basis = bases.local_basis[view_ids]
        cross = torch.einsum("gd,bld->bgl", bases.global_basis, local_basis)
        return cross.square().sum(dim=(-2, -1)).mean()

    def test_step(self, batch, batch_idx):
        video = batch["video"].to(self.device, dtype=torch.float32)
        full_len = video.shape[1]
        fixed_length = self.cfg["dataset"]["fixed_length"]
        stride = self.cfg["test"]["stride"]
        if full_len < fixed_length:
            pad_len = fixed_length - full_len
            video = F.pad(
                video, (0, 0, 0, 0, 0, 0, 0, pad_len), mode="constant", value=0
            )
            full_len = fixed_length
        start_indices = list(range(0, full_len - fixed_length + 1, stride))
        if start_indices[-1] != full_len - fixed_length:
            start_indices.append(full_len - fixed_length)
        start_indices = sorted(start_indices)
        batch_info = {
            "gt": {},
            "pred": {},
            "mae": None,
            "clips": [],
            "mae_all": {},
            "meta": {},
        }
        score_dict = {i: [] for i in range(full_len)}
        for start in start_indices:
            clip = video[:, start : start + fixed_length]
            scores = self.forward(clip)
            clip_len = scores.shape[1]
            for local_idx in range(clip_len):
                global_idx = start + local_idx
                if global_idx < full_len:
                    score_dict[global_idx].append(scores[0, local_idx].item())
            batch_info["clips"].append({"start": start, "end": start + fixed_length})
        score_curve = np.zeros(full_len)
        for idx in range(full_len):
            if len(score_dict[idx]) > 0:
                score_curve[idx] = np.mean(score_dict[idx])
        sigma = self.cfg["test"].get("gaussian_sigma", 2.0)
        score_curve = gaussian_filter1d(score_curve, sigma=sigma)
        (ES_idx_list, ED_idx_list) = self.find_extrema_from_score_curve(
            score_curve, full_len
        )
        label_ES = batch["label_ES"].to(self.device)
        label_ED = batch["label_ED"].to(self.device)
        (es_valid, es_reason) = self.is_valid_label(label_ES, full_len)
        (ed_valid, ed_reason) = self.is_valid_label(label_ED, full_len)
        if not es_valid or not ed_valid:
            batch_info["gt"] = {
                "ES": (
                    f"invalid_{es_reason}"
                    if not es_valid
                    else (
                        int(label_ES.cpu())
                        if label_ES.dim() == 0
                        else label_ES.cpu().tolist()
                    )
                ),
                "ED": (
                    f"invalid_{ed_reason}"
                    if not ed_valid
                    else (
                        int(label_ED.cpu())
                        if label_ED.dim() == 0
                        else label_ED.cpu().tolist()
                    )
                ),
            }
            batch_info["pred"] = {
                "ES_idxs": ",".join((str(v) for v in ES_idx_list)),
                "ED_idxs": ",".join((str(v) for v in ED_idx_list)),
            }
            batch_info["mae"] = {
                "final_ES_mae": f"invalid_{es_reason}" if not es_valid else "valid",
                "final_ED_mae": f"invalid_{ed_reason}" if not ed_valid else "valid",
            }
            return (None, None, batch_info)
        mae_es = self._compute_mae(ES_idx_list, label_ES, fixed_length)
        mae_ed = self._compute_mae(ED_idx_list, label_ED, fixed_length)
        batch_info["mae"] = {
            "final_ES_mae": (
                int(mae_es)
                if not isinstance(mae_es, torch.Tensor)
                else int(mae_es.item())
            ),
            "final_ED_mae": (
                int(mae_ed)
                if not isinstance(mae_ed, torch.Tensor)
                else int(mae_ed.item())
            ),
        }
        batch_info["gt"] = {
            "ES": (
                int(label_ES.cpu()) if label_ES.dim() == 0 else label_ES.cpu().tolist()
            ),
            "ED": (
                int(label_ED.cpu()) if label_ED.dim() == 0 else label_ED.cpu().tolist()
            ),
        }
        batch_info["pred"] = {
            "ES_idxs": ",".join((str(v) for v in ES_idx_list)),
            "ED_idxs": ",".join((str(v) for v in ED_idx_list)),
        }
        return (mae_es, mae_ed, batch_info)

    def is_valid_label(self, label, video_length):
        if label is None:
            return (False, "None")
        if label.dim() == 0:
            label_val = label.item()
            if label_val < 0:
                return (False, "negative")
            if label_val <= 2 or label_val >= video_length - 2:
                return (False, "boundary")
            return (True, "valid")
        else:
            if label.numel() == 0:
                return (False, "empty")
            nonzero_indices = torch.nonzero(label, as_tuple=False)
            if len(nonzero_indices) == 0:
                return (False, "no_nonzero")
            return (True, "valid")

    def find_extrema_from_score_curve(self, curve, full_len, return_groups=False):
        base_es_threshold = self.cfg["test"].get("es_threshold", 0.5)
        base_ed_threshold = self.cfg["test"].get("ed_threshold", 0.5)
        window_size = self.cfg["test"].get("window_size", 5)
        es_threshold = base_es_threshold
        ed_threshold = base_ed_threshold
        es_candidates = np.where(curve < es_threshold)[0]
        ed_candidates = np.where(curve > ed_threshold)[0]

        def group_consecutive(indices):
            if len(indices) == 0:
                return []
            groups = []
            current_group = [indices[0]]
            for i in range(1, len(indices)):
                if indices[i] == indices[i - 1] + 1:
                    current_group.append(indices[i])
                else:
                    groups.append(current_group)
                    current_group = [indices[i]]
            groups.append(current_group)
            return groups

        es_groups = group_consecutive(es_candidates)
        ed_groups = group_consecutive(ed_candidates)
        use_fallback = self.cfg["test"].get("use_fallback", True)
        if use_fallback:
            if len(es_candidates) == 0:
                global_min_idx = np.argmin(curve)
                es_groups = [[global_min_idx]]
            if len(ed_candidates) == 0:
                global_max_idx = np.argmax(curve)
                ed_groups = [[global_max_idx]]

        def find_extremum_in_group(group, curve, window_size, find_min=True):
            if len(group) == 0:
                return None
            if len(group) < window_size:
                if find_min:
                    local_idx = np.argmin(curve[group])
                else:
                    local_idx = np.argmax(curve[group])
                return group[local_idx]
            best_idx = None
            best_val = float("inf") if find_min else float("-inf")
            for i in range(len(group) - window_size + 1):
                window_indices = group[i : i + window_size]
                window_values = curve[window_indices]
                if find_min:
                    local_idx = np.argmin(window_values)
                    if window_values[local_idx] < best_val:
                        best_val = window_values[local_idx]
                        best_idx = window_indices[local_idx]
                else:
                    local_idx = np.argmax(window_values)
                    if window_values[local_idx] > best_val:
                        best_val = window_values[local_idx]
                        best_idx = window_indices[local_idx]
            return best_idx

        ES_indices = []
        for group in es_groups:
            idx = find_extremum_in_group(group, curve, window_size, find_min=True)
            if idx is not None:
                ES_indices.append(idx)
        ED_indices = []
        for group in ed_groups:
            idx = find_extremum_in_group(group, curve, window_size, find_min=False)
            if idx is not None:
                ED_indices.append(idx)
        if return_groups:
            return (sorted(ES_indices), sorted(ED_indices), es_groups, ed_groups)
        else:
            return (sorted(ES_indices), sorted(ED_indices))

    def _compute_mae(self, pred_idx_list, label_idx, default_val):
        if len(pred_idx_list) > 0:
            mae_list = [abs(label_idx.item() - pred) for pred in pred_idx_list]
            return min(mae_list)
        else:
            return default_val
