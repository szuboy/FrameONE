import math
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            d_model = self.pe.size(2)
            pe_ext = torch.zeros(seq_len, d_model, device=x.device, dtype=x.dtype)
            position = torch.arange(
                0, seq_len, dtype=torch.float, device=x.device
            ).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, dtype=torch.float, device=x.device)
                * (-math.log(10000.0) / d_model)
            )
            pe_ext[:, 0::2] = torch.sin(position * div_term)
            pe_ext[:, 1::2] = torch.cos(position * div_term)
            return x + pe_ext.unsqueeze(0)
        return x + self.pe[:, :seq_len]


class CNNFeatureExtractor(nn.Module):

    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        if in_channels != 3:
            raise ValueError("The ResNet18 backbone expects three-channel RGB frames.")
        self.backbone = resnet18(weights=None)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, feature_dim)

    def forward(self, x):
        return self.backbone(x)


class MultiScaleMotionExtractor(nn.Module):

    def __init__(self, feature_dim=256, scales=(1, 2, 4, 8)):
        super().__init__()
        self.temporal_convs = nn.ModuleList(
            [
                nn.Conv1d(
                    feature_dim,
                    feature_dim,
                    kernel_size=2 * s + 1,
                    padding=s,
                    groups=feature_dim,
                    bias=False,
                )
                for s in scales
            ]
        )
        self.fusion = nn.Sequential(
            nn.Linear(feature_dim * len(scales), feature_dim), nn.GELU()
        )

    def forward(self, frame_feats):
        x = frame_feats.transpose(1, 2)
        multi = [conv(x).transpose(1, 2) for conv in self.temporal_convs]
        fused = self.fusion(torch.cat(multi, dim=-1))
        return fused[:, :-1, :]


class MotionBasisDecomposition(nn.Module):

    def __init__(self, feature_dim=256, K_global=4, K_local=3, num_views=4):
        super().__init__()
        self.feature_dim = feature_dim
        self.K_global = K_global
        self.K_local = K_local
        self.num_views = num_views
        if K_global > 0:
            self.global_basis = nn.Parameter(torch.randn(K_global, feature_dim))
            nn.init.orthogonal_(self.global_basis)
            self.global_activation = nn.Linear(feature_dim, K_global)
            nn.init.xavier_uniform_(self.global_activation.weight)
            nn.init.zeros_(self.global_activation.bias)
        if K_local > 0:
            self.local_basis = nn.Parameter(
                torch.randn(num_views, K_local, feature_dim)
            )
            for v in range(num_views):
                nn.init.orthogonal_(self.local_basis[v])
            self.local_activation = nn.Linear(feature_dim, K_local)
            nn.init.xavier_uniform_(self.local_activation.weight)
            nn.init.zeros_(self.local_activation.bias)

    def forward(self, motion_feats, view_ids):
        (B, T_minus1) = (motion_feats.shape[0], motion_feats.shape[1])
        device = motion_feats.device
        alpha = None
        motion_global = torch.zeros(B, T_minus1, self.feature_dim, device=device)
        if self.K_global > 0:
            alpha = self.global_activation(motion_feats)
            alpha = F.softmax(alpha, dim=-1)
            motion_global = torch.einsum("btk,kd->btd", alpha, self.global_basis)
        beta = None
        motion_local = torch.zeros(B, T_minus1, self.feature_dim, device=device)
        if self.K_local > 0:
            beta = self.local_activation(motion_feats)
            beta = F.softmax(beta, dim=-1)
            local_basis_batch = self.local_basis[view_ids]
            motion_local = torch.einsum("btk,bkd->btd", beta, local_basis_batch)
        motion_recon = motion_global + motion_local
        return (alpha, beta, motion_recon)


class TransformerRegressionModel(nn.Module):

    def __init__(
        self,
        in_channels=3,
        feature_dim=256,
        d_model=256,
        nhead=8,
        num_layers=4,
        dim_feedforward=1024,
        dropout=0.1,
        K_global=4,
        K_local=3,
        motion_scales=(1, 2, 4, 8),
    ):
        super().__init__()
        if feature_dim != d_model:
            raise ValueError("feature_dim must equal d_model for view prediction.")
        if K_global < 1 or K_local < 1:
            raise ValueError("Both global and local motion bases are required.")
        self.feature_extractor = CNNFeatureExtractor(in_channels, feature_dim)
        self.motion_extractor = MultiScaleMotionExtractor(feature_dim, motion_scales)
        self.motion_basis = MotionBasisDecomposition(feature_dim, K_global, K_local)
        self.basis_proj = nn.Linear(K_global + K_local, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.regression_head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)
        self.view_classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 4)
        )

    def extract_frame_features(self, x):
        (batch, frames, channels, height, width) = x.shape
        features = self.feature_extractor(
            x.reshape(batch * frames, channels, height, width)
        )
        return features.reshape(batch, frames, -1)

    def forward(
        self, x, view_ids=None, return_view_logits=False, return_motion_basis=False
    ):
        if x.shape[1] < 2:
            raise ValueError("At least two frames are required.")
        frame_feats = self.extract_frame_features(x)
        motion_feats = self.motion_extractor(frame_feats)
        if view_ids is None:
            with torch.no_grad():
                temp = self.transformer(self.pos_encoder(self.dropout(frame_feats)))
                view_ids = self.view_classifier(temp.mean(dim=1)).argmax(dim=1)
        (alpha, beta, motion_recon) = self.motion_basis(motion_feats, view_ids)
        basis_feats = F.pad(
            torch.cat([alpha, beta], dim=-1), (0, 0, 0, 1), mode="replicate"
        )
        feats = self.transformer(
            self.pos_encoder(self.dropout(self.basis_proj(basis_feats)))
        )
        scores = self.regression_head(feats).squeeze(-1)
        if not return_view_logits and (not return_motion_basis):
            return scores
        output = {"scores": scores}
        if return_view_logits:
            output["view_logits"] = self.view_classifier(feats.mean(dim=1))
        if return_motion_basis:
            output.update(
                alpha=alpha,
                beta=beta,
                motion_recon=motion_recon,
                motion_gt=motion_feats,
            )
        return output
