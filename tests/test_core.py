import copy
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from evaluation import calculate_fp_fn
from model import CNNFeatureExtractor
from train import load_config, run_epoch
from trainer import RegressionTrainer


class FrameOneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
        cls.cfg["model"].update(
            feature_dim=16,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.0,
        )

    def test_forward_loss_and_gradients(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        batch = {
            "video": torch.rand(2, 8, 3, 16, 16),
            "view_idx": torch.tensor([0, 3]),
            "label_ES": torch.zeros(2, 8),
            "label_ED": torch.zeros(2, 8),
        }
        batch["label_ES"][:, 2] = 1
        batch["label_ED"][:, 6] = 1
        loss = model.training_step(batch, 0)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for parameter in model.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
        model.eval()
        with torch.no_grad():
            scores = model(batch["video"])
        self.assertEqual(scores.shape, (2, 8))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(((scores >= 0) & (scores <= 1)).all())

    def test_standard_resnet18_backbone(self):
        backbone = CNNFeatureExtractor(feature_dim=16).backbone
        self.assertEqual(
            [len(getattr(backbone, f"layer{i}")) for i in range(1, 5)], [2, 2, 2, 2]
        )
        self.assertEqual(backbone.layer1[0].conv1.out_channels, 64)
        self.assertEqual(backbone.fc.in_features, 512)
        self.assertEqual(backbone.fc.out_features, 16)

    def test_phase_target_interpolation(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        es, ed = torch.zeros(1, 13), torch.zeros(1, 13)
        es[0, 2], ed[0, 6], es[0, 10] = 1, 1, 1
        expected = torch.tensor(
            [[0, 0, 0, 0.25, 0.5, 0.75, 1, 0.75, 0.5, 0.25, 0, 0, 0]]
        )
        torch.testing.assert_close(model.build_target_curve(es, ed, 13), expected)
        torch.testing.assert_close(model.build_target_curve(ed, es, 13), 1 - expected)

    def test_gaussian_weights_do_not_cancel(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        es, ed = torch.zeros(1, 12), torch.zeros(1, 12)
        es[0, 5], ed[0, 6] = 1, 1
        weights = model.build_frame_weights(es, ed)
        expected_at_keyframe = 2 + np.exp(-0.125)
        self.assertAlmostEqual(weights[0, 5].item(), expected_at_keyframe, places=6)
        self.assertAlmostEqual(weights[0, 6].item(), expected_at_keyframe, places=6)
        torch.testing.assert_close(weights, model.build_frame_weights(ed, es))
        target = model.build_target_curve(es, ed, 12)
        model.weight_sigma = 4
        torch.testing.assert_close(target, model.build_target_curve(es, ed, 12))

    def test_cross_basis_orthogonality(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        bases = model.encoder.motion_basis
        with torch.no_grad():
            bases.global_basis.copy_(torch.eye(16)[:4])
            bases.local_basis[0].copy_(torch.eye(16)[4:7])
            bases.local_basis[3].copy_(torch.eye(16)[:3])
        self.assertEqual(
            model.cross_basis_orthogonality_loss(torch.tensor([0])).item(), 0
        )
        self.assertEqual(
            model.cross_basis_orthogonality_loss(torch.tensor([3])).item(), 3
        )
        self.assertEqual(
            model.cross_basis_orthogonality_loss(torch.tensor([0, 3])).item(), 1.5
        )

    def test_total_loss_has_no_amplitude_term(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        batch = {
            "video": torch.zeros(1, 8, 3, 16, 16),
            "view_idx": torch.tensor([0]),
            "label_ES": torch.zeros(1, 8),
            "label_ED": torch.zeros(1, 8),
        }
        batch["label_ES"][0, 1], batch["label_ED"][0, 5] = 1, 1
        with torch.no_grad():
            model.encoder.motion_basis.global_basis.copy_(torch.eye(16)[:4])
            model.encoder.motion_basis.local_basis[0].copy_(torch.eye(16)[:3])
        scores = torch.tensor([[0.8, 0.8, 0.7, 0.6, 0.2, 0.2, 0.1, 0.1]])
        output = {
            "scores": scores,
            "view_logits": torch.zeros(1, 4),
            "motion_recon": torch.full((1, 7, 16), 2.0),
            "motion_gt": torch.zeros(1, 7, 16),
        }
        target = torch.tensor([[0, 0, 0.25, 0.5, 0.75, 1, 1, 1]])
        t = torch.arange(8).float()
        weights = 1 + torch.exp(-((t - 1) ** 2) / 8) + torch.exp(-((t - 5) ** 2) / 8)
        expected = (
            (weights * 0.5 * (scores - target).square()).mean()
            + 0.1 * 4
            + 0.01 * 3
            + 0.1 * np.log(4)
        )
        with patch.object(model.encoder, "forward", return_value=output):
            actual = model.training_step(batch, 0)
        torch.testing.assert_close(actual, expected)

    def test_fixed_threshold_extrema(self):
        model = RegressionTrainer(copy.deepcopy(self.cfg))
        curve = np.array(
            [0.5, 0.3, 0.1, 0.3, 0.5, 0.7, 0.9, 0.7, 0.5, 0.4, 0.2, 0.4, 0.5]
        )
        es, ed = model.find_extrema_from_score_curve(curve, len(curve))
        self.assertEqual(es, [2, 10])
        self.assertEqual(ed, [6])

    def test_fp_fn_with_boundary_filtering(self):
        self.assertEqual(
            calculate_fp_fn([0, 10, 30, 49], [11, 40], total_frames=50), (1, 1)
        )

    def test_partial_accumulation_group(self):
        class SmallModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(0.0))
                self.device = torch.device("cpu")

            def training_step(self, batch, batch_idx):
                return (self.weight - batch) ** 2

        model = SmallModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        run_epoch(model, [torch.tensor(1.0)] * 3, optimizer, scheduler, scaler, 2)
        self.assertAlmostEqual(model.weight.item(), 0.36, places=6)
        self.assertEqual(scheduler.last_epoch, 2)


if __name__ == "__main__":
    unittest.main()
