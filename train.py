import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import FrameOneDataset
from evaluation import calculate_best_model, run_inference, run_inference_samples
from trainer import RegressionTrainer


def load_config(path):
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    path_keys = {
        "dataset": ("json_path", "data_root"),
        "train": ("pretrained_model", "val_full_label_dir", "test_full_label_dir"),
        "output": ("output_dir", "result"),
    }
    for section, keys in path_keys.items():
        for key in keys:
            value = cfg[section].get(key)
            if value:
                cfg[section][key] = str((path.parent / value).resolve())
    if cfg["dataset"]["view_list"] != ["A4C", "A2C", "PSAX", "PLAX"]:
        raise ValueError("view_list must be [A4C, A2C, PSAX, PLAX].")
    length = cfg["dataset"]["fixed_length"]
    if length < 2 or not 0 < cfg["test"]["stride"] <= length:
        raise ValueError("Require fixed_length >= 2 and 0 < stride <= fixed_length.")
    return cfg


def make_loader(cfg, split, device):
    dataset = FrameOneDataset(
        json_path=Path(cfg["dataset"]["json_path"]) / f"{split.capitalize()}.json",
        root_dir=cfg["dataset"]["data_root"],
        cfg=cfg,
        split=split,
    )
    if not len(dataset):
        raise ValueError(f"The {split} dataset is empty.")
    return DataLoader(
        dataset,
        batch_size=cfg["train"]["batch_size"] if split == "train" else 1,
        shuffle=split == "train",
        num_workers=cfg["train"].get("num_workers", 4),
        pin_memory=device.type == "cuda",
    )


def make_scheduler(optimizer, cfg, steps_per_epoch):
    total_steps = cfg["num_epochs"] * steps_per_epoch
    warmup_steps = cfg.get("warmup", 5) * steps_per_epoch

    def factor(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def run_epoch(model, dataloader, optimizer, scheduler, scaler, accumulation_steps):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="train")):
        group_start = (batch_idx // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(dataloader) - group_start)
        with torch.autocast(device_type=model.device.type, enabled=scaler.is_enabled()):
            loss = model.training_step(batch, batch_idx)
        total_loss += loss.item()
        scaler.scale(loss / group_size).backward()
        if (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == len(
            dataloader
        ):
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= previous_scale:
                scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return total_loss / len(dataloader)


def evaluate(model, dataloader, cfg, phase):
    mae, mae_es, mae_ed, per_view = run_inference(model, dataloader, phase)
    subset = run_inference_samples(model, phase, cfg, model.device)
    metrics = {
        "mae": float(mae),
        "mae_ES": float(mae_es),
        "mae_ED": float(mae_ed),
        "per_view": per_view,
    }
    if subset[0] is not None:
        metrics.update(fp=float(subset[0]), fn=float(subset[1]))
        for name, gt, pred in (
            ("ES", subset[6], subset[7]),
            ("ED", subset[8], subset[9]),
        ):
            if len(gt):
                gt, pred = np.asarray(gt, dtype=float), np.asarray(pred, dtype=float)
                residual = np.sum((gt - pred) ** 2)
                variance = np.sum((gt - gt.mean()) ** 2)
                metrics[f"rmse_{name}"] = float(np.sqrt(np.mean((gt - pred) ** 2)))
                metrics[f"r2_{name}"] = (
                    float(1 - residual / variance) if variance > 0 else None
                )
    with (Path(cfg["output"]["result"]) / f"{phase}_metrics.json").open("w") as handle:
        json.dump(metrics, handle, indent=2)
    return metrics


def train(cfg):
    seed = cfg["train"].get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(
        cfg["train"].get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    output_dir = Path(cfg["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(cfg["output"]["result"]).mkdir(parents=True, exist_ok=True)
    with (output_dir / "config.yaml").open("w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    model = RegressionTrainer(cfg).to(device)
    checkpoint_path = cfg["train"].get("pretrained_model")
    checkpoint = None
    if checkpoint_path:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["state_dict"])
    if cfg["test"].get("run_test", False):
        if checkpoint is None:
            raise ValueError("Set train.pretrained_model before running evaluation.")
        metrics = evaluate(model, make_loader(cfg, "test", device), cfg, "test")
        print(json.dumps(metrics, indent=2))
        return

    train_loader = make_loader(cfg, "train", device)
    val_loader = make_loader(cfg, "val", device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["train"]["lr"]), weight_decay=1e-5
    )
    accumulation = cfg["train"].get("accumulation_steps", 2)
    if accumulation < 1:
        raise ValueError("accumulation_steps must be positive.")
    scheduler = make_scheduler(
        optimizer, cfg["train"], math.ceil(len(train_loader) / accumulation)
    )
    scaler = torch.amp.GradScaler(
        "cuda", enabled=cfg["train"].get("use_amp", False) and device.type == "cuda"
    )
    start_epoch = 0
    best_loss = best_mae = best_weighted = float("inf")
    validation_results = []
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["opt_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])
        if "scaler_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_loss = checkpoint.get("best_loss", best_loss)
        best_mae = checkpoint.get("best_mae", best_mae)
        best_weighted = checkpoint.get("best_weighted_score", best_weighted)
        validation_results = checkpoint.get("validation_results", [])

    history_path = output_dir / "history.csv"
    for epoch in range(start_epoch, cfg["train"]["num_epochs"]):
        loss = run_epoch(
            model, train_loader, optimizer, scheduler, scaler, accumulation
        )
        metrics = {}
        if epoch % cfg["train"].get("val_period", 1) == 0:
            metrics = evaluate(model, val_loader, cfg, "val")
            with (Path(cfg["output"]["result"]) / f"val_epoch_{epoch}.json").open(
                "w"
            ) as handle:
                json.dump({"epoch": epoch, "loss": loss, **metrics}, handle, indent=2)
        mae = metrics.get("mae", float("inf"))
        score = 0.6 * mae + 0.4 * metrics["fp"] if "fp" in metrics else mae
        improved_loss, improved_mae, improved_score = (
            loss < best_loss,
            mae < best_mae,
            score < best_weighted,
        )
        best_loss, best_mae, best_weighted = (
            min(loss, best_loss),
            min(mae, best_mae),
            min(score, best_weighted),
        )
        if metrics:
            validation_results.append(
                {"epoch": epoch, "mae": mae, "fp": metrics.get("fp", 0.0)}
            )
        state = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "opt_dict": optimizer.state_dict(),
            "scheduler_dict": scheduler.state_dict(),
            "scaler_dict": scaler.state_dict(),
            "best_loss": best_loss,
            "best_mae": best_mae,
            "best_weighted_score": best_weighted,
            "validation_results": validation_results,
            "loss": loss,
        }
        torch.save(state, output_dir / "last_checkpoint.pt")
        for improved, name in (
            (improved_loss, "best_loss"),
            (improved_mae, "best_mae"),
            (improved_score, "best_weighted"),
        ):
            if improved:
                torch.save(state, output_dir / f"{name}.pt")
        best_idx = calculate_best_model(validation_results)
        if (
            metrics
            and best_idx is not None
            and validation_results[best_idx]["epoch"] == epoch
        ):
            torch.save(state, output_dir / "best_rank.pt")
        new_history = not history_path.exists()
        with history_path.open("a", newline="") as handle:
            writer = csv.writer(handle)
            if new_history:
                writer.writerow(
                    ["epoch", "loss", "mae", "mae_ES", "mae_ED", "fp", "fn"]
                )
            writer.writerow(
                [
                    epoch,
                    loss,
                    *[
                        metrics.get(k, "")
                        for k in ("mae", "mae_ES", "mae_ED", "fp", "fn")
                    ],
                ]
            )
        print(f"Epoch {epoch}: loss={loss:.5f}, metrics={metrics}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=str(Path(__file__).with_name("config.yaml"))
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Evaluate train.pretrained_model on Test.json.",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.test:
        config["test"]["run_test"] = True
    train(config)
