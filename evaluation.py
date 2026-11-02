import json
import os
import numpy as np
import torch
import tqdm
from dataset import FrameOneDataset


def calculate_best_model(epoch_results):
    if not epoch_results:
        return None
    mae_values = [r["mae"] for r in epoch_results]
    fp_values = [r["fp"] for r in epoch_results]
    mae_ranks = [sorted(mae_values).index(x) + 1 for x in mae_values]
    fp_ranks = [sorted(fp_values).index(x) + 1 for x in fp_values]
    combined_scores = [
        0.6 * mae_ranks[i] + 0.4 * fp_ranks[i] for i in range(len(epoch_results))
    ]
    best_idx = combined_scores.index(min(combined_scores))
    return best_idx


def run_inference(model, test_dataloader, phase):
    test_mean_mae_list = []
    test_mean_mae_ES_list = []
    test_mean_mae_ED_list = []
    all_test_info = []
    view_names = ["A4C", "A2C", "PSAX", "PLAX"]
    view_mae_dict = {
        view: {"mae_list": [], "mae_ES_list": [], "mae_ED_list": []}
        for view in view_names
    }
    model.eval()
    with torch.no_grad():
        with tqdm.tqdm(total=len(test_dataloader)) as pbar:
            for batch_index, batch in enumerate(test_dataloader):
                (mae_ES, mae_ED, batch_info) = model.test_step(batch, batch_index)
                if mae_ES is None or mae_ED is None:
                    all_test_info.append(batch_info)
                    pbar.update()
                    continue
                all_test_info.append(batch_info)
                mae_mean = (mae_ES + mae_ED) / 2
                mae_mean_val = (
                    mae_mean if type(mae_mean) != torch.Tensor else mae_mean.item()
                )
                mae_ES_val = mae_ES if type(mae_ES) != torch.Tensor else mae_ES.item()
                mae_ED_val = mae_ED if type(mae_ED) != torch.Tensor else mae_ED.item()
                test_mean_mae_list.append(mae_mean_val)
                test_mean_mae_ES_list.append(mae_ES_val)
                test_mean_mae_ED_list.append(mae_ED_val)
                view_idx_val = None
                if "view_idx" in batch:
                    view_idx_val = (
                        batch["view_idx"][0].item()
                        if hasattr(batch["view_idx"][0], "item")
                        else batch["view_idx"][0]
                    )
                    if 0 <= view_idx_val < len(view_names):
                        view_name = view_names[view_idx_val]
                        view_mae_dict[view_name]["mae_list"].append(mae_mean_val)
                        view_mae_dict[view_name]["mae_ES_list"].append(mae_ES_val)
                        view_mae_dict[view_name]["mae_ED_list"].append(mae_ED_val)
                pbar.set_postfix_str("current_mae_mean: {:.2f}".format(mae_mean))
                pbar.update()
    test_save_path = os.path.join(
        model.cfg["output"]["result"], f"{phase}_info_results.json"
    )
    with open(test_save_path, "w") as f:
        json.dump(all_test_info, f, indent=4)
    if len(test_mean_mae_list) == 0:
        raise ValueError(f"No valid samples were evaluated in {phase}.")
    view_mae_summary = {}
    for view_name, mae_data in view_mae_dict.items():
        if len(mae_data["mae_list"]) > 0:
            view_mae_summary[view_name] = {
                "mean_mae": np.mean(mae_data["mae_list"]),
                "mae_ES": np.mean(mae_data["mae_ES_list"]),
                "mae_ED": np.mean(mae_data["mae_ED_list"]),
                "num_samples": len(mae_data["mae_list"]),
            }
    return (
        np.mean(test_mean_mae_list),
        np.mean(test_mean_mae_ES_list),
        np.mean(test_mean_mae_ED_list),
        view_mae_summary,
    )


def filter_boundary_frames(pred_list, total_frames):
    if total_frames is None or total_frames <= 0:
        return pred_list
    last_frame = total_frames - 1
    filtered_list = [frame for frame in pred_list if frame != 0 and frame != last_frame]
    return filtered_list


def calculate_fp_fn(pred_list, gt_list, match_threshold=2, total_frames=None):
    if total_frames is not None:
        pred_list = filter_boundary_frames(pred_list, total_frames)
        gt_list = filter_boundary_frames(gt_list, total_frames)
    fp = 0
    matched_gt = set()
    for pred_frame in pred_list:
        matched = False
        for gt_frame in gt_list:
            if abs(pred_frame - gt_frame) <= match_threshold:
                matched = True
                matched_gt.add(gt_frame)
                break
        if not matched:
            fp += 1
    fn = len(gt_list) - len(matched_gt)
    return (fp, fn)


def run_inference_samples(model, phase, cfg, device):
    import tempfile

    if phase == "val":
        full_label_dir = cfg["train"].get("val_full_label_dir", None)
        full_label_files = cfg["train"].get("val_full_label_files", {})
    elif phase == "test":
        full_label_dir = cfg["train"].get("test_full_label_dir", None)
        full_label_files = cfg["train"].get("test_full_label_files", {})
    else:
        return (None, None, None, None, None, None, [], [], [], [])
    if full_label_dir is None or not os.path.exists(full_label_dir):
        return (None, None, None, None, None, None, [], [], [], [])
    if not full_label_files:
        return (None, None, None, None, None, None, [], [], [], [])
    all_views_fp_ES = 0
    all_views_fn_ES = 0
    all_views_fp_ED = 0
    all_views_fn_ED = 0
    all_views_valid_samples = 0
    all_gt_ES = []
    all_pred_ES = []
    all_gt_ED = []
    all_pred_ED = []
    view_results = {}
    for view_name, json_filename in full_label_files.items():
        sample_json_path = os.path.join(full_label_dir, json_filename)
        if not os.path.exists(sample_json_path):
            continue
        with open(sample_json_path, "r") as f:
            samples = json.load(f)
        converted_samples = []
        for sample in samples:
            file_path = sample["file_path"]
            file_name = sample["patient_id"]
            ED_list = sample["keyframe"]["ED"]
            ES_list = sample["keyframe"]["ES"]
            view = sample.get("view", view_name)
            if len(ED_list) > 0 and len(ES_list) > 0:
                converted_sample = {
                    "patient_id": file_name,
                    "file_path": file_path,
                    "view": view,
                    "keyframe": {"ED": ED_list[0], "ES": ES_list[0]},
                    "keyframe_all": {"ED": ED_list, "ES": ES_list},
                    "dataset_name": sample.get("dataset_name", "sample"),
                    "ED_ES_len": abs(ED_list[0] - ES_list[0]),
                    "segment_id": 0,
                }
                converted_sample["labels_zero_based"] = sample.get(
                    "labels_zero_based", False
                )
                converted_samples.append(converted_sample)
        if len(converted_samples) == 0:
            continue
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp_file:
            json.dump(converted_samples, tmp_file, indent=2)
            tmp_json_path = tmp_file.name
        try:
            sample_dataset = FrameOneDataset(
                json_path=tmp_json_path,
                root_dir=cfg["dataset"]["data_root"],
                cfg=cfg,
                split="val",
            )
            sample_dataloader = torch.utils.data.DataLoader(
                sample_dataset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                pin_memory=device.type == "cuda",
            )
            view_fp_ES = 0
            view_fn_ES = 0
            view_fp_ED = 0
            view_fn_ED = 0
            match_threshold = 2
            view_gt_ES = []
            view_pred_ES = []
            view_gt_ED = []
            view_pred_ED = []
            model.eval()
            valid_sample_count = 0
            with torch.no_grad():
                for batch_index, batch in enumerate(sample_dataloader):
                    (mae_ES, mae_ED, batch_info) = model.test_step(batch, batch_index)
                    if mae_ES is None or mae_ED is None:
                        continue
                    pred_ES_str = batch_info["pred"]["ES_idxs"]
                    pred_ED_str = batch_info["pred"]["ED_idxs"]
                    pred_ES_list = [int(x) for x in pred_ES_str.split(",") if x.strip()]
                    pred_ED_list = [int(x) for x in pred_ED_str.split(",") if x.strip()]
                    sample_idx = batch_index
                    if sample_idx < len(converted_samples):
                        gt_ES_list = (
                            converted_samples[sample_idx]
                            .get("keyframe_all", {})
                            .get("ES", [])
                        )
                        gt_ED_list = (
                            converted_samples[sample_idx]
                            .get("keyframe_all", {})
                            .get("ED", [])
                        )
                        base = sample_dataset.label_base(converted_samples[sample_idx])
                        gt_ES_list = [int(index) - base for index in gt_ES_list]
                        gt_ED_list = [int(index) - base for index in gt_ED_list]
                    else:
                        continue
                    total_frames = None
                    if "video" in batch and batch["video"] is not None:
                        total_frames = batch["video"].shape[1]
                    (fp_ES, fn_ES) = calculate_fp_fn(
                        pred_ES_list, gt_ES_list, match_threshold, total_frames
                    )
                    (fp_ED, fn_ED) = calculate_fp_fn(
                        pred_ED_list, gt_ED_list, match_threshold, total_frames
                    )
                    view_fp_ES += fp_ES
                    view_fn_ES += fn_ES
                    view_fp_ED += fp_ED
                    view_fn_ED += fn_ED
                    valid_sample_count += 1
                    for gt_es in gt_ES_list:
                        if len(pred_ES_list) > 0:
                            best_pred = min(pred_ES_list, key=lambda p: abs(p - gt_es))
                            all_gt_ES.append(gt_es)
                            all_pred_ES.append(best_pred)
                            view_gt_ES.append(gt_es)
                            view_pred_ES.append(best_pred)
                    for gt_ed in gt_ED_list:
                        if len(pred_ED_list) > 0:
                            best_pred = min(pred_ED_list, key=lambda p: abs(p - gt_ed))
                            all_gt_ED.append(gt_ed)
                            all_pred_ED.append(best_pred)
                            view_gt_ED.append(gt_ed)
                            view_pred_ED.append(best_pred)
            if valid_sample_count > 0:
                view_fp_mean_ES = view_fp_ES / valid_sample_count
                view_fp_mean_ED = view_fp_ED / valid_sample_count
                view_fp_mean = (view_fp_mean_ES + view_fp_mean_ED) / 2.0
                view_fn_mean_ES = view_fn_ES / valid_sample_count
                view_fn_mean_ED = view_fn_ED / valid_sample_count
                view_fn_mean = (view_fn_mean_ES + view_fn_mean_ED) / 2.0
                view_results[view_name] = {
                    "fp_mean": view_fp_mean,
                    "fn_mean": view_fn_mean,
                    "fp_ES": view_fp_ES,
                    "fp_ED": view_fp_ED,
                    "fn_ES": view_fn_ES,
                    "fn_ED": view_fn_ED,
                    "fp_mean_ES": view_fp_mean_ES,
                    "fp_mean_ED": view_fp_mean_ED,
                    "fn_mean_ES": view_fn_mean_ES,
                    "fn_mean_ED": view_fn_mean_ED,
                    "valid_samples": valid_sample_count,
                }
                view_rmse_ES = view_rmse_ED = view_r2_ES = view_r2_ED = None
                view_rmse_mean = view_r2_mean = None
                if len(view_gt_ES) > 0:
                    gt_arr = np.array(view_gt_ES, dtype=np.float64)
                    pred_arr = np.array(view_pred_ES, dtype=np.float64)
                    view_rmse_ES = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
                    ss_res = np.sum((gt_arr - pred_arr) ** 2)
                    ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
                    view_r2_ES = (
                        float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    )
                if len(view_gt_ED) > 0:
                    gt_arr = np.array(view_gt_ED, dtype=np.float64)
                    pred_arr = np.array(view_pred_ED, dtype=np.float64)
                    view_rmse_ED = float(np.sqrt(np.mean((pred_arr - gt_arr) ** 2)))
                    ss_res = np.sum((gt_arr - pred_arr) ** 2)
                    ss_tot = np.sum((gt_arr - np.mean(gt_arr)) ** 2)
                    view_r2_ED = (
                        float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    )
                if view_rmse_ES is not None and view_rmse_ED is not None:
                    view_rmse_mean = (view_rmse_ES + view_rmse_ED) / 2.0
                    view_r2_mean = (view_r2_ES + view_r2_ED) / 2.0
                view_results[view_name]["rmse_ES"] = view_rmse_ES
                view_results[view_name]["rmse_ED"] = view_rmse_ED
                view_results[view_name]["rmse_mean"] = view_rmse_mean
                view_results[view_name]["r2_ES"] = view_r2_ES
                view_results[view_name]["r2_ED"] = view_r2_ED
                view_results[view_name]["r2_mean"] = view_r2_mean
                all_views_fp_ES += view_fp_ES
                all_views_fn_ES += view_fn_ES
                all_views_fp_ED += view_fp_ED
                all_views_fn_ED += view_fn_ED
                all_views_valid_samples += valid_sample_count
        finally:
            if os.path.exists(tmp_json_path):
                os.remove(tmp_json_path)
    if all_views_valid_samples > 0:
        fp_mean_ES = all_views_fp_ES / all_views_valid_samples
        fp_mean_ED = all_views_fp_ED / all_views_valid_samples
        fp_mean = (fp_mean_ES + fp_mean_ED) / 2.0
        fn_mean_ES = all_views_fn_ES / all_views_valid_samples
        fn_mean_ED = all_views_fn_ED / all_views_valid_samples
        fn_mean = (fn_mean_ES + fn_mean_ED) / 2.0
        result_output_dir = cfg["output"]["result"]
        multi_view_result = {
            "phase": phase,
            "overall": {
                "fp_mean": fp_mean,
                "fn_mean": fn_mean,
                "fp_ES": all_views_fp_ES,
                "fp_ED": all_views_fp_ED,
                "fn_ES": all_views_fn_ES,
                "fn_ED": all_views_fn_ED,
                "valid_samples": all_views_valid_samples,
            },
            "per_view": view_results,
        }
        multi_view_result_path = os.path.join(
            result_output_dir, f"{phase}_multi_view_fp_fn.json"
        )
        with open(multi_view_result_path, "w") as f:
            json.dump(multi_view_result, f, indent=4)
        return (
            fp_mean,
            fn_mean,
            all_views_fp_ES,
            all_views_fp_ED,
            all_views_fn_ES,
            all_views_fn_ED,
            all_gt_ES,
            all_pred_ES,
            all_gt_ED,
            all_pred_ED,
        )
    else:
        return (None, None, None, None, None, None, [], [], [], [])
