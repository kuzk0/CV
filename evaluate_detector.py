from __future__ import annotations

from datetime import datetime
import csv
import re
import time
from multiprocessing import freeze_support
from pathlib import Path

import torch
from PIL import __version__ as pillow_version
from torch.utils.data import DataLoader

from detector_common import (
    DEFAULT_INPUT_SIZE,
    DEFAULT_NUM_SLOTS,
    BoundingBoxDataset,
    compute_detection_loss,
    compute_detection_metrics,
    create_model,
    decode_predictions,
    detection_collate_fn,
    get_device,
    load_class_names,
)

# ----------------------------- CONFIGURATION -----------------------------
DATASET_ROOT = Path("./Datasets/Plates")
CLASSES_FILE = Path(
    "./Datasets/Plates/classes.names"
)  # Должен совпадать с файлом при обучении.
TEST_IMAGES = DATASET_ROOT / "images" / "test"
TEST_LABELS = DATASET_ROOT / "labels" / "test"
CHECKPOINT = Path("./outputs/best_detector.pth")
OUTPUT_CSV = Path("./outputs/test_metrics.csv")
BATCH_SIZE = 4
NUM_WORKERS = 2
SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.50
# ------------------------------------------------------------------------


def calculate_test_loss(model, loader, device, input_size: int) -> float:
    """Средний тестовый loss по известной разметке; тестовые данные не обновляют веса."""
    model.eval()
    total_loss = 0.0
    image_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            image_batch = torch.stack(images).to(device, non_blocking=True)
            losses = compute_detection_loss(model(image_batch), targets, input_size)
            total_loss += float(losses["total"].item()) * len(images)
            image_count += len(images)
    return total_loss / max(image_count, 1)


@torch.inference_mode()
def run_inference(
    model,
    loader,
    device,
    input_size: int,
    score_threshold: float,
    nms_iou_threshold: float,
):
    model.eval()
    predictions = []
    targets_all = []
    inference_seconds = 0.0
    postprocessing_seconds = 0.0
    image_count = 0

    for images, targets in loader:
        image_batch = torch.stack(images).to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        raw_predictions = model(image_batch)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        inference_seconds += time.perf_counter() - start

        postprocess_start = time.perf_counter()
        batch_predictions = decode_predictions(
            raw_predictions,
            input_size=input_size,
            score_threshold=score_threshold,
            nms_iou_threshold=nms_iou_threshold,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        postprocessing_seconds += time.perf_counter() - postprocess_start

        predictions.extend(
            {key: value.detach().cpu() for key, value in prediction.items()}
            for prediction in batch_predictions
        )
        targets_all.extend(targets)
        image_count += len(images)

    return (
        predictions,
        targets_all,
        inference_seconds,
        postprocessing_seconds,
        image_count,
    )


def safe_column_name(name: str) -> str:
    normalized = re.sub(r"[^0-9A-Za-zА-Яа-я_-]+", "_", name.strip()).strip("_")
    return normalized or "unnamed"


def append_csv(path: Path, row: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    if file_exists:
        with path.open("r", newline="", encoding="utf-8-sig") as existing_file:
            existing_header = next(csv.reader(existing_file), [])
        if existing_header != list(row.keys()):
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
            file_exists = False

    with path.open("a", newline="", encoding="utf-8-sig") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def main():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(
            f"Не найден checkpoint: {CHECKPOINT.resolve()}. Сначала запустите train_detector.py."
        )

    device = get_device()
    checkpoint = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    checkpoint_classes = checkpoint.get("class_names")
    if not checkpoint_classes:
        raise ValueError(
            "В checkpoint отсутствует class_names. Нужен файл, созданный train_detector.py."
        )

    class_names = load_class_names(CLASSES_FILE)
    if class_names != checkpoint_classes:
        raise ValueError(
            "Список классов из classes.names отличается от списка в checkpoint.\n"
            f"В checkpoint: {checkpoint_classes}\nВ файле: {class_names}\n"
            "Используйте тот же classes.names, что и при обучении."
        )

    if checkpoint.get("architecture") != "CustomCNNDetector":
        raise ValueError(
            "Этот checkpoint не создан собственной архитектурой CustomCNNDetector. "
            "Запустите train_detector.py заново, чтобы обучить и сохранить новую модель без готовых архитектур."
        )

    model_config = checkpoint.get("config", {})
    input_size = int(model_config.get("input_size", DEFAULT_INPUT_SIZE))
    num_slots = int(model_config.get("num_slots", DEFAULT_NUM_SLOTS))
    # Эти пороги задаются независимо для каждого тестового запуска.
    score_threshold = SCORE_THRESHOLD
    nms_iou_threshold = NMS_IOU_THRESHOLD

    test_dataset = BoundingBoxDataset(
        TEST_IMAGES, TEST_LABELS, class_names, image_size=input_size
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    model = create_model(len(class_names), num_slots=num_slots).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    run_start = time.perf_counter()
    test_loss = calculate_test_loss(model, test_loader, device, input_size)
    predictions, targets, inference_seconds, postprocessing_seconds, image_count = (
        run_inference(
            model, test_loader, device, input_size, score_threshold, nms_iou_threshold
        )
    )
    metrics = compute_detection_metrics(
        predictions, targets, len(class_names), score_threshold
    )
    total_seconds = time.perf_counter() - run_start

    row = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": checkpoint.get("epoch", ""),
        "architecture": checkpoint.get("architecture", "unknown"),
        "dataset_root": str(DATASET_ROOT),
        "test_images_dir": str(TEST_IMAGES),
        "test_labels_dir": str(TEST_LABELS),
        "classes_file": str(CLASSES_FILE),
        "class_names": " | ".join(class_names),
        "num_classes": len(class_names),
        "num_test_images": image_count,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "input_size": input_size,
        "num_slots_per_cell": num_slots,
        "score_threshold": score_threshold,
        "nms_iou_threshold": nms_iou_threshold,
        "iou_threshold_for_precision_recall_f1": 0.5,
        "device": str(device),
        "torch_version": torch.__version__,
        "pillow_version": pillow_version,
        "parameter_count": parameter_count,
        "test_loss_per_image": test_loss,
        "precision_at_threshold": metrics["precision"],
        "recall_at_threshold": metrics["recall"],
        "f1_at_threshold": metrics["f1"],
        "mAP50": metrics["mAP50"],
        "mAP50_95": metrics["mAP50_95"],
        "mean_iou_of_matched_boxes": metrics["mean_iou_matched"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "ground_truth_objects": metrics["ground_truth_objects"],
        "inference_seconds_model_only": inference_seconds,
        "postprocessing_seconds": postprocessing_seconds,
        "avg_postprocessing_ms_per_image": 1000.0
        * postprocessing_seconds
        / max(image_count, 1),
        "avg_inference_ms_per_image_model_only": 1000.0
        * inference_seconds
        / max(image_count, 1),
        "images_per_second_model_only": (
            image_count / inference_seconds if inference_seconds > 0 else 0.0
        ),
        "total_test_seconds_including_loss_and_metrics": total_seconds,
    }
    for class_id, class_name in enumerate(class_names):
        class_metrics = metrics["per_class"][class_id]
        prefix = f"class_{class_id}_{safe_column_name(class_name)}"
        row[f"{prefix}_ground_truth_count"] = class_metrics["gt_count"]
        row[f"{prefix}_AP50"] = class_metrics["ap50"]
        row[f"{prefix}_AP50_95"] = class_metrics["ap50_95"]

    saved_csv = append_csv(OUTPUT_CSV, row)
    print(f"Device: {device}")
    print(f"Test images: {image_count}")
    print(f"Test loss/image: {test_loss:.5f}")
    print(
        f"Precision={metrics['precision']:.4f} Recall={metrics['recall']:.4f} F1={metrics['f1']:.4f}"
    )
    print(f"mAP@0.5={metrics['mAP50']:.4f} mAP@0.5:0.95={metrics['mAP50_95']:.4f}")
    print(f"Mean IoU of matched boxes={metrics['mean_iou_matched']:.4f}")
    print(
        f"Inference: {1000.0 * inference_seconds / max(image_count, 1):.2f} ms/image, "
        f"{image_count / inference_seconds if inference_seconds > 0 else 0.0:.2f} images/s"
    )
    print(f"CSV: {saved_csv.resolve()}")


if __name__ == "__main__":
    freeze_support()
    main()
