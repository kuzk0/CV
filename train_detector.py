from __future__ import annotations

from datetime import datetime
import csv
import math
from pathlib import Path
import random
import time
from multiprocessing import freeze_support

import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

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
CLASSES_FILE = Path("./Datasets/Plates/classes.names")
TRAIN_IMAGES = DATASET_ROOT / "images" / "train"
TRAIN_LABELS = DATASET_ROOT / "labels" / "train"

OUTPUT_DIR = Path("./outputs")
LOG_ROOT = Path("./runs/custom_cnn_detector")
EPOCHS = 30
BATCH_SIZE = 4
NUM_WORKERS = 2
VALIDATION_RATIO = 0.15
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
SCORE_THRESHOLD = 0.25
NMS_IOU_THRESHOLD = 0.50
INPUT_SIZE = DEFAULT_INPUT_SIZE
NUM_SLOTS = DEFAULT_NUM_SLOTS
RANDOM_SEED = 42
# ------------------------------------------------------------------------


def move_targets_to_device(targets, device):
    return [
        {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in target.items()
        }
        for target in targets
    ]


def evaluate_loss(model, loader, device, input_size: int) -> float:
    model.eval()
    loss_sum = 0.0
    image_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            image_batch = torch.stack(images).to(device, non_blocking=True)
            targets = move_targets_to_device(targets, device)
            raw_predictions = model(image_batch)
            losses = compute_detection_loss(raw_predictions, targets, input_size)
            loss_sum += float(losses["total"].item()) * len(images)
            image_count += len(images)
    return loss_sum / max(image_count, 1)


@torch.inference_mode()
def predict_and_evaluate(model, loader, device, num_classes: int, input_size: int):
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
            score_threshold=SCORE_THRESHOLD,
            nms_iou_threshold=NMS_IOU_THRESHOLD,
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

    metrics = compute_detection_metrics(
        predictions, targets_all, num_classes, SCORE_THRESHOLD
    )
    metrics["inference_seconds"] = inference_seconds
    metrics["postprocessing_seconds"] = postprocessing_seconds
    metrics["avg_inference_ms_per_image"] = (
        1000.0 * inference_seconds / max(image_count, 1)
    )
    metrics["images_per_second"] = (
        image_count / inference_seconds if inference_seconds > 0 else 0.0
    )
    return metrics


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    class_names = load_class_names(CLASSES_FILE)
    dataset = BoundingBoxDataset(
        TRAIN_IMAGES, TRAIN_LABELS, class_names, image_size=INPUT_SIZE
    )
    if len(dataset) < 2:
        raise ValueError(
            "Для автоматического разделения на train/validation требуется минимум 2 изображения."
        )

    validation_size = max(1, int(round(len(dataset) * VALIDATION_RATIO)))
    training_size = len(dataset) - validation_size
    if training_size < 1:
        raise ValueError(
            "Слишком большой VALIDATION_RATIO: в train не осталось изображений."
        )

    split_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_dataset, validation_dataset = random_split(
        dataset, [training_size, validation_size], generator=split_generator
    )
    loader_generator = torch.Generator().manual_seed(RANDOM_SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
        generator=loader_generator,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=detection_collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    device = get_device()
    print(f"Device: {device}")
    print(f"Classes ({len(class_names)}): {', '.join(class_names)}")
    print(
        f"Images: total={len(dataset)}, train={training_size}, validation={validation_size}"
    )
    model = create_model(len(class_names), NUM_SLOTS).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Model: CustomCNNDetector | parameters={parameter_count:,} | "
        f"input={INPUT_SIZE}x{INPUT_SIZE} | slots/cell={NUM_SLOTS} | initialized from scratch"
    )
    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, EPOCHS // 3), gamma=0.3
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = LOG_ROOT / run_id
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(log_dir))

    best_map50 = -1.0
    history_path = OUTPUT_DIR / f"training_history_{run_id}.csv"
    history_columns = [
        "epoch",
        "train_loss",
        "train_objectness_loss",
        "train_box_loss",
        "train_class_loss",
        "validation_loss",
        "validation_precision",
        "validation_recall",
        "validation_f1",
        "validation_mAP50",
        "validation_mAP50_95",
        "validation_mean_iou_matched",
        "validation_inference_ms_per_image",
        "validation_postprocessing_ms_per_image",
        "validation_images_per_second",
        "learning_rate",
        "epoch_seconds",
    ]

    try:
        with history_path.open("w", newline="", encoding="utf-8-sig") as history_file:
            history_writer = csv.DictWriter(history_file, fieldnames=history_columns)
            history_writer.writeheader()

            for epoch in range(1, EPOCHS + 1):
                epoch_start = time.perf_counter()
                model.train()
                running = {"total": 0.0, "objectness": 0.0, "box": 0.0, "class": 0.0}
                seen_images = 0

                for step, (images, targets) in enumerate(train_loader, start=1):
                    image_batch = torch.stack(images).to(device, non_blocking=True)
                    targets = move_targets_to_device(targets, device)
                    optimizer.zero_grad(set_to_none=True)
                    raw_predictions = model(image_batch)
                    losses = compute_detection_loss(
                        raw_predictions, targets, INPUT_SIZE
                    )
                    loss = losses["total"]
                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"Получена нечисловая ошибка на эпохе {epoch}, batch {step}: {loss.item()}"
                        )
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    batch_count = len(images)
                    for name in running:
                        running[name] += (
                            float(losses[name].detach().item()) * batch_count
                        )
                    seen_images += batch_count

                train_stats = {
                    name: value / max(seen_images, 1) for name, value in running.items()
                }
                validation_loss = evaluate_loss(
                    model, validation_loader, device, INPUT_SIZE
                )
                metrics = predict_and_evaluate(
                    model, validation_loader, device, len(class_names), INPUT_SIZE
                )
                scheduler.step()
                epoch_seconds = time.perf_counter() - epoch_start
                current_lr = optimizer.param_groups[0]["lr"]

                row = {
                    "epoch": epoch,
                    "train_loss": train_stats["total"],
                    "train_objectness_loss": train_stats["objectness"],
                    "train_box_loss": train_stats["box"],
                    "train_class_loss": train_stats["class"],
                    "validation_loss": validation_loss,
                    "validation_precision": metrics["precision"],
                    "validation_recall": metrics["recall"],
                    "validation_f1": metrics["f1"],
                    "validation_mAP50": metrics["mAP50"],
                    "validation_mAP50_95": metrics["mAP50_95"],
                    "validation_mean_iou_matched": metrics["mean_iou_matched"],
                    "validation_inference_ms_per_image": metrics[
                        "avg_inference_ms_per_image"
                    ],
                    "validation_postprocessing_ms_per_image": 1000.0
                    * metrics["postprocessing_seconds"]
                    / max(len(validation_dataset), 1),
                    "validation_images_per_second": metrics["images_per_second"],
                    "learning_rate": current_lr,
                    "epoch_seconds": epoch_seconds,
                }
                history_writer.writerow(row)
                history_file.flush()

                writer.add_scalar("Loss/train_total", train_stats["total"], epoch)
                writer.add_scalar(
                    "Loss/train_objectness", train_stats["objectness"], epoch
                )
                writer.add_scalar("Loss/train_box", train_stats["box"], epoch)
                writer.add_scalar("Loss/train_class", train_stats["class"], epoch)
                writer.add_scalar("Loss/validation", validation_loss, epoch)
                for metric_name in (
                    "precision",
                    "recall",
                    "f1",
                    "mAP50",
                    "mAP50_95",
                    "mean_iou_matched",
                    "avg_inference_ms_per_image",
                    "postprocessing_seconds",
                    "images_per_second",
                ):
                    value = metrics[metric_name]
                    if math.isfinite(value):
                        writer.add_scalar(f"Metrics/{metric_name}", value, epoch)
                writer.add_scalar("Training/learning_rate", current_lr, epoch)
                writer.add_scalar("Training/epoch_seconds", epoch_seconds, epoch)
                writer.flush()

                print(
                    f"Epoch {epoch:03d}/{EPOCHS} | train={train_stats['total']:.4f} "
                    f"(obj={train_stats['objectness']:.3f}, box={train_stats['box']:.3f}, "
                    f"class={train_stats['class']:.3f}) | val={validation_loss:.4f} "
                    f"P={metrics['precision']:.3f} R={metrics['recall']:.3f} F1={metrics['f1']:.3f} "
                    f"mAP50={metrics['mAP50']:.3f} mAP50-95={metrics['mAP50_95']:.3f} "
                    f"time={epoch_seconds:.1f}s"
                )

                score = metrics["mAP50"]
                if not math.isfinite(score):
                    score = 0.0
                is_best = score >= best_map50
                if is_best:
                    best_map50 = score
                checkpoint = {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "num_classes": len(class_names),
                    "architecture": "CustomCNNDetector",
                    "epoch": epoch,
                    "best_validation_mAP50": best_map50,
                    "config": {
                        "dataset_root": str(DATASET_ROOT),
                        "classes_file": str(CLASSES_FILE),
                        "epochs": EPOCHS,
                        "batch_size": BATCH_SIZE,
                        "learning_rate": LEARNING_RATE,
                        "weight_decay": WEIGHT_DECAY,
                        "validation_ratio": VALIDATION_RATIO,
                        "score_threshold": SCORE_THRESHOLD,
                        "nms_iou_threshold": NMS_IOU_THRESHOLD,
                        "input_size": INPUT_SIZE,
                        "num_slots": NUM_SLOTS,
                        "pretrained_weights": False,
                    },
                }
                torch.save(checkpoint, OUTPUT_DIR / "last_detector.pth")
                # Сохраняем best после первого прохода или когда метрика улучшилась.
                if is_best:
                    torch.save(checkpoint, OUTPUT_DIR / "best_detector.pth")
                    print(
                        f"  -> Сохранена лучшая модель: {OUTPUT_DIR / 'best_detector.pth'}"
                    )

    finally:
        writer.close()

    print("Обучение завершено.")
    print(f"Лучшая модель: {OUTPUT_DIR / 'best_detector.pth'}")
    print(f"Последняя модель: {OUTPUT_DIR / 'last_detector.pth'}")
    print(f"История CSV: {history_path}")
    print(f"TensorBoard logdir: {log_dir}")


if __name__ == "__main__":
    freeze_support()
    main()
