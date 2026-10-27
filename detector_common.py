from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_INPUT_SIZE = 320
DEFAULT_NUM_SLOTS = 3


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_class_names(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Не найден файл классов: {path.resolve()}\n"
            "Проверьте CLASSES_FILE. В classes.names должно быть по одному названию класса на строку."
        )
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"Файл классов пуст: {path.resolve()}")
    if len(set(names)) != len(names):
        raise ValueError(
            f"В файле классов есть повторяющиеся названия: {path.resolve()}"
        )
    return names


class BoundingBoxDataset(Dataset):
    """Изображения и текстовые рамки в формате: class_id x_center y_center width height.

    Координаты рамок нормализованы относительно исходного изображения и находятся в диапазоне 0..1.
    Все изображения масштабируются до квадратного image_size x image_size без сохранения пропорций.
    Разметка переводится в пиксельные координаты уже после масштабирования.
    """

    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        class_names: list[str],
        image_size: int,
    ):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.class_names = class_names
        self.image_size = int(image_size)
        if self.image_size <= 0 or self.image_size % 8 != 0:
            raise ValueError(
                "image_size должен быть положительным числом, кратным 8, например 320."
            )
        if not self.images_dir.is_dir():
            raise FileNotFoundError(
                f"Не найдена папка изображений: {self.images_dir.resolve()}"
            )
        if not self.labels_dir.is_dir():
            raise FileNotFoundError(
                f"Не найдена папка разметки: {self.labels_dir.resolve()}"
            )

        self.images = sorted(
            p
            for p in self.images_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.images:
            raise ValueError(f"В папке нет изображений: {self.images_dir.resolve()}")

        missing = []
        for image_path in self.images:
            label_path = self.labels_dir / image_path.relative_to(
                self.images_dir
            ).with_suffix(".txt")
            if not label_path.is_file():
                missing.append(label_path)
                if len(missing) >= 5:
                    break
        if missing:
            raise FileNotFoundError(
                "Не найдены одноимённые .txt-файлы разметки. Структура папок labels должна повторять images.\n"
                + "\n".join(str(path) for path in missing)
            )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        image_path = self.images[index]
        label_path = self.labels_dir / image_path.relative_to(
            self.images_dir
        ).with_suffix(".txt")
        resampling = getattr(Image, "Resampling", Image).BILINEAR
        with Image.open(image_path) as im:
            image = im.convert("RGB").resize(
                (self.image_size, self.image_size), resampling
            )

        # Преобразование PIL -> Tensor без torchvision.
        image_bytes = bytearray(image.tobytes())
        image_tensor = torch.frombuffer(image_bytes, dtype=torch.uint8)
        image_tensor = image_tensor.view(self.image_size, self.image_size, 3).permute(
            2, 0, 1
        )
        image_tensor = image_tensor.to(torch.float32).div_(255.0)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
        ):
            line = line.strip()
            if not line:
                continue  # Пустой файл означает изображение без объектов.
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(
                    f"Неверная разметка {label_path}, строка {line_number}: "
                    "ожидается class_id x_center y_center width height."
                )
            try:
                class_id = int(parts[0])
                x_center, y_center, box_width, box_height = map(float, parts[1:])
            except ValueError as exc:
                raise ValueError(
                    f"Не удалось прочитать {label_path}, строка {line_number}: {line}"
                ) from exc

            values = (x_center, y_center, box_width, box_height)
            if not all(math.isfinite(v) for v in values):
                raise ValueError(
                    f"NaN/inf в разметке {label_path}, строка {line_number}."
                )
            if not 0 <= class_id < len(self.class_names):
                raise ValueError(
                    f"class_id={class_id} вне диапазона в {label_path}, строка {line_number}; "
                    f"допустимы значения 0..{len(self.class_names) - 1}."
                )
            if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0):
                raise ValueError(
                    f"Центр рамки должен быть в диапазоне 0..1: {label_path}, строка {line_number}."
                )
            if not (box_width > 0.0 and box_height > 0.0):
                raise ValueError(
                    f"Ширина/высота рамки должны быть положительными: {label_path}, строка {line_number}."
                )

            x1 = max(0.0, x_center - box_width / 2.0)
            y1 = max(0.0, y_center - box_height / 2.0)
            x2 = min(1.0, x_center + box_width / 2.0)
            y2 = min(1.0, y_center + box_height / 2.0)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Пустая/некорректная рамка: {label_path}, строка {line_number}."
                )

            scale = float(self.image_size)
            boxes.append([x1 * scale, y1 * scale, x2 * scale, y2 * scale])
            labels.append(
                class_id
            )  # Классы начинаются с 0; отдельной метки background нет.

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([index], dtype=torch.int64),
        }
        return image_tensor, target


def detection_collate_fn(batch):
    # Рамки имеют разную длину у разных изображений, поэтому объединяем их отдельно от изображений.
    return tuple(zip(*batch))


class ConvBlock(nn.Module):
    """Обычный свёрточный блок, написанный из базовых слоёв PyTorch."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNNDetector(nn.Module):
    """Самостоятельный CNN-детектор: backbone, свёрточная голова и grid-предсказания.

    Архитектура не загружает сторонние модели или веса. Для каждой ячейки сетки модель
    предсказывает num_slots рамок; каждая рамка содержит objectness, x/y смещение,
    ширину/высоту и оценки классов.
    """

    def __init__(self, num_classes: int, num_slots: int = DEFAULT_NUM_SLOTS):
        super().__init__()
        if num_classes < 1:
            raise ValueError("num_classes должен быть >= 1.")
        if num_slots < 1:
            raise ValueError("num_slots должен быть >= 1.")
        self.num_classes = int(num_classes)
        self.num_slots = int(num_slots)
        self.attributes_per_box = 5 + self.num_classes

        self.features = nn.Sequential(
            ConvBlock(3, 32, stride=2),  # H/2
            ConvBlock(32, 48, stride=2),  # H/4
            ConvBlock(48, 80, stride=2),  # H/8 — при 320x320 получаем сетку 40x40
            ConvBlock(80, 128, stride=1),
            ConvBlock(128, 128, stride=1),
        )
        self.head = nn.Sequential(
            ConvBlock(128, 128, stride=1),
            nn.Conv2d(128, self.num_slots * self.attributes_per_box, kernel_size=1),
        )
        self._initialize_detection_head()

    def _initialize_detection_head(self) -> None:
        final_conv = self.head[-1]
        nn.init.normal_(final_conv.weight, mean=0.0, std=0.01)
        nn.init.zeros_(final_conv.bias)
        # Начальное objectness ниже 0.5 уменьшает поток случайных детекций в начале обучения.
        for slot in range(self.num_slots):
            base = slot * self.attributes_per_box
            final_conv.bias.data[base] = -2.0
            # softplus(приблизительно 1.85) ~= 2 клетки по ширине и высоте.
            final_conv.bias.data[base + 3] = 1.85
            final_conv.bias.data[base + 4] = 1.85

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError(
                "Ожидается батч изображений формы [batch, 3, height, width]."
            )
        if images.shape[-1] % 8 or images.shape[-2] % 8:
            raise ValueError("Высота и ширина входа должны быть кратны 8.")
        features = self.features(images)
        prediction = self.head(features)
        batch_size, _, grid_h, grid_w = prediction.shape
        prediction = (
            prediction.view(
                batch_size, self.num_slots, self.attributes_per_box, grid_h, grid_w
            )
            .permute(0, 3, 4, 1, 2)
            .contiguous()
        )
        return prediction  # [B, grid_h, grid_w, slots, 5 + num_classes]


def create_model(
    num_classes: int, num_slots: int = DEFAULT_NUM_SLOTS
) -> CustomCNNDetector:
    """Создаёт только описанную выше собственную CNN, с нуля."""
    return CustomCNNDetector(num_classes=num_classes, num_slots=num_slots)


def compute_detection_loss(
    predictions: torch.Tensor,
    targets: tuple[dict[str, torch.Tensor], ...] | list[dict[str, torch.Tensor]],
    input_size: int,
    negative_objectness_weight: float = 0.25,
    box_loss_weight: float = 5.0,
) -> dict[str, torch.Tensor]:
    """Loss objectness + координаты рамки + класс для grid-детектора."""
    if predictions.ndim != 5:
        raise ValueError(
            "predictions должны иметь форму [B, grid_h, grid_w, slots, attributes]."
        )
    batch_size, grid_h, grid_w, num_slots, attributes = predictions.shape
    num_classes = attributes - 5
    if num_classes < 1:
        raise ValueError("В prediction tensor отсутствуют выходы классов.")
    if len(targets) != batch_size:
        raise ValueError("Количество targets должно совпадать с размером батча.")

    device = predictions.device
    dtype = predictions.dtype
    objectness_target = torch.zeros(
        (batch_size, grid_h, grid_w, num_slots), device=device, dtype=dtype
    )
    box_target = torch.zeros(
        (batch_size, grid_h, grid_w, num_slots, 4), device=device, dtype=dtype
    )
    class_target = torch.full(
        (batch_size, grid_h, grid_w, num_slots), -1, device=device, dtype=torch.long
    )

    for batch_index, target in enumerate(targets):
        boxes = target["boxes"].to(device=device, dtype=dtype)
        labels = target["labels"].to(device=device, dtype=torch.long)
        if boxes.numel() == 0:
            continue
        if torch.any(labels < 0) or torch.any(labels >= num_classes):
            raise ValueError("Метка класса вне диапазона выходной головы модели.")

        x1, y1, x2, y2 = boxes.unbind(dim=1)
        center_x = ((x1 + x2) * 0.5 / input_size).clamp(0.0, 1.0 - 1e-7)
        center_y = ((y1 + y2) * 0.5 / input_size).clamp(0.0, 1.0 - 1e-7)
        width = ((x2 - x1) / input_size).clamp(min=1e-7)
        height = ((y2 - y1) / input_size).clamp(min=1e-7)
        cell_x = torch.floor(center_x * grid_w).long().clamp(0, grid_w - 1)
        cell_y = torch.floor(center_y * grid_h).long().clamp(0, grid_h - 1)
        areas = width * height

        # Сначала назначаем рамки в одну ячейку по убыванию площади.
        # Если объектов в ячейке больше слотов, явно останавливаем обучение вместо молчаливой потери меток.
        groups: dict[tuple[int, int], list[int]] = {}
        for object_index in range(len(boxes)):
            key = (int(cell_y[object_index].item()), int(cell_x[object_index].item()))
            groups.setdefault(key, []).append(object_index)
        for (gy, gx), object_indices in groups.items():
            object_indices.sort(
                key=lambda item: float(areas[item].item()), reverse=True
            )
            if len(object_indices) > num_slots:
                raise ValueError(
                    f"В изображении из батча {batch_index} найдено {len(object_indices)} объектов "
                    f"с центром в одной ячейке сетки ({gx}, {gy}), но NUM_SLOTS={num_slots}. "
                    "Увеличьте NUM_SLOTS или уменьшите масштаб сетки/изображения."
                )
            for slot, object_index in enumerate(object_indices):
                objectness_target[batch_index, gy, gx, slot] = 1.0
                box_target[batch_index, gy, gx, slot, 0] = (
                    center_x[object_index] * grid_w - gx
                )
                box_target[batch_index, gy, gx, slot, 1] = (
                    center_y[object_index] * grid_h - gy
                )
                box_target[batch_index, gy, gx, slot, 2] = width[object_index] * grid_w
                box_target[batch_index, gy, gx, slot, 3] = height[object_index] * grid_h
                class_target[batch_index, gy, gx, slot] = labels[object_index]

    objectness_logits = predictions[..., 0]
    positive_mask = objectness_target > 0.5
    negative_mask = ~positive_mask
    objectness_raw_loss = F.binary_cross_entropy_with_logits(
        objectness_logits, objectness_target, reduction="none"
    )
    if positive_mask.any():
        positive_objectness_loss = objectness_raw_loss[positive_mask].mean()
    else:
        positive_objectness_loss = objectness_logits.sum() * 0.0
    if negative_mask.any():
        negative_objectness_loss = objectness_raw_loss[negative_mask].mean()
    else:
        negative_objectness_loss = objectness_logits.sum() * 0.0
    objectness_loss = (
        positive_objectness_loss + negative_objectness_weight * negative_objectness_loss
    )

    if positive_mask.any():
        predicted_boxes = torch.cat(
            (torch.sigmoid(predictions[..., 1:3]), F.softplus(predictions[..., 3:5])),
            dim=-1,
        )
        box_loss = F.smooth_l1_loss(
            predicted_boxes[positive_mask], box_target[positive_mask], reduction="mean"
        )
        class_logits = predictions[..., 5:][positive_mask]
        class_labels = class_target[positive_mask]
        class_loss = F.cross_entropy(class_logits, class_labels)
    else:
        box_loss = predictions[..., 1:5].sum() * 0.0
        class_loss = predictions[..., 5:].sum() * 0.0

    total_loss = objectness_loss + box_loss_weight * box_loss + class_loss
    return {
        "total": total_loss,
        "objectness": objectness_loss,
        "box": box_loss,
        "class": class_loss,
        "positive_cells": positive_mask.sum().to(dtype=torch.float32),
    }


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Парный по всем комбинациям IoU, написанный на базовых операциях torch."""
    boxes1 = boxes1.to(torch.float32).reshape(-1, 4)
    boxes2 = boxes2.to(device=boxes1.device, dtype=torch.float32).reshape(-1, 4)
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros(
            (len(boxes1), len(boxes2)), dtype=torch.float32, device=boxes1.device
        )

    top_left = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection_wh = (bottom_right - top_left).clamp(min=0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-8)


def non_max_suppression(
    boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float, max_detections: int
) -> torch.Tensor:
    """NMS без torchvision.ops: оставляет рамки с максимальными оценками."""
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    order = torch.argsort(scores, descending=True)
    kept: list[torch.Tensor] = []
    while order.numel() > 0 and len(kept) < max_detections:
        current = order[0]
        kept.append(current)
        if order.numel() == 1:
            break
        rest = order[1:]
        ious = box_iou(boxes[current].reshape(1, 4), boxes[rest]).reshape(-1)
        order = rest[ious <= iou_threshold]
    return (
        torch.stack(kept).to(torch.long)
        if kept
        else torch.empty(0, dtype=torch.long, device=boxes.device)
    )


@torch.inference_mode()
def decode_predictions(
    raw_predictions: torch.Tensor,
    input_size: int,
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    max_candidates: int = 300,
    max_detections: int = 100,
) -> list[dict[str, torch.Tensor]]:
    """Преобразует сеточные выходы в рамки xyxy, оценки и нулевые class_id."""
    batch_size, grid_h, grid_w, num_slots, attributes = raw_predictions.shape
    num_classes = attributes - 5
    device = raw_predictions.device
    y_grid, x_grid = torch.meshgrid(
        torch.arange(grid_h, device=device, dtype=raw_predictions.dtype),
        torch.arange(grid_w, device=device, dtype=raw_predictions.dtype),
        indexing="ij",
    )
    x_grid = x_grid[None, :, :, None]
    y_grid = y_grid[None, :, :, None]

    center_x = (x_grid + torch.sigmoid(raw_predictions[..., 1])) / grid_w * input_size
    center_y = (y_grid + torch.sigmoid(raw_predictions[..., 2])) / grid_h * input_size
    width = (
        F.softplus(raw_predictions[..., 3]).clamp(max=grid_w * 2) / grid_w * input_size
    )
    height = (
        F.softplus(raw_predictions[..., 4]).clamp(max=grid_h * 2) / grid_h * input_size
    )
    boxes = torch.stack(
        (
            (center_x - width * 0.5).clamp(0, input_size),
            (center_y - height * 0.5).clamp(0, input_size),
            (center_x + width * 0.5).clamp(0, input_size),
            (center_y + height * 0.5).clamp(0, input_size),
        ),
        dim=-1,
    )
    class_probabilities = torch.softmax(raw_predictions[..., 5:], dim=-1)
    class_confidence, class_ids = class_probabilities.max(dim=-1)
    scores = torch.sigmoid(raw_predictions[..., 0]) * class_confidence

    decoded: list[dict[str, torch.Tensor]] = []
    for batch_index in range(batch_size):
        image_boxes = boxes[batch_index].reshape(-1, 4)
        image_scores = scores[batch_index].reshape(-1)
        image_labels = class_ids[batch_index].reshape(-1).to(torch.int64)
        valid = (
            (image_scores >= score_threshold)
            & (image_boxes[:, 2] > image_boxes[:, 0])
            & (image_boxes[:, 3] > image_boxes[:, 1])
        )
        candidate_indices = torch.where(valid)[0]
        if candidate_indices.numel() == 0:
            decoded.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device),
                }
            )
            continue
        if candidate_indices.numel() > max_candidates:
            top = torch.topk(image_scores[candidate_indices], k=max_candidates).indices
            candidate_indices = candidate_indices[top]
        image_boxes = image_boxes[candidate_indices]
        image_scores = image_scores[candidate_indices]
        image_labels = image_labels[candidate_indices]

        kept_by_class: list[torch.Tensor] = []
        for class_id in image_labels.unique():
            class_indices = torch.where(image_labels == class_id)[0]
            keep_local = non_max_suppression(
                image_boxes[class_indices],
                image_scores[class_indices],
                nms_iou_threshold,
                max_detections,
            )
            if keep_local.numel():
                kept_by_class.append(class_indices[keep_local])
        if kept_by_class:
            keep_indices = torch.cat(kept_by_class)
            keep_indices = keep_indices[
                torch.argsort(image_scores[keep_indices], descending=True)[
                    :max_detections
                ]
            ]
            decoded.append(
                {
                    "boxes": image_boxes[keep_indices].to(torch.float32),
                    "scores": image_scores[keep_indices].to(torch.float32),
                    "labels": image_labels[keep_indices].to(torch.int64),
                }
            )
        else:
            decoded.append(
                {
                    "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
                    "scores": torch.empty((0,), dtype=torch.float32, device=device),
                    "labels": torch.empty((0,), dtype=torch.int64, device=device),
                }
            )
    return decoded


def _average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    recall = torch.tensor([0.0, *recalls, 1.0], dtype=torch.float64)
    precision = torch.tensor([0.0, *precisions, 0.0], dtype=torch.float64)
    for i in range(precision.numel() - 2, -1, -1):
        precision[i] = torch.maximum(precision[i], precision[i + 1])
    change_points = torch.where(recall[1:] != recall[:-1])[0]
    ap = (
        (recall[change_points + 1] - recall[change_points])
        * precision[change_points + 1]
    ).sum()
    return float(ap.item())


def _class_ap(
    pred_records: list[dict[str, torch.Tensor]],
    gt_records: list[dict[str, torch.Tensor]],
    class_id: int,
    iou_threshold: float,
) -> tuple[float, int]:
    gt_by_image: dict[int, torch.Tensor] = {}
    total_gt = 0
    for image_index, gt in enumerate(gt_records):
        class_boxes = gt["boxes"][gt["labels"] == class_id]
        gt_by_image[image_index] = class_boxes
        total_gt += len(class_boxes)
    if total_gt == 0:
        return float("nan"), 0

    entries: list[tuple[float, int, torch.Tensor]] = []
    for image_index, pred in enumerate(pred_records):
        mask = pred["labels"] == class_id
        entries.extend(
            (float(score.item()), image_index, box)
            for box, score in zip(pred["boxes"][mask], pred["scores"][mask])
        )
    entries.sort(key=lambda item: item[0], reverse=True)
    if not entries:
        return 0.0, total_gt

    matched: dict[int, set[int]] = {i: set() for i in range(len(gt_records))}
    tps: list[float] = []
    fps: list[float] = []
    for _score, image_index, pred_box in entries:
        gt_boxes = gt_by_image[image_index]
        if len(gt_boxes) == 0:
            tps.append(0.0)
            fps.append(1.0)
            continue
        ious = box_iou(pred_box.reshape(1, 4), gt_boxes).reshape(-1)
        found_match = False
        for gt_index in torch.argsort(ious, descending=True).tolist():
            if float(ious[gt_index].item()) < iou_threshold:
                break
            if gt_index not in matched[image_index]:
                matched[image_index].add(gt_index)
                found_match = True
                break
        tps.append(1.0 if found_match else 0.0)
        fps.append(0.0 if found_match else 1.0)

    tp_cum = torch.tensor(tps, dtype=torch.float64).cumsum(0)
    fp_cum = torch.tensor(fps, dtype=torch.float64).cumsum(0)
    recalls = (tp_cum / total_gt).tolist()
    precisions = (tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-12)).tolist()
    return _average_precision(recalls, precisions), total_gt


def _threshold_counts(
    pred_records: list[dict[str, torch.Tensor]],
    gt_records: list[dict[str, torch.Tensor]],
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    tp_total = fp_total = fn_total = 0
    matched_ious: list[float] = []
    gt_total = sum(len(gt["boxes"]) for gt in gt_records)

    for pred, gt in zip(pred_records, gt_records):
        class_ids = set(gt["labels"].tolist()) | set(pred["labels"].tolist())
        for class_id in class_ids:
            gt_boxes = gt["boxes"][gt["labels"] == class_id]
            pred_mask = (pred["labels"] == class_id) & (
                pred["scores"] >= score_threshold
            )
            pred_boxes = pred["boxes"][pred_mask]
            pred_scores = pred["scores"][pred_mask]
            if len(pred_boxes):
                pred_boxes = pred_boxes[torch.argsort(pred_scores, descending=True)]
            matched_gt: set[int] = set()
            tp_class = 0
            for pred_box in pred_boxes:
                if len(gt_boxes) == 0:
                    continue
                ious = box_iou(pred_box.reshape(1, 4), gt_boxes).reshape(-1)
                for gt_index in torch.argsort(ious, descending=True).tolist():
                    iou_value = float(ious[gt_index].item())
                    if iou_value < iou_threshold:
                        break
                    if gt_index not in matched_gt:
                        matched_gt.add(gt_index)
                        tp_class += 1
                        matched_ious.append(iou_value)
                        break
            fp_class = len(pred_boxes) - tp_class
            fn_class = len(gt_boxes) - tp_class
            tp_total += tp_class
            fp_total += fp_class
            fn_total += fn_class

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "ground_truth_objects": gt_total,
        "mean_iou_matched": (
            sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
        ),
    }


def compute_detection_metrics(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    num_classes: int,
    score_threshold: float = 0.25,
) -> dict[str, Any]:
    """Precision/Recall/F1 at a fixed threshold and mAP at IoU 0.5:0.95."""
    pred_records = [
        {
            "boxes": pred["boxes"].detach().cpu().to(torch.float32),
            "scores": pred["scores"].detach().cpu().to(torch.float32),
            "labels": pred["labels"].detach().cpu().to(torch.int64),
        }
        for pred in predictions
    ]
    gt_records = [
        {
            "boxes": target["boxes"].detach().cpu().to(torch.float32),
            "labels": target["labels"].detach().cpu().to(torch.int64),
        }
        for target in targets
    ]

    iou_thresholds = [round(0.50 + 0.05 * i, 2) for i in range(10)]
    per_class: dict[int, dict[str, float | int]] = {}
    for class_id in range(num_classes):
        ap50, gt_count = _class_ap(pred_records, gt_records, class_id, 0.5)
        class_aps = []
        for threshold in iou_thresholds:
            ap, threshold_gt_count = _class_ap(
                pred_records, gt_records, class_id, threshold
            )
            if threshold_gt_count > 0 and not math.isnan(ap):
                class_aps.append(ap)
        per_class[class_id] = {
            "gt_count": gt_count,
            "ap50": ap50,
            "ap50_95": sum(class_aps) / len(class_aps) if class_aps else float("nan"),
        }

    ap50_values = [
        value["ap50"] for value in per_class.values() if not math.isnan(value["ap50"])
    ]
    ap5095_values = [
        value["ap50_95"]
        for value in per_class.values()
        if not math.isnan(value["ap50_95"])
    ]
    return {
        **_threshold_counts(pred_records, gt_records, score_threshold),
        "mAP50": sum(ap50_values) / len(ap50_values) if ap50_values else float("nan"),
        "mAP50_95": (
            sum(ap5095_values) / len(ap5095_values) if ap5095_values else float("nan")
        ),
        "per_class": per_class,
    }
