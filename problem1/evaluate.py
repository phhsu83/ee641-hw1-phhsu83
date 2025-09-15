# evaluate.py  — 完整可執行版本

import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from typing import Optional, Sequence, Dict, Any, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ShapeDetectionDataset
from model import MultiScaleDetector      
from loss import DetectionLoss       
from utils import generate_anchors

from utils import compute_iou, generate_anchors # 需為：接收 [N,4]、[M,4] Tensor，回傳 IoU 矩陣
from PIL import Image, ImageDraw, ImageFont



# ------------------------------------------------------------
# 1) 單一類別 AP 計算
# ------------------------------------------------------------
def compute_ap(predictions, ground_truths, iou_threshold: float = 0.5):
    """
    計算單一類別的 AP。
    predictions: list[[x1,y1,x2,y2,score]]
    ground_truths: list[[x1,y1,x2,y2]]
    回傳：ap(float)
    """
    if len(ground_truths) == 0:
        return 0.0

    # 依分數排序
    predictions = sorted(predictions, key=lambda x: x[4], reverse=True)
    tp = np.zeros(len(predictions), dtype=np.float32)
    fp = np.zeros(len(predictions), dtype=np.float32)
    matched = [False] * len(ground_truths)

    for i, pred in enumerate(predictions):
        best_iou, best_gt = 0.0, -1
        pb = torch.tensor(pred[:4], dtype=torch.float32).unsqueeze(0)  # [1,4]
        for j, gt in enumerate(ground_truths):
            gb = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)     # [1,4]
            iou = compute_iou(pb, gb).item()  # 取純量
            if iou > best_iou:
                best_iou, best_gt = iou, j

        if best_iou >= iou_threshold and best_gt >= 0 and not matched[best_gt]:
            tp[i] = 1.0
            matched[best_gt] = True
        else:
            fp[i] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    recalls = tp_cum / max(1, len(ground_truths))
    precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)

    # 讓 precision 隨 recall 單調不升
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]
    ap = np.trapz(precisions, recalls)
    return float(ap)



def visualize_detections(
    image,
    predictions: Dict[str, Any],
    ground_truths: Dict[str, Any],
    save_path,
    class_names: Optional[Sequence[str]] = None,
):
    """
    Visualize predictions and ground truth boxes (PIL 版本).

    支援：
      - image: str (檔案路徑) 或 PIL.Image.Image
      - predictions: dict，需包含
          {"boxes": Tensor/list[[x1,y1,x2,y2]], "scores": Tensor/list[float], "labels": Tensor/list[int]}
      - ground_truths: dict，需包含
          {"boxes": Tensor/list[[x1,y1,x2,y2]], "labels": Tensor/list[int]}
      - save_path: 儲存路徑 (str/Path)
      - class_names: 類別名稱列表（預設 ["circle","square","triangle"]）
    """
    # --- 讀圖 ---
    if isinstance(image, str):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, Image.Image):
        img = image.copy()
    else:
        raise ValueError("image 必須是路徑字串或 PIL.Image.Image")

    draw = ImageDraw.Draw(img, "RGBA")

    # --- 類別名稱 ---
    if class_names is None:
        class_names = ["circle", "square", "triangle"]

    # --- 安全轉 list 的小工具 ---
    def _to_list(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        return x

    # --- 取 GT ---
    gt_boxes = _to_list(ground_truths.get("boxes", [])) or []
    gt_labels = _to_list(ground_truths.get("labels", [])) or []
    if len(gt_labels) == 0 and len(gt_boxes) > 0:
        # 若沒提供 labels，就填 None（不顯示類別文字）
        gt_labels = [None] * len(gt_boxes)

    # --- 取 Pred ---
    pred_boxes = _to_list(predictions.get("boxes", [])) or []
    pred_scores = _to_list(predictions.get("scores", [])) or []
    pred_labels = _to_list(predictions.get("labels", [])) or []
    if len(pred_labels) == 0 and len(pred_boxes) > 0:
        pred_labels = [None] * len(pred_boxes)
    if len(pred_scores) < len(pred_boxes):
        # 補齊分數長度，避免 zip 掉資料
        pred_scores = list(pred_scores) + [0.0] * (len(pred_boxes) - len(pred_scores))

    # --- 畫框 helper ---
    def draw_box(b, color, width=2, text=None, fill=None):
        x1, y1, x2, y2 = [float(v) for v in b]
        # 邊框
        for k in range(width):
            draw.rectangle([x1 - k, y1 - k, x2 + k, y2 + k], outline=color)
        # 透明填色
        if fill is not None:
            draw.rectangle([x1, y1, x2, y2], outline=color, fill=fill)
        # 文字
        if text:
            # 讓文字不要貼邊
            draw.text((x1 + 2, y1 + 2), text, fill=color)

    # --- 畫 GT（綠）---
    for b, lb in zip(gt_boxes, gt_labels):
        txt = None
        if lb is not None and 0 <= int(lb) < len(class_names):
            txt = f"GT:{class_names[int(lb)]}"
        draw_box(b, color=(0, 200, 0, 255), width=2, text=txt)

    # --- 畫 Pred（紅，半透明填色）---
    for b, sc, lb in zip(pred_boxes, pred_scores, pred_labels):
        txt = None
        if lb is not None and 0 <= int(lb) < len(class_names):
            txt = f"{class_names[int(lb)]} {float(sc):.2f}"
        else:
            txt = f"{float(sc):.2f}"
        draw_box(b, color=(255, 50, 50, 255), width=2, text=txt, fill=(255, 0, 0, 40))

    # --- 存檔 ---
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(str(save_path))




def _to_cxcywh(boxes: torch.Tensor):
    """將 (x1,y1,x2,y2) 轉 (cx,cy,w,h)；boxes: [N,4] 或 [*,4]"""
    x1, y1, x2, y2 = boxes.unbind(-1)
    w = (x2 - x1).clamp(min=1e-6)
    h = (y2 - y1).clamp(min=1e-6)
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h
    return cx, cy, w, h

def decode_predictions(predictions, anchors, num_classes: int = 3, conf_thresh: float = 0.3):
    """
    將 raw 模型輸出解碼成每張圖的偵測結果（NMS 前）：
      {
        "boxes": Tensor[K,4]  (x1,y1,x2,y2),
        "scores": Tensor[K],  # objectness * softmax(class)
        "labels": Tensor[K],  # 0..C-1
        "scale_ids": Tensor[K]  # 0/1/2 表示來自哪個 scale head
      }

    期待每個 scale 的張量 shape 為 [B, A*(5+C), H, W]，通道順序：
      [tx, ty, tw, th, objectness, class_1..class_C]
    anchors[scale] 為展平後 [H*W*A, 4] 的 (x1,y1,x2,y2)。
    """
    device = predictions[0].device
    B = predictions[0].shape[0]
    out = [{"boxes": [], "scores": [], "labels": [], "scale_ids": []} for _ in range(B)]

    for s_id, (pred_s, anc_s) in enumerate(zip(predictions, anchors)):
        B, ch, H, W = pred_s.shape
        A_per = ch // (5 + num_classes)

        # [B, H*W*A, 5+C]
        pred_s = (
            pred_s
            .reshape(B, A_per, 5 + num_classes, H, W)
            .permute(0, 3, 4, 1, 2)
            .reshape(B, -1, 5 + num_classes)
        )

        # split
        loc = pred_s[..., 0:4].contiguous()                      # [B,N,4] (tx,ty,tw,th)
        obj = pred_s[..., 4].contiguous()                        # [B,N]
        cls = pred_s[..., 5:5 + num_classes].contiguous()        # [B,N,C]

        anc_s = anc_s.to(device)                                  # [N,4]
        N = anc_s.shape[0]
        assert N == loc.shape[1], f"anchors({N}) 與輸出長度({loc.shape[1]})不一致"

        # anchor 轉 cx,cy,w,h
        a_cx, a_cy, a_w, a_h = _to_cxcywh(anc_s)                 # [N]

        # decode to (x1,y1,x2,y2)
        tx, ty, tw, th = loc.unbind(-1)                          # [B,N]
        p_cx = a_cx.unsqueeze(0) + tx * a_w.unsqueeze(0)
        p_cy = a_cy.unsqueeze(0) + ty * a_h.unsqueeze(0)
        p_w  = a_w.unsqueeze(0) * torch.exp(tw)
        p_h  = a_h.unsqueeze(0) * torch.exp(th)

        x1 = p_cx - 0.5 * p_w
        y1 = p_cy - 0.5 * p_h
        x2 = p_cx + 0.5 * p_w
        y2 = p_cy + 0.5 * p_h
        boxes = torch.stack([x1, y1, x2, y2], dim=-1)            # [B,N,4]

        # scores = objectness * softmax(class)
        obj_prob = torch.sigmoid(obj)                            # [B,N]
        cls_prob = F.softmax(cls, dim=-1)                        # [B,N,C]
        comb = obj_prob.unsqueeze(-1) * cls_prob                 # [B,N,C]
        best_scores, best_cls = comb.max(dim=-1)                 # [B,N], [B,N]

        # 蒐集 >= 門檻的偵測
        for b in range(B):
            keep = best_scores[b] >= conf_thresh
            if keep.any():
                out[b]["boxes"].append(boxes[b][keep])
                out[b]["scores"].append(best_scores[b][keep].float())
                out[b]["labels"].append(best_cls[b][keep].to(torch.long))
                out[b]["scale_ids"].append(
                    torch.full((int(keep.sum()),), s_id, dtype=torch.long, device=device)
                )

    # 合併各 scale 的結果
    for b in range(B):
        if len(out[b]["boxes"]) == 0:
            out[b] = {
                "boxes": torch.zeros((0, 4), device=device),
                "scores": torch.zeros((0,), device=device),
                "labels": torch.zeros((0,), dtype=torch.long, device=device),
                "scale_ids": torch.zeros((0,), dtype=torch.long, device=device),
            }
        else:
            out[b] = {
                "boxes": torch.cat(out[b]["boxes"], dim=0),
                "scores": torch.cat(out[b]["scores"], dim=0),
                "labels": torch.cat(out[b]["labels"], dim=0),
                "scale_ids": torch.cat(out[b]["scale_ids"], dim=0),
            }
    return out



# ------------------------------------------------------------
# 5) 簡易 per-class（或單一類別）NMS
# ------------------------------------------------------------
def _apply_nms_per_class(det: Dict[str, torch.Tensor],
                         iou_thresh: float = 0.5,
                         topk_per_class: Optional[int] = 200) -> Dict[str, torch.Tensor]:
    """
    對單張圖片的 decode 結果做 per-class（或單一類別）NMS。
    """
    boxes, scores, labels, scale_ids = det["boxes"], det["scores"], det["labels"], det["scale_ids"]
    device = boxes.device
    kept_b, kept_s, kept_l, kept_sc = [], [], [], []

    uniq = labels.unique().tolist() if labels.numel() > 0 else []
    for c in uniq:
        m = (labels == c)
        b, s, sc = boxes[m], scores[m], scale_ids[m]
        if b.numel() == 0:
            continue
        order = torch.argsort(s, descending=True)
        if topk_per_class is not None:
            order = order[:topk_per_class]
        b, s, sc = b[order], s[order], sc[order]

        keep_idx = []
        suppressed = torch.zeros(b.shape[0], dtype=torch.bool, device=device)
        for i in range(b.shape[0]):
            if suppressed[i]:
                continue
            keep_idx.append(i)
            if i + 1 < b.shape[0]:
                ious = compute_iou(b[i].unsqueeze(0), b[i+1:]).squeeze(0)
                suppressed[i+1:] |= (ious >= iou_thresh)
        kept_b.append(b[keep_idx])
        kept_s.append(s[keep_idx])
        kept_l.append(torch.full((len(keep_idx),), int(c), dtype=torch.long, device=device))
        kept_sc.append(sc[keep_idx])

    if len(kept_b) == 0:
        return {
            "boxes": torch.zeros((0, 4), device=device),
            "scores": torch.zeros((0,), device=device),
            "labels": torch.zeros((0,), dtype=torch.long, device=device),
            "scale_ids": torch.zeros((0,), dtype=torch.long, device=device),
        }
    return {
        "boxes": torch.cat(kept_b, dim=0),
        "scores": torch.cat(kept_s, dim=0),
        "labels": torch.cat(kept_l, dim=0),
        "scale_ids": torch.cat(kept_sc, dim=0),
    }



def visualize_anchor_coverage(
    image,
    gt_boxes,
    anchors_per_scale,
    save_dir,
    iou_thr: float = 0.5,
):
    """
    在單張影像上視覺化「每個 scale 的 anchor 覆蓋情況」：
      - 對每個 scale，計算每個 anchor 與任一 GT 的最大 IoU
      - 在影像上畫出 anchor 中心點：IoU >= iou_thr 視為覆蓋（高亮），否則為未覆蓋（淡色）

    參數：
      image : str（路徑）或 PIL.Image.Image 或 numpy.ndarray(H,W,3)
      gt_boxes : [[x1,y1,x2,y2], ...] 或 Tensor[G,4]
      anchors_per_scale : List[Tensor[A_i,4]]（各 scale 展平的 anchors，座標為 [x1,y1,x2,y2]）
      save_dir : 輸出資料夾路徑
      iou_thr : 覆蓋門檻（預設 0.5）

    產出：
      save_dir/anchor_coverage_scale_{i}.png  （i = 1..num_scales）

    依賴：
      需有 compute_iou(boxes1, boxes2) -> Tensor[A,G] 可用
    """
   

    # ----- 小工具：把各種輸入轉成 PIL.Image -----
    def _as_pil(img_any) -> Image.Image:
        if isinstance(img_any, str):
            return Image.open(img_any).convert("RGB")
        if isinstance(img_any, Image.Image):
            return img_any.copy().convert("RGB")
        if isinstance(img_any, np.ndarray):
            arr = img_any
            if arr.ndim == 2:  # 灰階轉 RGB
                arr = np.stack([arr] * 3, axis=-1)
            if arr.dtype != np.uint8:
                # 嘗試把 float/其它型別正規化到 uint8
                if arr.max() <= 1.0:
                    arr = np.clip(arr, 0, 1)
                    arr = (arr * 255.0 + 0.5).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise ValueError("image 必須是路徑字串、PIL.Image 或 numpy.ndarray")


    # ----- 準備影像 / 路徑 -----
    img = _as_pil(image)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- 準備裝置 / GT -----
    device = anchors_per_scale[0].device if isinstance(anchors_per_scale[0], torch.Tensor) else torch.device("cpu")
    if isinstance(gt_boxes, torch.Tensor):
        gtb = gt_boxes.to(device).float()
    else:
        gtb = torch.as_tensor(gt_boxes, dtype=torch.float32, device=device) if len(gt_boxes) > 0 \
              else torch.zeros((0, 4), dtype=torch.float32, device=device)

    # ----- 逐 scale 視覺化 -----
    for s_id, anc in enumerate(anchors_per_scale):
        anc_t = anc.to(device).float()  # [A,4]

        # 計算覆蓋（以最大 IoU 判斷）
        if gtb.numel() > 0:
            ious = compute_iou(anc_t, gtb)                 # [A, G]
            cover = (ious.max(dim=1).values >= iou_thr)    # [A]
        else:
            cover = torch.zeros((anc_t.shape[0],), dtype=torch.bool, device=device)

        # anchor 中心座標
        cx, cy, _, _ = _to_cxcywh(anc_t)
        cx = cx.detach().cpu().numpy()
        cy = cy.detach().cpu().numpy()
        cover_np = cover.detach().cpu().numpy()

        # 作圖
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.scatter(cx[~cover_np], cy[~cover_np], s=1, alpha=0.2, label="no-match")
        plt.scatter(cx[cover_np],  cy[cover_np],  s=2, alpha=0.9, label=f"IoU≥{iou_thr:g}")
        plt.title(f"Anchor coverage (Scale {s_id + 1})")
        plt.legend(markerscale=4)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_dir / f"anchor_coverage_scale_{s_id + 1}.png", dpi=180)
        plt.close()
        

def analyze_scale_performance(model, dataloader, anchors):
    """
    Analyze which scales detect which object sizes (使用回歸後的預測框).
    產出：
      - results/visualizations/scale_performance.png
      - results/visualizations/scale_stats.json
    回傳：dict，如 {"scale_1": {"small": x, "medium": y, "large": z}, ...}
    """
    device = next(model.parameters()).device
    save_dir = Path("results/visualizations")
    save_dir.mkdir(parents=True, exist_ok=True)

    # 依 bbox 的 sqrt(area) 分桶
    def size_bucket(box):
        x1, y1, x2, y2 = [float(v) for v in box]
        side = max(1e-6, np.sqrt(max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))))
        if side <= 40:
            return "small"
        elif side <= 96:
            return "medium"
        else:
            return "large"

    # 根據 anchors 長度自動決定 scale 數
    num_scales = len(anchors)
    scales = list(range(num_scales))
    buckets = ["small", "medium", "large"]
    stats = {(s, b): 0 for s in scales for b in buckets}

    model_was_training = model.training
    model.eval()

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # 使用呼叫者提供的 anchors（不在函式中重建）
            decoded = decode_predictions(images=images) if callable(getattr(decode_predictions, "__call__", None)) and "images" in decode_predictions.__code__.co_varnames else decode_predictions(
                model(images), anchors, num_classes=3, conf_thresh=0.3
            )
            # 若你的 decode_predictions 簽名不同，請依實際情況調整上一行

            for det_b, t in zip(decoded, targets):
                kept = _apply_nms_per_class(det_b, iou_thresh=0.5, topk_per_class=200)

                # 沒 GT 或沒預測就跳過
                if len(t["boxes"]) == 0 or kept["boxes"].numel() == 0:
                    continue

                gt_boxes = t["boxes"].to(device)  # [G,4]
                ious = compute_iou(kept["boxes"], gt_boxes)  # [D, G]
                best_iou, gt_idx = ious.max(dim=1)           # 每個預測對應的最佳 GT

                tp_mask = best_iou >= 0.5
                if tp_mask.any():
                    gt_used = set()
                    for i in torch.nonzero(tp_mask, as_tuple=False).squeeze(1).tolist():
                        g = int(gt_idx[i].item())
                        # 避免多個預測重複計同一個 GT
                        if g in gt_used:
                            continue
                        gt_used.add(g)

                        s_id = int(kept["scale_ids"][i].item())
                        s_id = max(0, min(num_scales - 1, s_id))  # 保險夾取
                        bucket = size_bucket(gt_boxes[g].tolist())
                        stats[(s_id, bucket)] += 1

    if model_was_training:
        model.train()

    # 畫長條圖
    values = np.array([[stats[(s, b)] for b in buckets] for s in scales])  # [S,3]
    x = np.arange(len(buckets))
    width = 0.8 / max(num_scales, 1)
    plt.figure(figsize=(7, 4))
    for i, s in enumerate(scales):
        plt.bar(x + (i - (num_scales - 1) / 2) * width, values[i], width=width, label=f"Scale {s + 1}")
    plt.xticks(x, buckets)
    plt.ylabel("True Positives (post-NMS)")
    plt.title("Scale specialization by object size")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "scale_performance.png", dpi=150)
    plt.close()

    # 存 JSON
    stats_json = {f"scale_{s + 1}": {b: int(stats[(s, b)]) for b in buckets} for s in scales}
    with open(save_dir / "scale_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2)

    return stats_json




def collate_fn(batch): 

    images, targets = zip(*batch)       # tuple of images, tuple of targets
    images = torch.stack(images, dim=0).contiguous()
    return images, list(targets)

def main():
    os.makedirs("results/visualizations", exist_ok=True)

    batch_size = 16
    learning_rate = 0.001


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # === Dataset & Loader ===

    val_dataset = ShapeDetectionDataset(
        image_dir="datasets/detection/val",
        annotation_file="datasets/detection/val_annotations.json",
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


    # === Model / Loss / Optimizer ===
    model = MultiScaleDetector(num_classes=3, num_anchors=3).to(device)
    ckpt = torch.load("results/best_model.pth", map_location=device)
    model.load_state_dict(ckpt)

    model.eval()

    

    # === Anchors ===
    feature_map_sizes = [(56, 56), (28, 28), (14, 14)]
    anchor_scales = [[16, 24, 32], [48, 64, 96], [96, 128, 192]]
    anchors_list = generate_anchors(feature_map_sizes, anchor_scales, image_size=224)
    anchors_device = [a.to(device) for a in anchors_list]


    # ---------- Visualize detections for first 10 images ----------
    conf_thresh = 0.5
    nms_iou = 0.6
    num_to_vis = min(10, len(val_dataset))
    # For convenience re-run a single-batch decode to get per-image kept detections
    # (We can iterate one-by-one for simplicity)
    for img_id in range(num_to_vis):
        img_path = val_dataset.samples[img_id]["path"]
        # Single image forward
        img = Image.open(img_path).convert("RGB")
        x = torch.from_numpy(np.asarray(img, dtype=np.float32)).permute(2, 0, 1).unsqueeze(0) / 255.0
        x = x.to(device)
        with torch.no_grad():
            raw = model(x)
            decoded = decode_predictions(raw, anchors_device, num_classes=3, conf_thresh=conf_thresh)[0]
            kept = _apply_nms_per_class(decoded, iou_thresh=nms_iou, topk_per_class=200)

        # Build GT for this image
        gt_boxes = torch.as_tensor(val_dataset.samples[img_id]["boxes"], dtype=torch.float32)
        gt_labels = torch.as_tensor(val_dataset.samples[img_id]["label"], dtype=torch.long)

        visualize_detections(
            image=img,
            predictions={"boxes": kept["boxes"].cpu(), "scores": kept["scores"].cpu(), "labels": kept["labels"].cpu()},
            ground_truths={"boxes": gt_boxes, "labels": gt_labels},
            save_path=f"results/visualizations/val_det_{img_id:03d}.png"
        )

    # ---------- Anchor coverage visualization (use the first val image) ----------
    if len(val_dataset) > 0:
        img0_path = val_dataset.samples[0]["path"]
        gt0 = val_dataset.samples[0]["boxes"]
        visualize_anchor_coverage(img0_path, gt0, anchors_device, "results/visualizations/", iou_thr=0.5)

    # ---------- Scale specialization analysis ----------

    val_loader_small = DataLoader(val_dataset, batch_size=8, shuffle=False,
                                  collate_fn=lambda b: (torch.stack([x for x, _ in b]), [t for _, t in b]))
    stats_json = analyze_scale_performance(model, val_loader_small, anchors_device)




if __name__ == '__main__':
    main()
