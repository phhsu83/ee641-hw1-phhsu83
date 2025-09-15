import torch
import numpy as np
import matplotlib.pyplot as plt

import os
import math
import csv
from torch.utils.data import DataLoader
from dataset import KeypointDataset
from model import HeatmapNet, RegressionNet

def analyze_failure_cases(
    hm_model,
    rg_model,
    test_loader,
    pck_t: float = 0.1,
    success_ratio: float = 0.8,
    out_dir: str = "results/visualizations",
    max_per_category: int = 30,
    device=None,
):
    """
    Identify and visualize failure cases.

    分類規則（以每張圖的 PCK@pck_t 為準，成功門檻 = success_ratio）：
      - hm_only:   Heatmap 成功、Regression 失敗
      - rg_only:   Regression 成功、Heatmap 失敗
      - both_fail: 兩者皆失敗
      - both_ok:   兩者皆成功（通常只放少量作對照）

    會輸出：
      - results/failures/{category}_XXXX.png （每類最多 max_per_category 張）
      - results/failures/failure_index.csv   （每張圖的 pck_hm, pck_rg, hits, denom, 類別）
    """
    os.makedirs(out_dir, exist_ok=True)
    if device is None:
        device = next(hm_model.parameters()).device

    hm_model.eval()
    rg_model.eval()

    # 小工具：計算每張圖的「命中 keypoints 數」與分母（bbox 正規化）
    def _per_sample_hits(pred_xy: torch.Tensor, gt_xy: torch.Tensor, t: float):
        """
        pred_xy, gt_xy: [B,K,2] (像素座標)
        return hits[B], denom[B]
        """
        B, K, _ = pred_xy.shape
        valid = torch.isfinite(pred_xy).all(-1) & torch.isfinite(gt_xy).all(-1)  # [B,K]
        d = torch.linalg.norm(pred_xy - gt_xy, dim=-1)                           # [B,K]

        x = gt_xy[..., 0]
        y = gt_xy[..., 1]
        inf = torch.tensor(float('inf'), device=gt_xy.device)
        x_min = torch.where(valid, x, inf).min(1).values
        y_min = torch.where(valid, y, inf).min(1).values
        x_max = torch.where(valid, x, -inf).max(1).values
        y_max = torch.where(valid, y, -inf).max(1).values
        w = (x_max - x_min).clamp_min(0.0)
        h = (y_max - y_min).clamp_min(0.0)
        norm = torch.sqrt(w*w + h*h).clamp_min(1e-8)                             # [B]

        thr = norm.unsqueeze(1) * float(t)                                       # [B,1]
        correct = (d <= thr) & valid                                             # [B,K]
        hits = correct.sum(1)                                                    # [B]
        denom = valid.sum(1)                                                     # [B]
        return hits, denom

    # 小工具：同圖疊 HM/REG/GT（不改你現有 visualize_predictions）
    def _visualize_both(image_t, pred_hm_t, pred_rg_t, gt_t, save_path, t=None):
        img = image_t.detach().cpu()
        ph  = pred_hm_t.detach().cpu().numpy()
        pr  = pred_rg_t.detach().cpu().numpy()
        gt  = gt_t.detach().cpu().numpy()

        # 轉成 3 通道方便疊色
        if img.ndim == 3 and img.shape[0] == 1:
            vis_img = img[0].numpy()
        elif img.ndim == 2:
            vis_img = img.numpy()
        else:
            raise ValueError("image tensor must be [1,H,W] or [H,W]")
        H, W = vis_img.shape

        plt.figure(figsize=(5,5), dpi=140)
        plt.imshow(vis_img, cmap="gray", vmin=0, vmax=1)
        plt.scatter(gt[:,0], gt[:,1], s=30, marker="o", facecolors="none", edgecolors="lime", linewidths=2, label="GT")
        plt.scatter(ph[:,0], ph[:,1], s=28, marker="x", c="deepskyblue", linewidths=2, label="Heatmap")
        plt.scatter(pr[:,0], pr[:,1], s=28, marker="x", c="red", linewidths=2, label="Regression")
        # 門檻圈（bbox 對角線 * t）
        if t is not None and np.all(np.isfinite(gt)):
            x_min, y_min = gt[:,0].min(), gt[:,1].min()
            x_max, y_max = gt[:,0].max(), gt[:,1].max()
            diag = max(np.hypot(max(x_max-x_min,0), max(y_max-y_min,0)), 1e-8)
            r = float(t) * diag
            for i in range(gt.shape[0]):
                circ = plt.Circle((gt[i,0], gt[i,1]), r, edgecolor='yellow', facecolor='none', lw=1, alpha=0.7)
                plt.gca().add_patch(circ)

        plt.xlim(-0.5, W-0.5); plt.ylim(H-0.5, -0.5)
        plt.legend(loc="upper right", fontsize=8)
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    csv_path = os.path.join(out_dir, "failure_index.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "global_index", "category", "pck_hm", "pck_rg", "hits_hm", "hits_rg", "denom"
        ])
        writer.writeheader()

        saved = {"hm_only":0, "rg_only":0, "both_fail":0, "both_ok":0}
        global_idx = 0

        with torch.no_grad():
            for batch_idx, (images, gt_norm) in enumerate(test_loader):
                images = images.to(device)
                B, C, H_img, W_img = images.shape
                K = gt_norm.size(1) // 2

                # GT: [B,2K] -> [B,K,2] (像素座標)
                gt_xy = gt_norm.reshape(B, K, 2).to(device).clone()
                gt_xy[..., 0] *= W_img
                gt_xy[..., 1] *= H_img

                # Heatmap 預測 → 影像座標
                hm_out = hm_model(images)                    # [B,K,Hm,Wm]
                coords_hm = extract_keypoints_from_heatmaps(hm_out)  # 熱圖座標
                Hm, Wm = hm_out.shape[-2:]
                coords_hm = coords_hm.to(device)
                coords_hm[..., 0] *= (W_img / max(Wm, 1))
                coords_hm[..., 1] *= (H_img / max(Hm, 1))

                # Regression 預測 → 影像座標
                rg_out = rg_model(images)                    # [B,2K] 或 [B,K,2]
                coords_rg = (rg_out.reshape(B, K, 2)).to(device)
                # 大多回歸是 [0,1]，保守一點 clamp 後轉像素
                coords_rg = coords_rg.clamp(0, 1)
                coords_rg[..., 0] *= W_img
                coords_rg[..., 1] *= H_img

                # per-image PCK@t
                hits_hm, denom = _per_sample_hits(coords_hm, gt_xy, pck_t)
                hits_rg, _     = _per_sample_hits(coords_rg, gt_xy, pck_t)
                score_hm = (hits_hm / denom.clamp_min(1)).cpu().numpy()
                score_rg = (hits_rg / denom.clamp_min(1)).cpu().numpy()

                for i in range(B):
                    ok_hm = score_hm[i] >= success_ratio
                    ok_rg = score_rg[i] >= success_ratio
                    if   ok_hm and not ok_rg: category = "hm_only"
                    elif ok_rg and not ok_hm: category = "rg_only"
                    elif not ok_hm and not ok_rg: category = "both_fail"
                    else: category = "both_ok"

                    # 存圖（每類限額）
                    if saved[category] < max_per_category:
                        save_path = os.path.join(out_dir, f"{category}_{saved[category]:04d}.png")
                        _visualize_both(
                            images[i].detach().cpu(),
                            coords_hm[i].detach().cpu(),
                            coords_rg[i].detach().cpu(),
                            gt_xy[i].detach().cpu(),
                            save_path,
                            t=pck_t
                        )
                        saved[category] += 1

                    
                    writer.writerow({
                        "global_index": global_idx,
                        "category": category,
                        "pck_hm": float(score_hm[i]),
                        "pck_rg": float(score_rg[i]),
                        "hits_hm": int(hits_hm[i].item()),
                        "hits_rg": int(hits_rg[i].item()),
                        "denom": int(denom[i].item()),
                    })
                    
                    global_idx += 1

    print(f"[analyze_failure_cases] Done. Saved images & CSV to: {out_dir}")


def extract_keypoints_from_heatmaps(heatmaps):
    """
    Extract (x, y) coordinates from heatmaps.
    
    Args:
        heatmaps: Tensor of shape [batch, num_keypoints, H, W]
        
    Returns:
        coords: Tensor of shape [batch, num_keypoints, 2]
    """
    # Find argmax location in each heatmap
    # Convert to (x, y) coordinates

    B, K, H, W = heatmaps.shape
    

    # flat = heatmaps.view(B, K, -1)                   # [B,K,H*W]
    flat = heatmaps.flatten(start_dim=2) 

    idx = flat.argmax(dim=-1)                        # [B,K]

    '''
    # 把座標 從熱圖座標（0..W-1, 0..H-1）縮放到影像座標，但寫死了 H_img=W_img=128。
    # y = (idx // W).float()
    # x = (idx %  W).float()
    y = (idx // W).to(torch.float32)
    x = (idx %  W).to(torch.float32)
    coords = torch.stack([x, y], dim=-1)             # [B,K,2]

    H_img, W_img = 128, 128
    scale_x = (W_img - 1) / max(W - 1, 1)
    scale_y = (H_img - 1) / max(H - 1, 1)
    coords[..., 0] = coords[..., 0] * scale_x
    coords[..., 1] = coords[..., 1] * scale_y

    return coords
    '''

    y = (idx // W).to(heatmaps.dtype)
    x = (idx %  W).to(heatmaps.dtype)
    return torch.stack([x, y], dim=-1)            # [B,K,2]



def compute_pck(predictions, ground_truths, thresholds, normalize_by='bbox'):
    """
    Compute PCK at various thresholds.
    
    Args:
        predictions: Tensor of shape [N, num_keypoints, 2]
        ground_truths: Tensor of shape [N, num_keypoints, 2]
        thresholds: List of threshold values (as fraction of normalization)
        normalize_by: 'bbox' for bounding box diagonal, 'torso' for torso length
        
    Returns:
        pck_values: Dict mapping threshold to accuracy
    """
    # For each threshold:
    # Count keypoints within threshold distance of ground truth
    

    """
    Compute PCK (Percentage of Correct Keypoints) at various thresholds.

    Args:
        predictions:  [N, K, 2]  預測座標 (x,y)，單位須與 GT 相同
        ground_truths:[N, K, 2]  GT 座標 (x,y)
        thresholds:   可迭代的數值（例如 [0.05, 0.1, 0.2]）
        normalize_by: 'bbox' 或 'torso'
                      - bbox: 用 GT keypoints 的包圍盒對角線作尺度
                      - torso: 用 (肩中點) 到 (髖中點) 的距離作尺度（COCO: 5,6,11,12）

    Returns:
        dict: {thr(float): accuracy(float)}
    """

    
    device = predictions.device

    N, K, _ = predictions.shape
    eps = 1e-8

    # 歐式距離 [N, K]
    dists = torch.linalg.norm(predictions - ground_truths, dim=-1) # (x_pred - x_gt, y_pred - y_gt)

    # 有效點（pred/gt 都是有限值）[N, K]
    valid = torch.isfinite(predictions).all(dim=-1) & torch.isfinite(ground_truths).all(dim=-1)

    # ---- 正規化尺度 per-sample: norm [N]，同時產出該樣本尺度是否可用 norm_ok [N] ----
    if normalize_by == 'bbox':
        x = ground_truths[..., 0]
        y = ground_truths[..., 1]

        # min/max 要忽略無效點：無效設為 +inf / -inf，確保不影響 min/max
        inf = torch.tensor(float('inf'), device=device)
        x_min = torch.where(valid, x, inf).min(dim=1).values
        y_min = torch.where(valid, y, inf).min(dim=1).values
        x_max = torch.where(valid, x, -inf).max(dim=1).values
        y_max = torch.where(valid, y, -inf).max(dim=1).values

        # 樣本至少要有一個有效點
        norm_ok = torch.isfinite(x_min) & torch.isfinite(x_max) & torch.isfinite(y_min) & torch.isfinite(y_max)

        w = (x_max - x_min).clamp_min(0.0)
        h = (y_max - y_min).clamp_min(0.0)
        diag = torch.sqrt(w * w + h * h)
        # 沒有 bbox（沒有效點）的樣本先設 1，等等會把它們從有效集合剔除
        norm = torch.where(norm_ok, diag, torch.ones_like(diag)).clamp_min(eps)

    elif normalize_by == 'torso':
        # COCO: 5=left_shoulder, 6=right_shoulder, 11=left_hip, 12=right_hip
        def get_point(idx):
            pt = ground_truths[:, idx, :]  # [N,2]
            v  = torch.isfinite(pt).all(dim=-1)  # [N]
            return pt, v

        if K <= 12:
            # keypoints 不足以取 torso，全部視為無效樣本
            norm_ok = torch.zeros((N,), dtype=torch.bool, device=device)
            norm = torch.ones((N,), dtype=torch.float32, device=device)
        else:
            ls, v_ls = get_point(5)
            rs, v_rs = get_point(6)
            lh, v_lh = get_point(11)
            rh, v_rh = get_point(12)

            sh_mid = (ls + rs) / 2.0
            hip_mid = (lh + rh) / 2.0
            norm_ok = v_ls & v_rs & v_lh & v_rh

            torso_len = torch.linalg.norm(sh_mid - hip_mid, dim=-1)  # [N]
            norm = torch.where(norm_ok, torso_len, torch.ones_like(torso_len)).clamp_min(eps)
    else:
        raise ValueError("normalize_by must be 'bbox' or 'torso'")

    # 只有同時具備：該 keypoint 有效 且 該樣本的正規化尺度可用，才計入分母
    valid_all = valid & norm_ok.unsqueeze(1)  # [N, K]

    results = {}
    denom = valid_all.sum().item()  # 總有效 keypoints 數
    for t in thresholds:
        t = float(t)
        thr_mat = norm.unsqueeze(1) * t  # [N, 1]
        correct = (dists <= thr_mat) & valid_all
        num = correct.sum().item()
        acc = (float(num) / float(denom)) if denom > 0 else float('nan')
        results[t] = acc

    return results



def plot_pck_curves(pck_heatmap, pck_regression, save_path):
    """
    Plot PCK curves comparing both methods.
    """
    
    """
    Plot PCK curves comparing both methods and save the figure.

    Args:
        pck_heatmap:   dict-like {threshold(float): accuracy(float in [0,1])}
        pck_regression:dict-like {threshold(float): accuracy(float in [0,1])}
        save_path:     path to save the figure (e.g., "results/pck.png")

    Returns:
        aucs: dict with keys {"heatmap_auc", "regression_auc"} (AUC in [0,1])
    """
    
    def to_xy(pck_dict):
        # 轉成排序後的 (x, y)
        xs = np.array(sorted(pck_dict.keys()), dtype=float)
        ys = np.array([pck_dict[x] for x in xs], dtype=float)
        return xs, ys

    def auc_safe(x, y):
        # 去掉 NaN；若點數不足回傳 NaN
        mask = np.isfinite(x) & np.isfinite(y)
        x2, y2 = x[mask], y[mask]
        if x2.size < 2:
            return float("nan")
        # 以 x 升冪排序（保險）
        order = np.argsort(x2)
        x2, y2 = x2[order], y2[order]
        # 梯形積分，並以 x 的範圍歸一化到 [0,1]
        width = x2[-1] - x2[0]
        if width <= 0:
            return float("nan")
        auc = np.trapz(y2, x2) / width
        return float(np.clip(auc, 0.0, 1.0))

    # 取得座標點
    hx, hy = to_xy(pck_heatmap)
    rx, ry = to_xy(pck_regression)

    # 計算 AUC
    heatmap_auc   = auc_safe(hx, hy)
    regression_auc= auc_safe(rx, ry)

    # 畫圖
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.figure(figsize=(6, 4.5), dpi=140)

    # 兩條曲線各自用自己的門檻點（不強行對齊 union，避免不必要的插值/NaN）
    plt.plot(hx, hy, marker="o", linewidth=1.8, label=f"Heatmap (AUC={heatmap_auc:.3f})")
    plt.plot(rx, ry, marker="s", linewidth=1.8, label=f"Regression (AUC={regression_auc:.3f})")

    # 美化
    plt.xlabel("Threshold (fraction of normalization)")
    plt.ylabel("PCK (accuracy)")
    plt.title("PCK Curves: Heatmap vs. Regression")
    plt.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    # x 軸範圍根據兩組門檻的 min/max 自動決定
    all_x = np.concatenate([hx, rx]) if hx.size and rx.size else (hx if hx.size else rx)
    if all_x.size:
        xmin, xmax = float(np.min(all_x)), float(np.max(all_x))
        if xmin == xmax:  # 單一門檻的情況
            xmin, xmax = xmin - 0.05, xmax + 0.05
        plt.xlim(xmin, xmax)
    plt.ylim(0.0, 1.0)
    plt.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    # return {"heatmap_auc": heatmap_auc, "regression_auc": regression_auc}
    
    '''
    thrs_h = sorted(pck_heatmap.keys())
    thrs_r = sorted(pck_regression.keys())
    xs = sorted(set(thrs_h) | set(thrs_r))

    y_h = [pck_heatmap.get(t, np.nan) for t in xs]
    y_r = [pck_regression.get(t, np.nan) for t in xs]

    plt.figure(figsize=(6, 4.5))
    plt.plot(xs, y_h, marker="o", label="Heatmap")
    plt.plot(xs, y_r, marker="s", label="Regression")
    plt.xlabel("Threshold (fraction of normalization)")
    plt.ylabel("PCK (accuracy)")
    plt.ylim(0.0, 1.0)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()
    '''


def visualize_predictions(image, pred_keypoints, gt_keypoints, save_path):
    """
    Visualize predicted and ground truth keypoints on image.
    """
    
    # 处理图像到 numpy HxW
    if isinstance(image, torch.Tensor):
        img = image.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        elif img.ndim == 2:
            pass
        else:
            raise ValueError("image tensor must be [1,H,W] or [H,W]")
    else:
        img = np.array(image)
        if img.ndim == 3:
            img = img[..., 0]  # 取灰度

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap="gray", vmin=0, vmax=1)

    # GT (绿色) 与 Pred (红色)
    pk = pred_keypoints.detach().cpu().numpy() if isinstance(pred_keypoints, torch.Tensor) else np.asarray(pred_keypoints)
    gk = gt_keypoints.detach().cpu().numpy() if isinstance(gt_keypoints, torch.Tensor) else np.asarray(gt_keypoints)

    plt.scatter(gk[:, 0], gk[:, 1], s=30, marker="o", facecolors="none", edgecolors="lime", linewidths=2, label="GT")
    plt.scatter(pk[:, 0], pk[:, 1], s=18, marker="x", c="red", linewidths=2, label="Pred")

    # 连线方便观察误差
    for (x1, y1), (x2, y2) in zip(gk, pk):
        plt.plot([x1, x2], [y1, y2], linestyle="--")

    plt.xlim(-0.5, img.shape[1] - 0.5)
    plt.ylim(img.shape[0] - 0.5, -0.5)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=180)
    plt.close()




def main():

    os.makedirs("results/visualizations", exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    THRESHOLDS = [0.05, 0.1, 0.15, 0.2]
    BATCH_SIZE = 32
    IMG_SIZE = 128
    
    # === Dataset & Loader ===



    val_dataset_reg = KeypointDataset(
        image_dir="datasets/keypoints/val",
        annotation_file="datasets/keypoints/val_annotations.json",
        output_type="regression"
    )
    val_loader_reg = DataLoader(val_dataset_reg, batch_size=BATCH_SIZE, shuffle=False)


    # 2) 載入最佳模型
    hm_model = HeatmapNet(num_keypoints=5).to(device) 
    rg_model = RegressionNet(num_keypoints=5).to(device) 
    hm_model.load_state_dict(torch.load("results/heatmap_model.pth", map_location=device))
    rg_model.load_state_dict(torch.load("results/regression_model.pth", map_location=device))
    hm_model.eval() 
    rg_model.eval()


    # ---------- Inference & PCK ----------
    all_pred_hm, all_pred_rg, all_gt = [], [], []
    sample_cache = []

    with torch.no_grad():
        for images, gt_norm in val_loader_reg:
            # images: [B, C, H, W] ; gt_norm: [B, 2K] (0~1)
            images = images.to(device)
            B, C, H_img, W_img = images.shape

            # 取得 GT (影像像素座標)
            gt_xy = gt_norm.view(B, 5, 2).clone()
            gt_xy[..., 0] *= IMG_SIZE
            gt_xy[..., 1] *= IMG_SIZE

            # 2a) Heatmap 模型 → 熱圖 → 座標（影像座標）
            hm = hm_model(images)                           # 期望 [B, K, Hm, Wm]
            coords_hm = extract_keypoints_from_heatmaps(hm)  # [B,K,2]
            Hm, Wm = hm.shape[-2:]
            coords_hm[..., 0] *= (IMG_SIZE / Wm)
            coords_hm[..., 1] *= (IMG_SIZE / Hm)

            # 2b) Regression 模型 → 直接輸出座標（通常是 0~1），再轉像素
            rg_out = rg_model(images)                          # 可能是 [B, 2K] 或 [B, K, 2]
            rg_out = rg_out.view(B, 5, 2)

            coords_rg = rg_out.clone()
            coords_rg[..., 0] *= IMG_SIZE
            coords_rg[..., 1] *= IMG_SIZE

            all_pred_hm.append(coords_hm.cpu())
            all_pred_rg.append(coords_rg.cpu())
            all_gt.append(gt_xy.cpu())

            if len(sample_cache) < 10:
                for i in range(min(B, 10 - len(sample_cache))):
                    sample_cache.append((
                        images[i].detach().cpu(),          # [1,128,128]
                        coords_hm[i].detach().cpu(),     # [K,2]
                        coords_rg[i].detach().cpu(),     # [K,2]
                        gt_xy[i].detach().cpu(),           # [K,2]
                    ))

    pred_xy_hm = torch.cat(all_pred_hm, dim=0)   # [N,K,2]
    pred_xy_rg = torch.cat(all_pred_rg, dim=0)   # [N,K,2]
    gt_xy      = torch.cat(all_gt,      dim=0)   # [N,K,2]

    # 3) 計算 PCK（用 bbox 對角線作正規化，或改用你前面定義的 height_proxy/hand_span）
    pck_hm = compute_pck(pred_xy_hm, gt_xy, THRESHOLDS, normalize_by='bbox')
    pck_rg = compute_pck(pred_xy_rg, gt_xy, THRESHOLDS, normalize_by='bbox')
    print("PCK (Heatmap):   ", {float(k): float(v) for k, v in pck_hm.items()})
    print("PCK (Regression):", {float(k): float(v) for k, v in pck_rg.items()})


    # 4) 畫 PCK 曲線（兩法同圖）
    plot_pck_curves(pck_hm, pck_rg, "results/visualizations/pck_curves.png")

    # ---------- Visualize predictions (前10张) ----------
    # 分别存两张：heatmap 预测 vs GT、regression 预测 vs GT
    for i, (img, pred_h, pred_r, gt) in enumerate(sample_cache):
        visualize_predictions(
            image=img,
            pred_keypoints=pred_h,
            gt_keypoints=gt,
            save_path=f"results/visualizations/val_vis_heatmap_{i:03d}.png"
        )
        visualize_predictions(
            image=img,
            pred_keypoints=pred_r,
            gt_keypoints=gt,
            save_path=f"results/visualizations/val_vis_regression_{i:03d}.png"
        )


    analyze_failure_cases(
        hm_model, rg_model, val_loader_reg,
        pck_t=0.1,          # PCK 門檻（相對 bbox 對角線）
        success_ratio=0.8,  # 視為「此圖成功」的比例門檻
        out_dir="results/visualizations",
        max_per_category=20,
        device=device
    )


if __name__ == '__main__':
    main()