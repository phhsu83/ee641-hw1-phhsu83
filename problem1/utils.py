import torch
import numpy as np

def generate_anchors(feature_map_sizes, anchor_scales, image_size=224):
    """
    Generate anchors for multiple feature maps.
    
    Args:
        feature_map_sizes: List of (H, W) tuples for each feature map
        anchor_scales: List of lists, scales for each feature map
        image_size: Input image size
        
    Returns:
        anchors: List of tensors, each of shape [H*W*num_anchors, 4]
                 in [x1, y1, x2, y2] format
    """
    # For each feature map:
    # 1. Create grid of anchor centers
    # 2. Generate anchors with specified scales and ratios
    # 3. Convert to absolute coordinates
    
    all_anchors = []

    for (H, W), scales in zip(feature_map_sizes, anchor_scales):

        # Create grid of anchor centers (H*W, 2)
        y_centers = (torch.arange(H) + 0.5) * (image_size / H)
        x_centers = (torch.arange(W) + 0.5) * (image_size / W)
        y_grid, x_grid = torch.meshgrid(y_centers, x_centers, indexing="ij")
        centers = torch.stack([x_grid, y_grid], dim=-1).reshape(-1, 2) # [[center1, center1],...]


        anchors = []
        for scale in scales:
            half_scale = scale / 2
            x1y1 = centers - half_scale
            x2y2 = centers + half_scale
            anchor = torch.cat([x1y1, x2y2], dim=1)  # [H*W, 4]
            anchors.append(anchor)
        
        anchors = torch.cat(anchors, dim=0)  # [H*W*num_scales, 4]
        all_anchors.append(anchors)
    
    return all_anchors


def compute_iou(boxes1, boxes2):
    """
    Compute IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape [N, 4]
        boxes2: Tensor of shape [M, 4]
        
    Returns:
        iou: Tensor of shape [N, M]
    """
    
    boxes1 = boxes1[:, None, :]  # [N, 1, 4]
    boxes2 = boxes2[None, :, :]  # [1, M, 4]

    # intersection
    inter_x1 = torch.max(boxes1[..., 0], boxes2[..., 0])
    inter_y1 = torch.max(boxes1[..., 1], boxes2[..., 1])
    inter_x2 = torch.min(boxes1[..., 2], boxes2[..., 2])
    inter_y2 = torch.min(boxes1[..., 3], boxes2[..., 3])

    inter_w = (inter_x2 - inter_x1).clamp(min=0)
    inter_h = (inter_y2 - inter_y1).clamp(min=0)
    inter_area = inter_w * inter_h

    # areas
    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    union = area1 + area2 - inter_area

    return inter_area / union.clamp(min=1e-6)  # [N, M]



def match_anchors_to_targets(anchors, target_boxes, target_labels, 
                            pos_threshold=0.5, neg_threshold=0.3):
    """
    Match anchors to ground truth boxes.
    
    Args:
        anchors: Tensor of shape [num_anchors, 4]
        target_boxes: Tensor of shape [num_targets, 4]
        target_labels: Tensor of shape [num_targets]
        pos_threshold: IoU threshold for positive anchors
        neg_threshold: IoU threshold for negative anchors
        
    Returns:
        matched_labels: Tensor of shape [num_anchors]
                       (0: background, 1-N: classes)
        matched_boxes: Tensor of shape [num_anchors, 4]
        pos_mask: Boolean tensor indicating positive anchors
        neg_mask: Boolean tensor indicating negative anchors
    """

    '''
    算出 IoU → 每個 anchor 跟所有 target 的重疊程度。
    找出每個 anchor 的最佳匹配 target。
    把 anchor 分成 正樣本 (positive)、負樣本 (negative)、忽略 (ignore)。
    確保 每個 target 至少有一個 anchor（避免漏檢）。
    '''
    
    num_anchors = anchors.size(0)
    num_targets = target_boxes.size(0)

    # IoU
    ious = compute_iou(anchors, target_boxes) # [num_anchors, num_targets]

    # Find the best target for each anchor
    max_ious, max_idx = ious.max(dim=1)  # [num_anchors] # max_idx: the index for the best target

    # Init
    matched_labels = torch.zeros(num_anchors, dtype=torch.long)   # 0 = background
    matched_boxes = torch.zeros(num_anchors, 4, dtype=torch.float)

    # Set positive anchors
    pos_mask = max_ious >= pos_threshold
    matched_labels[pos_mask] = target_labels[max_idx[pos_mask]]
    matched_boxes[pos_mask] = target_boxes[max_idx[pos_mask]]

    # Set negative anchors
    neg_mask = max_ious < neg_threshold

    # Ignore
    ignore_mask = (~pos_mask) & (~neg_mask)

    # Make sure every target has at least one anchor
    if num_targets > 0:
        best_anchor_per_target = ious.argmax(dim=0)  # [num_targets]
        matched_labels[best_anchor_per_target] = target_labels
        matched_boxes[best_anchor_per_target] = target_boxes
        pos_mask[best_anchor_per_target] = True
        neg_mask[best_anchor_per_target] = False
        ignore_mask[best_anchor_per_target] = False

    return matched_labels, matched_boxes, pos_mask, neg_mask

    '''
    matched_labels: 每個 anchor 最後的分類標籤
    matched_boxes: 每個 anchor 對應的 ground truth box（用來算 regression loss）
    pos_mask: 哪些 anchors 是 positive
    neg_mask: 哪些 anchors 是 negative
    '''






