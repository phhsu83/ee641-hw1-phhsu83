import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import match_anchors_to_targets

class DetectionLoss(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.num_classes = num_classes

    def encode_boxes(self, boxes, anchors):
        """
        Encode ground truth boxes relative to anchors.
        """

        # Convert to center format
        # GT boxes
        gt_cx = (boxes[:, 0] + boxes [:, 2]) / 2.0
        gt_cy = (boxes [:, 1] + boxes [:, 3]) / 2.0
        gt_w = boxes[:, 2] - boxes [:, 0]
        gt_h = boxes [:, 3] - boxes [:, 1]
        # anchor boxes
        anchor_cx = (anchors[:, 0] + anchors [:, 2]) / 2.0
        anchor_cy = (anchors[:, 1] + anchors [:, 3]) / 2.0
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        # Avoid log(0) by adding small epsilon
        eps = 1e-8
        anchor_w = torch.clamp(anchor_w, min=eps)
        anchor_h = torch.clamp(anchor_h, min=eps)
        gt_w = torch.clamp(gt_w, min=eps)
        gt_h = torch.clamp(gt_h, min=eps)
        # Encode as offsets
        dx = (gt_cx - anchor_cx) / anchor_w
        dy = (gt_cy - anchor_cy) / anchor_h
        dw = torch.log(gt_w / anchor_w)
        dh = torch.log(gt_h / anchor_h)
        return torch.stack([dx, dy, dw, dh], dim=1)
        
    def forward(self, predictions, targets, anchors):
        """
        Compute multi-task loss.
        
        Args:
            predictions: List of tensors from each scale
            targets: List of dicts with 'boxes' and 'labels' for each image
            anchors: List of anchor tensors for each scale
            
        Returns:
            loss_dict: Dict containing:
                - loss_obj: Objectness loss
                - loss_cls: Classification loss  
                - loss_loc: Localization loss
                - loss_total: Weighted sum
        """
        # For each prediction scale:
        # 1. Match anchors to targets
        # 2. Compute objectness loss (BCE)
        # 3. Compute classification loss (CE) for positive anchors
        # 4. Compute localization loss (Smooth L1) for positive anchors
        # 5. Apply hard negative mining (3:1 ratio)


        loss_obj_all, loss_cls_all, loss_loc_all = [], [], []

        for preds, anchor_set in zip(predictions, anchors):
            

            B, _, H, W = preds.shape
            A = anchor_set.size(0) // (H * W)   # 每個 cell 的 anchor 數

            # (0) reshape 成 loss 可用的格式
            preds = preds.view(B, A, 5+self.num_classes, H, W)   # [B, A, 5+C, H, W]
            preds = preds.permute(0, 3, 4, 1, 2).contiguous()    # [B, H, W, A, 5+C]
            preds = preds.view(B, -1, 5+self.num_classes)        # [B, H*W*A, 5+C]

            # anchor_set 原本是 [num_anchors_total, 4]
            # 保持 [H*W*A, 4] 就好


            
            batch_size = preds.size(0)
            num_anchors = anchor_set.size(0)

            for b in range(batch_size):
                target_boxes = targets[b]["boxes"].to(preds.device)
                target_labels = targets[b]["labels"].to(preds.device)

                # (1) Anchor matching
                matched_labels, matched_boxes, pos_mask, neg_mask = \
                    match_anchors_to_targets(anchor_set, target_boxes, target_labels)

                '''
                # (2) Split predictions
                # preds[b] [num_anchors, 5 + C]
                # [objectness, class1..classC, dx, dy, dw, dh]
                pred_obj = preds[b, :, 0]           # [num_anchors]
                pred_cls = preds[b, :, 1:1 + self.num_classes]  # [num_anchors, C]
                pred_loc = preds[b, :, 1 + self.num_classes:]   # [num_anchors, 4]
                '''

                # (2) Split predictions
                # preds[b] [num_anchors, 5 + C]
                # where 5 = 4 bbox coords + 1 objectness score => [dx, dy, dw, dh, objectness, class1..classC]
                pred_loc = preds[b, :, 0:4] # bbox
                pred_obj = preds[b, :, 4] # objectness
                pred_cls = preds[b, :, 5:5 + self.num_classes] # anchors


                # (3) Objectness loss
                obj_target = pos_mask.float()
                obj_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    pred_obj, obj_target, reduction="none"
                )

                # Hard negative mining
                neg_sel = self.hard_negative_mining(obj_loss.detach(), pos_mask, neg_mask, ratio=3)
                obj_mask = pos_mask | neg_sel
                loss_obj = obj_loss[obj_mask].mean()

                # (4) Classification loss (only positives)
                loss_cls = torch.tensor(0., device=preds.device)
                if pos_mask.any():
                    loss_cls = torch.nn.functional.cross_entropy(
                        pred_cls[pos_mask], matched_labels[pos_mask],  # labels are 0 ~ 2
                        reduction='mean'
                    )
                # print(matched_labels[pos_mask])

                # (5) Localization loss (only positives)
                
                loss_loc = torch.tensor(0., device=preds.device)
                if pos_mask.any():
                    # encode ground truth boxes relative to positive anchors
                    pos_gt_boxes = matched_boxes[pos_mask] # absolute coordinates
                    pos_anchors = anchor_set[pos_mask] # anchor coord
                    encoded_boxes = self.encode_boxes(pos_gt_boxes, pos_anchors)

                    loss_loc = torch.nn.functional.smooth_l1_loss(
                        pred_loc[pos_mask], encoded_boxes, reduction='mean'
                    )

                # (6) Append batch loss
                loss_obj_all.append(loss_obj)
                loss_cls_all.append(loss_cls)
                loss_loc_all.append(loss_loc)

        # Loss
        loss_obj = torch.stack(loss_obj_all).mean()
        loss_cls = torch.stack(loss_cls_all).mean()
        loss_loc = torch.stack(loss_loc_all).mean()
        loss_total = (1.0 * loss_obj) + (1.0 * loss_cls) + (2.0 * loss_loc)

        return {
            "loss_obj": loss_obj,
            "loss_cls": loss_cls,
            "loss_loc": loss_loc,
            "loss_total": loss_total
        }
    
    def hard_negative_mining(self, loss, pos_mask, neg_mask, ratio=3):
        """
        Select hard negative examples.
        
        Args:
            loss: Loss values for all anchors
            pos_mask: Boolean mask for positive anchors
            neg_mask: Boolean mask for negative anchors
            ratio: Negative to positive ratio
            
        Returns:
            selected_neg_mask: Boolean mask for selected negatives
        """

        with torch.no_grad():
            num_pos = int(pos_mask.sum().item())
            # num_neg = min(int(num_pos * ratio), neg_mask.sum().item())
            # when no positives, still keep a few negatives
            max_neg = int(ratio * max(num_pos, 1))

            
            # if num_neg == 0:
            #     return torch.zeros_like(neg_mask, dtype=torch.bool)
            '''
            如果 正樣本數量為 0 → num_pos = 0 → num_pos * ratio = 0 → num_neg = 0
            如果 負樣本本身就太少（比如負樣本數 < num_pos * ratio） → num_neg = 負樣本總數，可能還是 0
            所以只要 num_neg == 0 就代表「這次沒有要挑選的負樣本」，不管是因為正樣本為 0 還是負樣本太少
            '''
            
            # Select every negative in loss
            neg_loss = loss.clone()
            neg_loss[~neg_mask] = -float("inf")  # Non negative -> -1, avoid to select

            # 真正能選的上限
            num_neg_total = int(neg_mask.sum().item())
            k = min(max_neg, num_neg_total)

            if k == 0:
                return torch.zeros_like(neg_mask, dtype=torch.bool)

            # 只取 top-k，不需全排序
            _, topk_idx = torch.topk(neg_loss, k=k, largest=True)
            
            # Select top-k hardest negatives
            # _, idx = neg_loss.sort(descending=True)
            # selected_idx = idx[:num_neg]
            
            selected_neg_mask = torch.zeros_like(neg_mask, dtype=torch.bool)
            selected_neg_mask[topk_idx] = True
            
            return selected_neg_mask

