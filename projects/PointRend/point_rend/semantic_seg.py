# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
from typing import Dict
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import ShapeSpec, cat
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from .point_features import (
    get_uncertain_point_coords_on_grid,
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from .point_head import build_point_head

def concatenate(feats):
    _, _, H, W = feats[0].size()
    x = []
    for feat in feats:
        x += [F.interpolate(feat, size=(H, W), mode='bilinear')]
    return torch.cat(x, dim=1)

def calculate_uncertainty(sem_seg_logits):
    """
    For each location of the prediction `sem_seg_logits` we estimate uncerainty as the
        difference between top first and top second predicted logits.

    Args:
        mask_logits (Tensor): A tensor of shape (N, C, ...), where N is the minibatch size and
            C is the number of foreground classes. The values are logits.

    Returns:
        scores (Tensor): A tensor of shape (N, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    top2_scores = torch.topk(sem_seg_logits, k=2, dim=1)[0]
    return (top2_scores[:, 1] - top2_scores[:, 0]).unsqueeze(1)

def one_hot(index, classes):
    # index is not flattened (pypass ignore) ############
    # size = index.size()[:1] + (classes,) + index.size()[1:]
    # view = index.size()[:1] + (1,) + index.size()[1:]
    #####################################################
    # index is flatten (during ignore) ##################
    size = index.size()[:1] + (classes,)
    view = index.size()[:1] + (1,)
    #####################################################

    # mask = torch.Tensor(size).fill_(0).to(device)
    mask = torch.zeros(size).to(index.device)
    index = index.view(view)
    ones = 1.

    return mask.scatter_(1, index, ones)

class ErrorPredictor(nn.Module):
    
    def __init__(self, cfg, ignore_value, input_shape):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(input_shape.channels, cfg.MODEL.POINT_HEAD.FC_DIM, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(cfg.MODEL.POINT_HEAD.FC_DIM), 
            nn.ReLU(), 
            nn.Conv2d(cfg.MODEL.POINT_HEAD.FC_DIM, cfg.MODEL.POINT_HEAD.FC_DIM, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(cfg.MODEL.POINT_HEAD.FC_DIM), 
            nn.ReLU(), 
            nn.Conv2d(cfg.MODEL.POINT_HEAD.FC_DIM, 1, kernel_size=1, padding=0), nn.Sigmoid())
        self.ignore_value = ignore_value
    
    def losses(self, inp, logits, target):
        B, C, H, W = logits.shape
        # import pdb; pdb.set_trace()
        # Reshape target
        if target.shape[1] != inp.shape[1] or target.shape[2] != inp.shape[2]:
            target = F.interpolate(target.unsqueeze(1).type(torch.float32), size=(inp.shape[1], inp.shape[2]), mode='nearest')
            target = target.squeeze(1).type(torch.int64)
        
        # Flatten and mask out
        target = target.view(-1)
        inp = inp.view(-1)
        logits = logits.permute(0, 2, 3, 1).contiguous().view(-1, C)

        valid = (target != self.ignore_value)
        target = target[valid]
        inp = inp[valid]
        logits = logits[valid, :]

        score = (one_hot(target, C) * torch.softmax(logits, dim=1)).sum(1)

        inp = inp.clamp(1e-7, 1.0 - 1e-7)
        loss = F.binary_cross_entropy(inp, score, reduction="mean", weight=torch.Tensor([10]).to(inp.device))
        if torch.isnan(loss):
            import pdb; pdb.set_trace()
        return loss
        

    def forward(self, features):
        return self.convs(features).squeeze(1)

@SEM_SEG_HEADS_REGISTRY.register()
class PointRendSemSegHead(nn.Module):
    """
    A semantic segmentation head that combines a head set in `POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME`
    and a point head set in `MODEL.POINT_HEAD.NAME`.
    """

    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        self.ignore_value = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE

        self.coarse_sem_seg_head = SEM_SEG_HEADS_REGISTRY.get(
            cfg.MODEL.POINT_HEAD.COARSE_SEM_SEG_HEAD_NAME
        )(cfg, input_shape)
        self._init_point_head(cfg, input_shape)

        # Missing points predictor
        in_channels             = sum([v.channels for k, v in input_shape.items()])
        self.error_predictor = ErrorPredictor(cfg, self.ignore_value, ShapeSpec(channels=in_channels, width=1, height=1))
        

    def _init_point_head(self, cfg, input_shape: Dict[str, ShapeSpec]):
        # fmt: off
        assert cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES == cfg.MODEL.POINT_HEAD.NUM_CLASSES
        feature_channels             = {k: v.channels for k, v in input_shape.items()}
        self.in_features             = cfg.MODEL.POINT_HEAD.IN_FEATURES
        self.train_num_points        = cfg.MODEL.POINT_HEAD.TRAIN_NUM_POINTS
        self.oversample_ratio        = cfg.MODEL.POINT_HEAD.OVERSAMPLE_RATIO
        self.importance_sample_ratio = cfg.MODEL.POINT_HEAD.IMPORTANCE_SAMPLE_RATIO
        self.subdivision_steps       = cfg.MODEL.POINT_HEAD.SUBDIVISION_STEPS
        self.subdivision_num_points  = cfg.MODEL.POINT_HEAD.SUBDIVISION_NUM_POINTS
        # fmt: on

        in_channels = np.sum([feature_channels[f] for f in self.in_features])
        self.point_head = build_point_head(cfg, ShapeSpec(channels=in_channels, width=1, height=1))

    def forward(self, features, targets=None):
        # import pdb; pdb.set_trace()
        '''
            - p2: 256 x 128 x 256 (1/4)
            - p3: 256 x 64 x 128 (1/8)
            - p4: 256 x 32 x 64 (1/16)
            - p5: 256 x 16 x 32 (1/32)
            - p6: 256 x 8 x 16 (1/64)
        '''
        coarse_sem_seg_logits = self.coarse_sem_seg_head.layers(features) # (conv + Upsample) --> add --> conv: n_classes x 128 x 256 ~ p2

        if self.training:
            losses = self.coarse_sem_seg_head.losses(coarse_sem_seg_logits, targets) # Upsample logits -> CE(logits, targets)
            if torch.isnan(losses['loss_sem_seg']):
                import pdb; pdb.set_trace()
            # Predict error
            all_features = concatenate([v.clone().detach() for v in features.values()])
            error_logits = self.error_predictor(all_features)
            losses["loss_sem_seg_refine"] = self.error_predictor.losses(error_logits, coarse_sem_seg_logits.clone().detach(), targets)

            with torch.no_grad():
                point_coords = get_uncertain_point_coords_with_randomness(
                    coarse_sem_seg_logits,
                    calculate_uncertainty,
                    self.train_num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                ) # batch x points x 2 (x, y)

            coarse_features = point_sample(coarse_sem_seg_logits, point_coords, align_corners=False) # batch x n_class x train_num_points

            fine_grained_features = cat(
                [
                    point_sample(features[in_feature], point_coords, align_corners=False)
                    for in_feature in self.in_features
                ],
                dim=1,
            ) # batch x 256 x n_points (for p2 only)
            # import pdb; pdb.set_trace()
            point_logits = self.point_head(fine_grained_features, coarse_features)
            point_targets = (
                point_sample(
                    targets.unsqueeze(1).to(torch.float),
                    point_coords,
                    mode="nearest",
                    align_corners=False,
                )
                .squeeze(1)
                .to(torch.long)
            )
            losses["loss_sem_seg_point"] = F.cross_entropy(
                point_logits, point_targets, reduction="mean", ignore_index=self.ignore_value
            )
            return None, losses
        else:
            sem_seg_logits = coarse_sem_seg_logits.clone()
            for _ in range(self.subdivision_steps):
                sem_seg_logits = F.interpolate(
                    sem_seg_logits, scale_factor=2, mode="bilinear", align_corners=False
                )

                # TODO: Predict better points
                uncertainty_map = calculate_uncertainty(sem_seg_logits)
                point_indices, point_coords = get_uncertain_point_coords_on_grid(
                    uncertainty_map, self.subdivision_num_points
                )

                fine_grained_features = cat(
                    [
                        point_sample(features[in_feature], point_coords, align_corners=False)
                        for in_feature in self.in_features
                    ]
                )
                coarse_features = point_sample(
                    coarse_sem_seg_logits, point_coords, align_corners=False
                )
                point_logits = self.point_head(fine_grained_features, coarse_features)

                # put sem seg point predictions to the right places on the upsampled grid.
                N, C, H, W = sem_seg_logits.shape
                point_indices = point_indices.unsqueeze(1).expand(-1, C, -1)
                sem_seg_logits = (
                    sem_seg_logits.reshape(N, C, H * W)
                    .scatter_(2, point_indices, point_logits)
                    .view(N, C, H, W)
                )
            return sem_seg_logits, {}
