from __future__ import annotations

import numpy as np
from typing import Optional, Union


IOU_THRESHOLD_SCORE = 0.2
VISUAL_2D_THRESHOLD_SCORE = 0.6


class Object3D:
    def __init__(
        self,
        identifier: Union[int, str],
        category: str,
        min_xyz: np.ndarray,  # min_xyz of the aabb bounding box
        max_xyz: np.ndarray,  # max_xyz of the aabb bounding box
        context_clip_feature: Optional[np.ndarray] = None,
        # the clip feature of the videntifiereo frames containing the object
        description: Optional[str] = None,
        object_clip_feature: Optional[np.ndarray] = None, # the clip feature of the cropped object images
        object_dinov2_feature: Optional[np.ndarray] = None, # the dinov2 feature of the cropped object images
    ):
        self.identifier = identifier
        self.category = category
        self.description = description
        self.min_xyz = min_xyz
        self.max_xyz = max_xyz
        self.position: np.ndarray = np.mean([self.min_xyz, self.max_xyz], axis=0)
        self.object_clip_feature = object_clip_feature
        self.object_dinov2_feature = object_dinov2_feature
        self.context_clip_feature = context_clip_feature
        self.image_size = 0

    def visual_similarity_score(self, compared_object: Object3D):
        """calculate the visual similarity score of the two objects"""
        # clip similarity
        clip_emb1 = self.object_clip_feature
        clip_emb2 = compared_object.object_clip_feature
        clip_cosine_score = np.dot(clip_emb1, clip_emb2) / (
            np.linalg.norm(clip_emb1) * np.linalg.norm(clip_emb2)
        )
        x = -20 * (clip_cosine_score - 0.925)
        x = np.clip(x, -30, 30)
        clip_score = 1 / (1 + np.exp(x))
        # dinov2 similarity
        dinov2_emb1 = self.object_dinov2_feature
        dinov2_emb2 = compared_object.object_dinov2_feature
        dinov2_cosine_score = np.dot(dinov2_emb1, dinov2_emb2) / (
            np.linalg.norm(dinov2_emb1) * np.linalg.norm(dinov2_emb2)
        )
        dinov2_score = dinov2_cosine_score
        # similartiy ensemble
        return 0.15 * clip_score + 0.85 * dinov2_score

    def spatial_iou(self, compared_object: Object3D):
        """calculate the bounding box IoU of the two objects"""
        min_xyz_1, max_xyz_1 = self.min_xyz, self.max_xyz
        min_xyz_2, max_xyz_2 = compared_object.min_xyz, compared_object.max_xyz
        inter_box_min = np.max(np.stack([min_xyz_1, min_xyz_2]), axis=0)
        inter_box_max = np.min(np.stack([max_xyz_1, max_xyz_2]), axis=0)
        inter_box_size = inter_box_max - inter_box_min
        box1_size = max_xyz_1 - min_xyz_1
        box2_size = max_xyz_2 - min_xyz_2
        inter_volume = (
            max(0, inter_box_size[0])
            * max(0, inter_box_size[1])
            * max(0, inter_box_size[2])
        )
        box_1_volume = box1_size[0] * box1_size[1] * box1_size[2]
        box_2_volume = box2_size[0] * box2_size[1] * box2_size[2]
        union_volume = box_1_volume + box_2_volume - inter_volume
        iou = inter_volume / (union_volume + 1e-10)
        return iou

    def spatial_max_ios(self, compared_object: Object3D):
        """calculate the bounding box MaxIoS of the two objects"""
        min_xyz_1, max_xyz_1 = self.min_xyz, self.max_xyz
        min_xyz_2, max_xyz_2 = compared_object.min_xyz, compared_object.max_xyz
        inter_box_min = np.max(np.stack([min_xyz_1, min_xyz_2]), axis=0)
        inter_box_max = np.min(np.stack([max_xyz_1, max_xyz_2]), axis=0)
        inter_box_size = inter_box_max - inter_box_min
        box1_size = max_xyz_1 - min_xyz_1
        box2_size = max_xyz_2 - min_xyz_2
        inter_volume = (
            max(0, inter_box_size[0])
            * max(0, inter_box_size[1])
            * max(0, inter_box_size[2])
        )
        box_1_volume = box1_size[0] * box1_size[1] * box1_size[2]
        box_2_volume = box2_size[0] * box2_size[1] * box2_size[2]
        maxios = max(
            inter_volume / (box_1_volume + 1e-10), inter_volume / (box_2_volume + 1e-10)
        )
        return maxios

    def spatial_vol_sim(self, compared_object: Object3D):
        """calculate the bounding box volume similarity of the two objects"""
        min_xyz_1, max_xyz_1 = self.min_xyz, self.max_xyz
        min_xyz_2, max_xyz_2 = compared_object.min_xyz, compared_object.max_xyz
        box1_size = max_xyz_1 - min_xyz_1
        box2_size = max_xyz_2 - min_xyz_2
        v1 = box1_size[0] * box1_size[1] * box1_size[2]
        v2 = box2_size[0] * box2_size[1] * box2_size[2]
        return min(v1, v2) / (max(v1, v2) + 1e-10)

    def merge(self, target_object: Object3D, ratio=0.2):
        """merge the target object into the current object"""
        self.category = target_object.category
        outer_min = (1 - ratio) * self.min_xyz + ratio * target_object.min_xyz
        outer_max = (1 - ratio) * self.max_xyz + ratio * target_object.max_xyz
        self.min_xyz = outer_min
        self.max_xyz = outer_max
        self.position = np.mean([self.min_xyz, self.max_xyz], axis=0)
        if (
            self.context_clip_feature is not None
            and target_object.context_clip_feature is not None
        ):
            self.context_clip_feature = (
                1 - ratio
            ) * self.context_clip_feature + ratio * target_object.context_clip_feature
        if (
            self.object_clip_feature is not None
            and target_object.object_clip_feature is not None
        ):
            self.object_clip_feature = (
                1 - ratio
            ) * self.object_clip_feature + ratio * target_object.object_clip_feature
        if (
            self.object_dinov2_feature is not None
            and target_object.object_dinov2_feature is not None
        ):
            self.object_dinov2_feature = (
                1 - ratio
            ) * self.object_dinov2_feature + ratio * target_object.object_dinov2_feature

    def to_dict(self):
        return {
            "object_id": self.identifier,
            "description": self.description,
            "category": self.category,
            "location": np.round(self.position, 2).tolist(),
        }