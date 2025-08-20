from object3d import Object3D
from reid import static_object_reid, dynamic_object_reid, split_static_dynamic_objects, remove_duplicate_objects
from collections import defaultdict
import cv2
import os
from utils import depth2d_to_world3d_transformation
import numpy as np
import torch
from PIL import Image
from ultralytics import SAM, YOLOWorld
import clip
import torchvision.transforms as T
from utils import plot_visible_object_ids, plot_visible_object_bboxes
from detection_classes import customized_classes



class ObjectMemory:
    def __init__(self, classes, save_dir=None, dynamic: bool=True):
        self.static_objects: list[Object3D] = []  # static objects
        self.dynamic_objects: list[Object3D] = []  # dynamic objects
        self.object_identifier_cnt = 0
        self.frames: list[dict] = []
        self.temporal_info = defaultdict(
            dict
        )  # keys: timestamps; values: dicts containing captions, features, etc.
        self.detection_classes = classes
        self.save_dir = save_dir
        self.seg_model = SAM("data/model_weights/ultralytics/sam2_b.pt")
        self.det_model = YOLOWorld("data/model_weights/ultralytics/yolov8x-world.pt")
        self.classes = classes
        self.det_model.set_classes(self.classes)
        self.clip_model, self.clip_transform = clip.load("data/model_weights/CLIP/ViT-L-14-336px.pt", device="cuda")
        self.dinov2_model = torch.hub.load('data/model_weights/facebookresearch_dinov2_main', 'dinov2_vitg14', source="local").cuda()
        self.dinov2_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.dynamic = dynamic


    def visual_processing(self, rgb, produce_object_feature=True):
        context_image = Image.fromarray(rgb)
        context_image = self.clip_transform(context_image).unsqueeze(0).to("cuda")
        with torch.no_grad():
            context_emb = (
                self.clip_model.encode_image(context_image).cpu().squeeze(0).numpy()
            )
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self.det_model.predict(bgr)[0]  # bgr input
        boxes = results.boxes
        if boxes.cls.shape[0] == 0:
            output_dict = {
                "context_clip_emb": context_emb,
                "cls_ids": None,
                "bboxes": None,
                "conf": None,
                "masks": None,
                "object_clip_embs": [],
                "object_dinov2_embs": [],
            }
            return output_dict
        cls = boxes.cls.cpu().numpy()
        xywh = boxes.xywh.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        bboxes = []
        for i in range(xywh.shape[0]):
            x, y, w, h = list(xywh[i])
            left_top_x = int(x - w / 2)
            left_top_y = int(y - h / 2)
            right_bottom_x = int(x + w / 2)
            right_bottom_y = int(y + h / 2)
            bboxes.append([left_top_x, left_top_y, right_bottom_x, right_bottom_y])
        results = self.seg_model(bgr, bboxes=bboxes, verbose=False)[0]
        mask_contours = results.masks.xy

        output_dict = {
            "context_clip_emb": context_emb,
            "cls_ids": cls,
            "bboxes": bboxes,
            "conf": conf,
            "masks": mask_contours,
            "object_clip_embs": [],
            "object_dinov2_embs": [],
        }

        if produce_object_feature and len(bboxes) != 0:
            clip_batch_inputs = []
            dinov2_batch_inputs = []
            #first add object crop input
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                object_crop = rgb[y1:y2, x1:x2]
                img = Image.fromarray(object_crop)
                clip_input = self.clip_transform(img)
                dinov2_input = self.dinov2_transform(img)
                clip_batch_inputs.append(clip_input)
                dinov2_batch_inputs.append(dinov2_input)
            clip_batch_inputs = torch.stack(clip_batch_inputs).to("cuda")
            dinov2_batch_inputs = torch.stack(dinov2_batch_inputs).to("cuda")
            with torch.no_grad():
                clip_embs = (
                    self.clip_model.encode_image(clip_batch_inputs).cpu().numpy()
                )
                dinov2_embs = self.dinov2_model(dinov2_batch_inputs).cpu().numpy()

            output_dict = {
                "context_clip_emb": context_emb, # numpy
                "cls_ids": cls, # numpy
                "bboxes": bboxes, # list
                "conf": conf,
                "masks": mask_contours, # list of numpy
                "object_clip_embs": clip_embs, # numpy
                "object_dinov2_embs": dinov2_embs, # numpy
            }
            return output_dict
        return output_dict


    # def visualization(self):
    #     xyzs = np.concatenate(self.xyzs, axis=0)
    #     colors = np.concatenate(self.colors, axis=0)
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(xyzs)
    #     pcd.colors = o3d.utility.Vector3dVector(colors)
    #     o3d.visualization.draw_geometries([pcd])


    def process_a_frame(
            self,
            timestamp,
            rgb,
            depth,
            depth_mask,
            pos,
            rmat,
            fov,
            produce_object_feature=True,
        ):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        xyz = depth2d_to_world3d_transformation(
            depth=depth, position=pos, rmat=rmat, hfov=fov
        )
        frame_info = self.visual_processing(
            rgb=rgb, produce_object_feature=produce_object_feature
        )
        context_emb = frame_info["context_clip_emb"]
        bboxes = frame_info["bboxes"]
        cls = frame_info["cls_ids"]
        mask_contours = frame_info["masks"]
        object_clip_embs = frame_info["object_clip_embs"]
        object_dinov2_embs = frame_info["object_dinov2_embs"]
 
        self.temporal_info[timestamp]["clip_feature"] = context_emb
        visible_identifiers = []
        updated_indices = []
        identifier2image_identifier = {}
        if cls is None:
            self.temporal_info[timestamp]["visible_object_identifiers"] = (
                visible_identifiers
            )
            return plot_visible_object_bboxes(self.static_objects, bgr, pos, rmat, fov, depth)
        # print("detected objects: ", len(cls))
        new_objects: list[Object3D] = []
        # mask_start_time = time.time()
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox
            h, w = rgb.shape[:2]
            category = self.detection_classes[int(cls[i])]
            if y1 <= 0.06 * h or x1 <= 0.06 * w or y2 >= 0.94 * h or x2 >= 0.94 * w:
                # make sure the detection is in the frame boundary
                continue
            if produce_object_feature:
                clip_emb = object_clip_embs[i]
                dinov2_emb = object_dinov2_embs[i]
            else:
                clip_emb = None
                dinov2_emb = None
            mask_contour = mask_contours[i]
            b_mask = np.zeros(rgb.shape[:2], np.uint8)
            contour = mask_contour.astype(np.int32).reshape(-1, 1, 2)
            # draw contour onto mask
            cv2.drawContours(b_mask, [contour], -1, (1, 1, 1), cv2.FILLED)  # mask
            b_mask = b_mask.astype(bool)
            b_mask = b_mask & depth_mask
            # get the object surface pcd
            object_surface_xyz = xyz[b_mask].reshape(-1, 3)
            object_surface_rgb = rgb[b_mask].reshape(-1, 3)
            # trick to delete outlier pcd (background points and foreground points)
            # caused by imperfect mask segmentation
            center = np.mean(object_surface_xyz, axis=0)
            direction = center - pos
            direction = direction / np.linalg.norm(direction)  # camera ray direction
            vectors = object_surface_xyz - pos
            projection_lengths = np.dot(vectors, direction)
            filter_num = int(object_surface_xyz.shape[0] * 0.1)
            indices = np.argsort(projection_lengths)[
                filter_num:-filter_num
            ]  # filter out the foreground and background points
            object_surface_xyz = object_surface_xyz[indices]
            object_surface_rgb = object_surface_rgb[indices]
            if object_surface_xyz.shape[0] == 0:
                continue
            object_max_xyz = np.max(object_surface_xyz, axis=0)
            object_min_xyz = np.min(object_surface_xyz, axis=0)
            obj = Object3D(
                identifier=f"tmp_{i}",
                category=category,
                min_xyz=object_min_xyz,
                max_xyz=object_max_xyz,
                object_clip_feature=clip_emb,
                object_dinov2_feature=dinov2_emb,
                context_clip_feature=context_emb,
            )
            new_objects.append(obj)
        if self.dynamic == True:
            self.static_objects, tmp_dynamic_objects = split_static_dynamic_objects(
                old_objects=self.static_objects,
                depth=depth,
                mask=depth_mask,
                pos=pos,
                rot=rmat,
                hfov=fov
                )
            self.dynamic_objects += tmp_dynamic_objects
            dynamic_ids = []
            for obj in self.dynamic_objects:
                dynamic_ids.append(obj.identifier)
            print("dynamic ids: ", dynamic_ids)
        matching = static_object_reid(
            old_objects=self.static_objects, new_objects=new_objects
        )
        # print(matching)
        static_merged_new_identifiers = []
        for pair in matching:
            obj1 = None
            obj2 = None
            updated_idx = None
            for obj1 in new_objects:
                if pair[0] == obj1.identifier:
                    break
            for idx, obj2 in enumerate(self.static_objects):
                if pair[1] == obj2.identifier:
                    updated_idx = idx
                    break
            # print("merged")
            if obj1 is not None and obj2 is not None:
                obj2.merge(target_object=obj1, ratio=0.2)
                updated_indices.append(updated_idx)
                static_merged_new_identifiers.append(obj1.identifier)
                visible_identifiers.append(obj2.identifier)
                identifier2image_identifier[obj2.identifier] = int(
                    obj1.identifier.split("_")[1]
                )
        new_objects = [
            obj
            for obj in new_objects
            if obj.identifier not in static_merged_new_identifiers
        ]
        if self.dynamic == True:
            matching = dynamic_object_reid(old_objects=self.dynamic_objects, new_objects=new_objects)
            print(matching)
            dynamic_merged_new_identifiers = []
            matched_dynamic_indices = []
            
            for pair in matching:
                obj1 = None
                obj2 = None
                for obj1 in new_objects:
                    if obj1.identifier == pair[0]:
                        break
                for idx, obj2 in enumerate(self.dynamic_objects):
                    if obj2.identifier == pair[1]:
                        matched_dynamic_indices.append(idx)
                        break
                obj2.merge(target_object=obj1, ratio=1.0)
                dynamic_merged_new_identifiers.append(obj1.identifier)
                visible_identifiers.append(obj2.identifier)
                identifier2image_identifier[obj2.identifier] = int(obj1.identifier.split("_")[1])
            for idx in matched_dynamic_indices:
                self.static_objects.append(self.dynamic_objects[idx]) # remove the matched dynamic candidates to static objects
            self.dynamic_objects = [self.dynamic_objects[idx] for idx in range(len(self.dynamic_objects)) if idx not in matched_dynamic_indices]
            new_objects = [obj for obj in new_objects if obj.identifier not in dynamic_merged_new_identifiers]

        for obj in new_objects:
            identifier2image_identifier[self.object_identifier_cnt] = int(
                obj.identifier.split("_")[1]
            )
            obj.identifier = self.object_identifier_cnt
            self.object_identifier_cnt += 1
            self.static_objects.append(obj)
            updated_indices.append(len(self.static_objects) - 1)
            visible_identifiers.append(obj.identifier)

        updated_indices = sorted(list(set(updated_indices)))
        #print(len(self.static_objects), updated_indices)
        self.static_objects = remove_duplicate_objects(
            self.static_objects, updated_indices
        )
        h, w = bgr.shape[:2]
        for obj in self.static_objects:
            if obj.identifier in visible_identifiers:  # can be directly viewed
                x1, y1, x2, y2 = bboxes[identifier2image_identifier[obj.identifier]]
                thumbnail_y1 = int(np.clip(y1 - (y2 - y1) * 0.2, 0, h - 1))
                thumbnail_y2 = int(np.clip(y2 + (y2 - y1) * 0.2, 0, h - 1))
                thumbnail_x1 = int(np.clip(x1 - (x2 - x1) * 0.2, 0, w - 1))
                thumbnail_x2 = int(np.clip(x2 + (x2 - x1) * 0.2, 0, w - 1))
                thumbnail_h = thumbnail_y2 - thumbnail_y1
                thumbnail_w = thumbnail_x2 - thumbnail_x1
                if thumbnail_w * thumbnail_h > obj.image_size:
                    if self.save_dir is None:
                        continue
                    object_save_path = os.path.join(
                        self.save_dir, "object", str(obj.identifier)
                    )
                    if not os.path.exists(object_save_path):
                        os.makedirs(object_save_path)

                    thumbnnail_path = os.path.join(object_save_path, "thumbnail.png")
                    cv2.imwrite(
                        thumbnnail_path,
                        bgr[thumbnail_y1:thumbnail_y2, thumbnail_x1:thumbnail_x2],
                    )
                    obj.image_size = thumbnail_w * thumbnail_h
                    obj_in_frame_path = os.path.join(
                        object_save_path, "object_in_frame.png"
                    )
                    cv2.imwrite(
                        obj_in_frame_path,
                        cv2.rectangle(np.copy(bgr), (x1, y1), (x2, y2), (0, 255, 0), 1),
                    )
                    object_item = {}
                    object_item["objID"] = str(obj.identifier)
                    object_item["thumbnnail_path"] = thumbnnail_path
                    object_item["obj_in_frame_path"] = obj_in_frame_path

        self.temporal_info[timestamp]["visible_object_identifiers"] = (
            visible_identifiers
        )
        return plot_visible_object_bboxes(self.static_objects, bgr, pos, rmat, fov, depth)


    def generate_object_memory_for_a_scene(self, scene_loader, save_dir:str, classes):
        video_save_path = os.path.join(save_dir, "video.mp4")
        frame_height, frame_width = scene_loader[0]["rgb"].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 output
        video = cv2.VideoWriter(video_save_path, fourcc, 30, (frame_width, frame_height))
        frame_num = len(scene_loader)
        for i in range(frame_num):
            content = scene_loader[i]
            rgb = content["rgb"]
            depth = content["depth"]
            mask = content["mask"]
            translation = content["translation"]
            rotation = content["rotation"]
            hfov = content["hfov"]
            frame_id = content["frame_id"]
            bgr = self.process_a_frame(timestamp=frame_id, rgb=rgb, depth=depth, depth_mask=mask, pos=translation, rmat=rotation, fov=hfov)
            cv2.imshow("test", bgr)
            cv2.waitKey(0)
        video.release()
        return


if __name__ == "__main__":
    detection_classes = list(customized_classes)
    mem = ObjectMemory(classes=detection_classes, save_dir="test_reid/save_dir", dynamic=True)