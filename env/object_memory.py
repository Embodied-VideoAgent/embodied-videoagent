from collections import defaultdict
import os
from env.simulator import Simulator
from PIL import Image
import torch
from env.utils import compute_cosine_similarity, top_k_indices
import clip
import numpy as np
import copy
import cv2
from typing import Union, Optional
from object3d import Object3D
from utils import check_visible_and_crop, plot_object_bbox_on_frame


class SimObject(Object3D):
    def __init__(
            self,
            identifier: Union[int, str],
            category: str,
            name: str,
            obj_type: str,
            contain_type: str,
            min_xyz: np.ndarray,  # min_xyz of the aabb bounding box
            max_xyz: np.ndarray,  # max_xyz of the aabb bounding box            
            description: Optional[str] = None,
            context_clip_feature: Optional[np.ndarray] = None, # the clip feature of the video frames containing the object
            object_clip_feature: Optional[np.ndarray] = None, # the clip feature of the cropped object images
            object_dinov2_feature: Optional[np.ndarray] = None # the dinov2 feature of the cropped object images
        ):
        super().__init__(
            identifier,
            category,
            min_xyz,
            max_xyz,
            context_clip_feature,
            description,
            object_clip_feature,
            object_dinov2_feature
        )
        self.name = name
        self.obj_type = obj_type
        self.contain_type = contain_type
        self.which_receptacle = None
        self.contained_objects = []
        self.visible = False


class ObjectMemory:
    def __init__(self, sim: Simulator, pickable, unpickable, receptacle, articulated_receptacle, category2relation, save_path):
        self.pickable_ids = set()
        self.unpickable_ids = set()
        self.receptacle_ids = set()
        self.articulated_receptacle_ids = set()
        self.type2relation = dict()
        self.name2id = dict()
        self.sim = sim
        self.id2object = dict()
        self.explored_ids = set()
        self.inventory_object_id = None

        # get the agent island in order to filter out unreachable receptacles
        agent_state = self.sim.get_agent_state(agent_id=0)
        agent_position = agent_state.position
        agent_island = self.sim.get_island(agent_position)
        all_handles = self.sim.get_all_rigid_objects()

        category2cnt = defaultdict(int)
        for handle in all_handles:
            template = handle.split("_:")[0]
            object = self.sim._rigid_obj_mgr.get_object_by_handle(handle)
            object_position = object.translation
            if self.sim.get_island(object_position) != agent_island: # not navigatable
                continue
            object_id = object.object_id
            # get the bbox of the object
            bb_min, bb_max, center_pos = self.sim.get_rigid_object_bbox(object)
            category = ""
            obj_type = None
            contain_type = None
            if template in pickable:
                self.pickable_ids.add(object_id)
                category = pickable[template]
                obj_type = "pickable"
            elif template in receptacle:
                self.receptacle_ids.add(object_id)
                category = receptacle[template]
                obj_type = "receptacle"
                contain_type = category2relation[category]
            elif template in unpickable:
                self.unpickable_ids.add(object_id)
                category = unpickable[template]
                obj_type = "unpickable"
            else:
                continue
            category2cnt[category] += 1
            name = f"{category}_{category2cnt[category]}"
            self.name2id[name] = object_id
            object = SimObject(identifier=object_id, name=name, category=category, obj_type=obj_type, contain_type=contain_type, min_xyz=bb_min, max_xyz=bb_max)
            self.id2object[object_id] = object

        articulated_handles = self.sim._articulated_obj_mgr.get_object_handles()
        #print(articulated_handles)
        for handle in articulated_handles:
            object = self.sim._articulated_obj_mgr.get_object_by_handle(handle)
            object_id = object.object_id
            self.articulated_receptacle_ids.add(object_id)
            template = handle.split("_:")[0]
            category = articulated_receptacle[template]
            category2cnt[category] += 1
            name = f"{category}_{category2cnt[category]}"
            self.name2id[name] = object_id
            bb_min, bb_max, center_pos = self.sim.get_articulated_object_bbox(object)
            contain_type = category2relation[category]
            #print(bb_min, bb_max, center_pos)
            object = SimObject(identifier=object_id, name=name, category=category, obj_type="articulated_receptacle", contain_type=contain_type, min_xyz=bb_min, max_xyz=bb_max)
            self.id2object[object_id] = object
    
        self.clip_model, self.clip_transform = clip.load("data/model_weights/CLIP/ViT-L-14-336px.pt", device="cuda")
        self.save_path = save_path


    def update(self, depth, bgr, pos, rot, hfov):
        # update the object bounding box
        for object_id in self.pickable_ids:
            object = self.sim._rigid_obj_mgr.get_object_by_id(object_id)
            bb_min, bb_max, center_pos = self.sim.get_rigid_object_bbox(object)
            self.id2object[object_id].min_xyz = bb_min
            self.id2object[object_id].max_xyz = bb_max
            self.id2object[object_id].position = center_pos
        # update in/on relations
        self.update_related_objects()
        # update occlusion state
        rgb_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb_frame)
        context_input = self.clip_transform(img).unsqueeze(0).cuda()
        with torch.no_grad():
            context_emb = self.clip_model.encode_image(context_input).cpu().squeeze(0).numpy()

        for object_id in self.id2object:
            if object_id in self.unpickable_ids:
                continue
            obj = self.id2object[object_id]
            visible, obj_bbox = check_visible_and_crop(obj=obj, depth=depth, pos=pos, rmat=rot, hfov=hfov)
            min_y, max_y, min_x, max_x = obj_bbox
            obj.visible = visible
            if obj.visible == False:
                continue
            self.explored_ids.add(object_id)
            if self.save_path == None:
                continue
            object_img = bgr[min_y: max_y, min_x: max_x]
            object_rgb = Image.fromarray(cv2.cvtColor(object_img, cv2.COLOR_BGR2RGB))
            object_input = self.clip_transform(object_rgb).unsqueeze(0).cuda()
            with torch.no_grad():
                object_emb = self.clip_model.encode_image(object_input).cpu().squeeze(0).numpy()
            obj.object_clip_feature = object_emb
            obj.context_clip_feature = context_emb
            obj_image_dir = os.path.join(self.save_path, obj.name)
            if not os.path.exists(obj_image_dir):
                os.makedirs(obj_image_dir)
            file_name = os.path.join(obj_image_dir, "thumbnail.png")
            cv2.imwrite(file_name, object_img)

            temp_frame = copy.deepcopy(bgr)
            temp_frame = cv2.rectangle(temp_frame, (min_x, min_y), (max_x, max_y), color=(0, 255, 0), thickness=2)
            file_name = os.path.join(obj_image_dir, "object_in_frame.png")
            cv2.imwrite(file_name, temp_frame)


    
    def get_objects_in_view(self):
        object_names = []
        for object_id in self.pickable_ids:
            object = self.id2object[object_id]
            if object.visible == False:
                continue
            agent_state = self.sim.get_agent_state(agent_id=0)
            agent_position = agent_state.position
            distance = np.linalg.norm(agent_position-object.position)
            if distance <= 2.0:
                object_names.append(object.name)
        return f"In the current view, you can see {str(object_names)}."



    def object_in_hand_feedback(self):
        if self.inventory_object_id == None:
            return ""
        else:
            name = self.id2object[self.inventory_object_id]
            return f"You are holding {name} now."


    def get_explored_objects(self):
        object_names = []
        for object_id in self.explored_ids:
            if object_id in self.pickable_ids:
                object = self.id2object[object_id]
                object_names.append(object.name)
        return f"You have found {str(object_names)} during exploration."


    def check_relation(self, receptacle_id, object_id):
        object = self.id2object[object_id]
        receptacle = self.id2object[receptacle_id]
        object_size = (object.max_xyz-object.min_xyz)/2
        if np.all(object.position - receptacle.min_xyz > 0) and np.all(receptacle.max_xyz - object.position > 0):
            return 'in'
        if (receptacle.min_xyz[0] < object.position[0] < receptacle.max_xyz[0]) and (receptacle.min_xyz[2] < object.position[2] < receptacle.max_xyz[2]) and (receptacle.max_xyz[1]+object_size[1]-0.05 < object.position[1] < receptacle.max_xyz[1]+object_size[1]+0.05): 
            return 'on'
        return None


    def update_related_objects(self):
        all_recep_ids = self.receptacle_ids.union(self.articulated_receptacle_ids)
        for recep_id in all_recep_ids:
            self.id2object[recep_id].contained_objects = []
        for object_id in self.pickable_ids:
            self.id2object[object_id].which_receptacle = None
        for recep_id in all_recep_ids:
            for object_id in self.pickable_ids:
                if self.check_relation(recep_id, object_id) != None:
                    self.id2object[recep_id].contained_objects.append(object_id)
                    self.id2object[object_id].which_receptacle = recep_id

        
    def query_object_state(self, object_name):
        if object_name not in self.name2id:
            return f"{object_name} does not exist!"
        object = self.id2object[self.name2id[object_name]]
        location_feedback = f"{object_name} is not in/on any receptacle."
        if object.which_receptacle != None:
            receptacle = self.id2object[object.which_receptacle]
            location_feedback = f" {object_name} is {receptacle.contain_type} {receptacle.name}."
        contained_object_feedback = ""
        if object.type == "receptacle" or object.type == "articulated_receptacle":
            contained_objects = []
            for obj_id in object.contained_objects:
                contained_objects.append(self.id2object[obj_id].name)
            contained_object_feedback = f" From past observations, {contained_objects} are {object.contain_type} {object.name}."
        in_hand_feedback = ""
        if object.id == self.inventory_object_id:
            in_hand_feedback = f" {object.name} is held by the robot!"
        content = f"{object_name} exists.{location_feedback}{contained_object_feedback}{in_hand_feedback}"
        return content



    def retrieve_objects_by_environment(self, description):
        with torch.no_grad():
            des_emb = self.clip_model.encode_text(clip.tokenize([description]).to("cuda")).cpu().numpy()
        obj_emb_list = []
        id_list = []
        
        for object_id in self.id2object:
            if self.id2object[object_id].visible == True:
                id_list.append(object_id)
                obj_emb_list.append(self.id2object[object_id].ctx_feature)
        scores = compute_cosine_similarity(des_emb, obj_emb_list)
        k_indices = top_k_indices(scores, 10)
        ans = []
        for i in k_indices:
            ans.append(self.id2object[id_list[i]].name)
        return ans


    def retrieve_objects_by_appearence(self, description):
        with torch.no_grad():
            des_emb = self.clip_model.encode_text(clip.tokenize([description]).to("cuda")).cpu().numpy()
        obj_emb_list = []
        id_list = []
        
        for object_id in self.id2object:
            if self.id2object[object_id].visible == True:
                id_list.append(object_id)
                obj_emb_list.append(self.id2object[object_id].obj_feature)
        scores = compute_cosine_similarity(des_emb, obj_emb_list)
        k_indices = top_k_indices(scores, 10)
        ans = []
        for i in k_indices:
            ans.append(self.id2object[id_list[i]].name)
        return ans
    

