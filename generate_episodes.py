from env.simulator import Simulator
import magnum as mn
import numpy as np
from omegaconf import OmegaConf
import os
import random as rd
import numpy as np
import magnum as mn
from env.object_memory import ObjectMemory
import habitat_sim
import random as rd
import json
import time


with open("data/objects/receptacle2relation.json") as f:
    receptacle2relation = json.load(f)


def propose_place_point(sim: Simulator, object_id, receptacle_id, receptacle_category, articulated):
    """propose a place point in/on the receptacle for the object."""
    relation = receptacle2relation[receptacle_category]
    object = sim._rigid_obj_mgr.get_object_by_id(object_id)
    object_min, object_max, object_center = sim.get_rigid_object_bbox(object)
    if articulated:
        receptacle = sim._articulated_obj_mgr.get_object_by_id(receptacle_id)
        receptacle_min, receptacle_max, receptacle_center = sim.get_articulated_object_bbox(receptacle)
    else:
        receptacle = sim._rigid_obj_mgr.get_object_by_id(receptacle_id)
        receptacle_min, receptacle_max, receptacle_center = sim.get_rigid_object_bbox(receptacle)
    
    object_size = (object_max-object_min)/2
    receptacle_size = (receptacle_max-receptacle_min)/2
    if relation == 'in':
        available_size = receptacle_size - object_size
        if np.any(available_size < 0): # object cannot be fit into the receptacle
            return None
        available_min_xyz = receptacle_center-available_size
        available_max_xyz = receptacle_center+available_size
        if receptacle_category == "fridge":
            available_min_xyz[1] = receptacle_center[1]
    else: # relation == 'on'
        temp = receptacle_size - object_size
        x_range = temp[0]-0.1
        z_range = temp[1]-0.1
        if x_range < 0 or z_range < 0:
            return None
        available_min_xyz = np.array([receptacle_center[0]-x_range, receptacle_center[1]+receptacle_size[1]+object_size[1], receptacle_center[2]-z_range])
        available_max_xyz = np.array([receptacle_center[0]+x_range, receptacle_center[1]+receptacle_size[1]+object_size[1]+0.05, receptacle_center[2]+z_range])
    old_object_position = object.translation
    object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
    for i in range(1000):
        random_pos = np.random.uniform(available_min_xyz, available_max_xyz)
        new_object_position = mn.Vector3(random_pos)
        final_translation = None
        for i in range(1000):
            translation = new_object_position - i * mn.Vector3([0.0, 0.01, 0.0])
            object.translation = translation
            if object.contact_test() or np.any(translation-available_min_xyz) < 0:
                break
            final_translation = translation
        if final_translation != None:
            object.translation = old_object_position
            object.motion_type = habitat_sim.physics.MotionType.STATIC
            return final_translation
    object.translation = old_object_position
    object.motion_type = habitat_sim.physics.MotionType.STATIC
    return None


def generate_episode_info(scene_config="data/scenes/home-robot-remake/scenes-uncluttered/104862660_172226844_new.scene_instance.json"):
    sim_settings = OmegaConf.load('config/default_sim_config.yaml') #default simulator settings
    sim_settings["scene_dataset_config_file"] = 'data/scenes/home-robot-remake/hssd-hab-uncluttered.scene_dataset_config.json'
    sim_settings["scene"] = scene_config 
    default_agent_settings = OmegaConf.load('config/default_agent_config.yaml')
    agents_settings = [default_agent_settings]
    sim = Simulator(sim_settings=sim_settings, agents_settings=agents_settings)
    sim.get_articulated_object_rotate_angle()
    sim.update_articulated_id_mapping()
    sim.recompute_navmesh()

    episode_info = dict()
    episode_info["scene"] = scene_config
    path_finder = sim.get_path_finder()
    island_num = path_finder.num_islands
    areas = [path_finder.island_area(island_index=i) for i in range(island_num)]
    max_area = max(areas)
    target_island = areas.index(max_area)
    position = sim.get_path_finder().get_random_navigable_point(max_tries=10, island_index=target_island)
    sim.initialize_agent(agent_id=0, position=position)
    episode_info["agent_position"] = position.tolist()
    episode_info["island_index"] = target_island
    episode_info["objects"] = list()


    
    with open("data/objects/pickable_objects.json") as f:
        pickable = json.load(f)
    with open("data/objects/unpickable_objects.json") as f:
        unpickable = json.load(f)
    with open("data/objects/rigid_receptacles.json") as f:
        receptacle = json.load(f)
    with open("data/objects/articulated_receptacles.json") as f:
        articulated_receptacle = json.load(f)
    with open("data/objects/receptacle2relation.json") as f:
        category2relation = json.load(f)
    
    
    object_memory = ObjectMemory(sim, pickable, unpickable, receptacle, articulated_receptacle, category2relation, save_path=None)
    print(object_memory.receptacle_ids)
    print(object_memory.articulated_receptacle_ids)
    
    
    with open("data/objects/pickable_object_configs.json") as f:
        content = json.load(f)
    categories = list(content.keys())
    handle2template = dict()

    

    for recep_id in object_memory.receptacle_ids:
        recep_category = object_memory.id2object[recep_id].category
        contained_object_num = rd.choice([2, 4])
        for i in range(contained_object_num):
            category = rd.choice(categories)
            config_file = rd.choice(content[category])
            obj = sim.load_object(object_config_path=config_file, translation=[0.0, 0.0, 0.0])
            position = propose_place_point(sim=sim, object_id=obj.object_id, receptacle_id=recep_id, receptacle_category=recep_category, articulated=False)
            print(position)
            if position == None:
                position = sim.get_path_finder().get_random_navigable_point(max_tries=10, island_index=target_island)
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj.translation = position
            handle2template[obj.handle] = config_file
            #episode_info["objects"][config_file] = np.array(position).tolist()
    
    for recep_id in object_memory.articulated_receptacle_ids:
        recep_category = object_memory.id2object[recep_id].category
        if recep_category == 'fridge':
            contained_object_num = rd.choice([1, 2])
        else:
            contained_object_num = rd.choice([0, 1])
        for i in range(contained_object_num):
            category = rd.choice(["egg", "tomato", "spicemill", "butter_dish", "pitcher", "candy_bar", "apple", "can", "potato", "bread", "bowl", "plate", "glass", "sushi_mat"])
            config_file = rd.choice(content[category])
            obj = sim.load_object(object_config_path=config_file, translation=[0.0, 0.0, 0.0])
            position = propose_place_point(sim=sim, object_id=obj.object_id, receptacle_id=recep_id, receptacle_category=recep_category, articulated=True)
            print(position)
            if position == None:
                position = sim.get_path_finder().get_random_navigable_point(max_tries=10, island_index=target_island)
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
            obj.translation = position
            handle2template[obj.handle] = config_file
            #episode_info["objects"][config_file] = np.array(position).tolist()


    for i in range(600):
        sim.step(actions="no_op")
    
    sim._simulator.close()
    return episode_info


if __name__ == "__main__":
    start_time = time.time()
    episode_base_dir = "data/scenes/home-robot-remake/new_episodes"
    for i in range(10):
        scene_config_path = "data/scenes/home-robot-remake/scenes-uncluttered/108736872_177263607.scene_instance.json"
        scene_id = os.path.basename(scene_config_path).split('.')[0]
        res = generate_episode_info(scene_config=scene_config_path)
        save_path = os.path.join(episode_base_dir, scene_id, f"layout_{i}")
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        with open(os.path.join(save_path, f"episode.json"), 'w') as f:
            json.dump(res, f)

    end_time = time.time()
    print(end_time-start_time)