from env.simulator import Simulator
from omegaconf import OmegaConf
import os
from object_memory import ObjectMemory
import json
import cv2
import numpy as np
import quaternion
from detection_classes import customized_classes


FORWARD_KEY="w"
BACKWARD_KEY='s'
LEFT_KEY="a"
RIGHT_KEY="d"
LOOK_UP_KEY = '1'
LOOK_DOWN_KEY = '2'
GRAB_RELEAESE_KEY = 'b'
OPEN_CLOSE_KEY = 'e'
INFO_KEY = 'i'
FINISH="f"
DEPTH_KEY="l"
NO_OP_KEY = 'n'

ARM_KEYS = ["t", "y", "u", "i", "o", "p", "["]
ARM_KEYS_REVERSE = ["g", "h", "j", "k", "l", ";", "'"]
ARM_RESET_KEY="r"

keyboard_status = []
action_pool = ["no_op", "move_forward", "move_backward", "turn_left", "turn_right", "look_up", "look_down", "grab_release", 'open_close', 'info']


def on_press(key):
    if key not in keyboard_status:
        keyboard_status.append(key)

def on_release(key):
    if key in keyboard_status:
        keyboard_status.remove(key)


if __name__ == "__main__":
    episode_dir="data/scenes/home-robot-remake/episodes/104862660_172226844_new/layout_0"
    with open(os.path.join(episode_dir, "episode.json")) as f:
        episode = json.load(f)
    sim_settings = OmegaConf.load('config/default_sim_config.yaml') #default simulator settings
    sim_settings["scene_dataset_config_file"] = 'data/scenes/home-robot-remake/hssd-hab-uncluttered.scene_dataset_config.json'
    sim_settings["scene"] = episode["scene"]
    #sim_settings["scene"] = "data/scenes/home-robot-remake/scenes-uncluttered/104862660_172226844.scene_instance.json"
    default_agent_settings = OmegaConf.load('config/default_agent_config.yaml')
    agents_settings = [default_agent_settings]
    sim = Simulator(sim_settings=sim_settings, agents_settings=agents_settings)
    sim.get_articulated_object_rotate_angle()
    sim.update_articulated_id_mapping()
    sim.recompute_navmesh()
    agent_position = episode["agent_position"]
    sim.initialize_agent(agent_id=0, position=agent_position)
    for obj_config_file in episode["objects"]:
        obj_translation = episode["objects"][obj_config_file]
        sim.load_object(object_config_path=obj_config_file, translation=obj_translation)
    target_island = episode["island_index"]
    
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

    # image_saving_path = os.path.join(episode_dir, "user_images")
    # if not os.path.exists(image_saving_path):
    #     os.makedirs(image_saving_path)
    object_memory_path = "test_reid/object_memory"
    object_memory = ObjectMemory(classes=customized_classes, save_dir="test_reid/object_memory")
    
    hfov = default_agent_settings["hfov"]
    timestamp = 0
    observations = sim.step(actions="no_op")
    rgb = observations[0]["rgb_1st"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imshow('test', bgr)
    while True:
        sgn = False
        keystroke = cv2.waitKey(0)
        if keystroke == ord(FORWARD_KEY):
            print("action: FORWARD")
            action = 'move_forward'
        elif keystroke == ord(BACKWARD_KEY):
            print("action: BACKWARD")
            action = 'move_backward'
        elif keystroke == ord(LEFT_KEY):
            print("action: LEFT")
            action = 'turn_left'
        elif keystroke == ord(RIGHT_KEY):
            print("action: RIGHT")
            action = 'turn_right'
        elif keystroke == ord(LOOK_UP_KEY):
            print("action: LOOK UP")
            action = 'look_up'
        elif keystroke == ord(LOOK_DOWN_KEY):
            print("action: LOOK DOWN")
            action = 'look_down'
        elif keystroke == ord(GRAB_RELEAESE_KEY):
            print("action: GRAB_RELEASE")
            action = 'grab_release'
        elif keystroke == ord(OPEN_CLOSE_KEY):
            print("action: OPEN_CLOSE")
            action = 'open_close'
        elif keystroke == ord(INFO_KEY):
            print("action: INFO")
            action = 'info'
        elif keystroke == ord(DEPTH_KEY):
            print("action: DEPTH")
            sgn = True
            action = 'no_op'
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
        observations = sim.step(actions=action)
        rgb = observations[0]["rgb_1st"][:, :, :3]
        depth = observations[0]["depth_1st"]
        pos = sim._simulator.get_agent(0).state.sensor_states['depth_1st'].position
        print(rgb.shape)
        print(depth.shape)
        rot = quaternion.as_rotation_matrix(sim._simulator.get_agent(0).state.sensor_states['depth_1st'].rotation)            
        rot = rot @ np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
        depth_mask = np.ones_like(depth, dtype=bool)
        bgr = object_memory.process_a_frame(
            timestamp=timestamp,
            rgb=rgb,
            depth=depth,
            depth_mask=depth_mask,
            pos=pos,
            rmat=rot,
            fov=hfov
        )
        cv2.imshow("test", bgr)