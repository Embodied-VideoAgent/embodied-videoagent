import numpy as np
import quaternion
import habitat_sim
import cv2
import magnum as mn
from collections import Counter
from habitat_sim.nav import MultiGoalShortestPath
from sklearn.cluster import DBSCAN
from env.object_memory import ObjectMemory
from env.simulator import Simulator
from utils import plot_object_bbox_on_frame
import os


class Executor:
    def __init__(self, sim: Simulator, object_memory: ObjectMemory, island_index, hfov=79, fps=30, visualize=True, plot_text=True, save_dir=None):
        self.object_memory = object_memory
        self.sim = sim
        self.fps = fps
        self.hfov = hfov
        self.visualize = visualize
        self.plot_text = plot_text
        self.save_dir = save_dir
        if self.save_dir != None and not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        points = self.sim._simulator.pathfinder.build_navmesh_vertices(island_index=island_index)
        # print(points)
        # print(Counter(np.array(points)[:, 1]))
        ground_height = list(Counter(np.array(points)[:, 1]))[0]

        points = list(filter(lambda x: np.abs(x[1] - ground_height) < 0.01, points))
        
        threshold = 0.5
        dbscan = DBSCAN(eps=threshold, min_samples=1)
        index = dbscan.fit_predict(points)
        tmp = {}
        for ind, p in zip(index, points):
            if ind not in tmp:
                tmp[ind] = []
            tmp[ind].append(p)
        self.grouped_points = []
        for k, v in tmp.items():
            self.grouped_points.append(np.mean(np.array(v), axis=0))
        self.point_visit = [False for _ in range(len(self.grouped_points))]
        self.exploration_done = False
        self.step_cnt = 0
        return


    def plot_bbox_annotation(self, bgr, pos, rot):
        """plot bboxes of the objects in view."""
        for object_id in self.object_memory.id2object:
            if object_id in self.object_memory.unpickable_ids:
                continue
            obj = self.object_memory.id2object[object_id]
            if obj.visible == True:
                plot_object_bbox_on_frame(obj, bgr, pos, rot, self.hfov)
        return bgr


    def execute_steps(self, action_list=[], cross_hair=True, bounding_box=True, text=None):
        """execute the steps and update the object memory."""
        self.step_cnt += len(action_list)
        for action in action_list:
            if action == None:
                break
            observations = self.sim.step(actions=action)
            rgb = observations[0]["rgb_1st"]
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            depth = observations[0]["depth_1st"]
            pos = self.sim._simulator.get_agent(0).state.sensor_states['depth_1st'].position
            rot = quaternion.as_rotation_matrix(self.sim._simulator.get_agent(0).state.sensor_states['depth_1st'].rotation)            
            rot = rot @ np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ]) # habitat-sim camera coordinate (right: x, up: y, front: -z) transformed to standard (right: x, up: -y, front: z) 
            self.object_memory.update(depth=depth, bgr=bgr, pos=pos, rot=rot, hfov=self.hfov)
            if bounding_box:
                bgr = self.plot_bbox_annotation(bgr, pos, rot)
            if cross_hair:
                w, h = bgr.shape[:2]
                cx, cy = w // 2, h // 2
                l = max(w * h // 100000, 1)
                thickness = max(w * h // 300000, 1)
                cv2.line(bgr, (cy, cx-l), (cy, cx+l), color=(0,255,0), thickness=thickness)
                cv2.line(bgr, (cy-l, cx), (cy+l, cx), color=(0,255,0), thickness=thickness)
            if self.plot_text == True and text != None:
                font = cv2.FONT_HERSHEY_COMPLEX 
                font_scale = 1
                color = (255, 255, 255)
                thickness = 2
                text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
                text_x = 10
                text_y = text_size[1] + 10
                cv2.putText(bgr, text, (text_x, text_y), font, font_scale, color, thickness)
            if self.visualize:
                cv2.imshow("Observations", bgr)
                cv2.waitKey(int(1/self.fps*1000))
            if self.save_dir != None:
                cv2.imwrite(os.path.join(self.save_dir, f"step_{self.step_cnt}.png"), bgr)


    def propose_place_point(self, object, receptacle, relation):
        """propose a place point in/on the receptacle for the object."""
        object_id = self.object_memory.name2id[object]
        receptacle_id = self.object_memory.name2id[receptacle]
        object = self.object_memory.id2object[object_id]
        receptacle = self.object_memory.id2object[receptacle_id]
        object_size = (object.max_xyz-object.min_xyz)/2
        receptacle_size = (receptacle.max_xyz-receptacle.min_xyz)/2
        if relation == 'in':
            available_size = receptacle_size - object_size
            if np.any(available_size < 0): # object cannot be fit into the receptacle
                return None
            available_min_xyz = receptacle.position-available_size
            available_max_xyz = receptacle.position+available_size
        else: # relation == 'on'
            temp = receptacle_size - object_size
            x_range = temp[0]-0.1
            z_range = temp[1]-0.1
            if x_range < 0 or z_range < 0:
                return None
            receptacle_center = receptacle.position
            available_min_xyz = np.array([receptacle_center[0]-x_range, receptacle_center[1]+receptacle_size[1]+object_size[1], receptacle_center[2]-z_range])
            available_max_xyz = np.array([receptacle_center[0]+x_range, receptacle_center[1]+receptacle_size[1]+object_size[1]+0.05, receptacle_center[2]+z_range])
        object = self.sim._rigid_obj_mgr.get_object_by_id(object.identifier)
        old_object_position = object.translation
        object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        for i in range(1000):
            random_pos = np.random.uniform(available_min_xyz, available_max_xyz)
            new_object_position = mn.Vector3(random_pos)
            final_translation = None
            for i in range(1000):
                translation = new_object_position - i*mn.Vector3([0.0, 0.01, 0.0])
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
        

    def get_distance_to_target(self, position):
        """get the distance between the agent and the position."""
        agent_state = self.sim.get_agent_state(agent_id=0)
        agent_position = agent_state.position
        return np.linalg.norm(agent_position-position)


    def simple_goto(self, goal, text):
        """go to a [x, y, z]."""
        actions = self.sim.reset_to_horizontal(agent_id=0)
        self.execute_steps(action_list=actions)
        goal_radius = 0.1
        follower = self.sim._simulator.make_greedy_follower(0, goal_radius)
        while True:
            try:
                a = follower.next_action_along(goal)
            except:
                return
            # goal is reached
            if a is None:
                return
            self.execute_steps(action_list=[a], text=text)


    def search(self, target):
        """Search for the target object by exploration."""
        explored_object_names = []
        for id in self.object_memory.explored_ids:
            obj = self.object_memory.id2object[id]
            explored_object_names.append(obj.name)
        
        target_objects = []
        parts = target.split(" ")
        for name in explored_object_names:
            sgn = True
            for part in parts:
                if part not in name:
                    sgn = False
                    break
            if sgn == True:
                target_objects.append(name)


        # target_object = llm_select_object(target, explored_object_names)
        if len(target_objects) != 0:
            #self.goto(target_object)
            return f"Target objects found in object memory! Their names are {target_objects}."
        while all(self.point_visit) is False and self.exploration_done == False:
            path = MultiGoalShortestPath()
            path.requested_start = self.sim.get_agent_state().position
            inds = [i[1] for i in list(filter(lambda x: not x[0], zip(self.point_visit, range(len(self.grouped_points)))))]
            path.requested_ends = [i[1] for i in list(filter(lambda x: not x[0], zip(self.point_visit, self.grouped_points)))]
            if self.sim._simulator.pathfinder.find_path(path):
                goal = self.grouped_points[inds[path.closest_end_point_index]]
                print('Current goal: ', goal, 'distance: ', path.geodesic_distance)
                self.simple_goto(goal, f"SEARCH {target}")
                self.point_visit[inds[path.closest_end_point_index]] = True
                explored_object_names = []
                for id in self.object_memory.explored_ids:
                    obj = self.object_memory.id2object[id]
                    explored_object_names.append(obj.name)
                   
                target_objects = []
                parts = target.split(" ")
                for name in explored_object_names:
                    sgn = True
                    for part in parts:
                        if part not in name:
                            sgn = False
                            break
                    if sgn == True:
                        target_objects.append(name)

                #target_object = llm_select_object(target, explored_object_names)
                if len(target_objects) != 0:
                    #self.goto(target_object)
                    #self.goto(target_object)
                    return f"Target object found! Their names are {target_objects}."
            else:
                self.exploration_done = True
                break
        return f"All places except articulated receptacles have been searched but no {target} was found!"
    
    
    def exhaustive_exploration(self):
        """Exhaustively explore the whole appartment."""
        while all(self.point_visit) is False and self.exploration_done == False:
            path = MultiGoalShortestPath()
            path.requested_start = self.sim.get_agent_state().position
            inds = [i[1] for i in list(filter(lambda x: not x[0], zip(self.point_visit, range(len(self.grouped_points)))))]
            path.requested_ends = [i[1] for i in list(filter(lambda x: not x[0], zip(self.point_visit, self.grouped_points)))]
            if self.sim._simulator.pathfinder.find_path(path):
                goal = self.grouped_points[inds[path.closest_end_point_index]]
                print('Current goal: ', goal, 'distance: ', path.geodesic_distance)
                self.simple_goto(goal)
                self.point_visit[inds[path.closest_end_point_index]] = True
            else:
                self.exploration_done = True
                break
        return


    def goto(self, target):
        """Go to the target (receptacle or object) and look at it."""
        target_position = None
        if target not in self.object_memory.name2id:
            return "Invalid target!"
        object_id = self.object_memory.name2id[target]
        # Determine whether it is an articulated object
        object_status_feedback = ""
        if object_id in self.object_memory.articulated_receptacle_ids:
            go_target_position = self.sim.get_articulated_object_nav_point(object_id)
            look_target_position = self.sim._articulated_obj_mgr.get_object_by_id(object_id).translation
            recep_state = self.sim.articulated_object_state[object_id]
            object_status_feedback = f"target is currently {recep_state}."
        else:
            go_target_position = look_target_position = self.sim._rigid_obj_mgr.get_object_by_id(object_id).translation
        action_list = self.sim.goto_action(target_position=go_target_position)
        self.execute_steps(action_list=action_list, text=f"GOTO {target}")
        action_list = self.sim.look_action(target_position=look_target_position)
        self.execute_steps(action_list=action_list, text=f"GOTO {target}")
        exploration_feedback = self.object_memory.get_explored_objects()
        view_feedback = self.object_memory.get_objects_in_view()
        #return f"Go to {target} successfully! {exploration_feedback} {object_status_feedback} {view_feedback}"
        return f"Go to {target} successfully! {object_status_feedback} {view_feedback}"
        

    def open(self, target):
        """Open an articulated object in view."""
        if target not in self.object_memory.name2id:
            return f"{target} does not exist!"
        object_id = self.object_memory.name2id[target]
        if object_id not in self.object_memory.articulated_receptacle_ids:
            return f"{target} is not an articulated receptacle!"
        # check the distance between agent and receptacle
        object = self.sim._articulated_obj_mgr.get_object_by_id(object_id)
        object_position = np.array(object.translation)
        distance = self.get_distance_to_target(object_position)
        if distance > 2.0: # too far away
            return f"{target} is too far away!"
        action_list = self.sim.look_action(target_position=object_position)
        self.execute_steps(action_list, text=f"OPEN {target}")
        object_state = self.sim.articulated_object_state[object_id]
        joint_velocities = object.joint_velocities
        if object_state == 'open':
            return f'{target} is already open!'
        new_joint_velocities = [5.0 for _ in range(len(joint_velocities))]
        self.sim.articulated_object_state[object.object_id] = 'open'
        object.joint_velocities = new_joint_velocities
        action_list = ["no_op" for i in range(80)]
        self.execute_steps(action_list, text=f"OPEN {target}")
        view_feedback = self.object_memory.get_objects_in_view()
        return f"Open {target} successfully! {view_feedback}"


    def close(self, target):
        """Close an articulated object in view."""
        if target not in self.object_memory.name2id:
            return f"{target} does not exist!"
        object_id = self.object_memory.name2id[target]
        if object_id not in self.object_memory.articulated_receptacle_ids:
            return f"{target} is not an articulated receptacle!"
        # check the distance between agent and receptacle
        object = self.sim._articulated_obj_mgr.get_object_by_id(object_id)
        object_position = np.array(object.translation)
        distance = self.get_distance_to_target(object_position)
        if distance > 2.0: # too far away
            return f"{target} is too far away!"
        action_list = self.sim.look_action(target_position=object_position)
        self.execute_steps(action_list, text=f"CLOSE {target}")
        object_state = self.sim.articulated_object_state[object_id]
        joint_velocities = object.joint_velocities
        if object_state == 'closed':
            return f'{target} is already closed!'
        new_joint_velocities = [-5.0 for _ in range(len(joint_velocities))]
        self.sim.articulated_object_state[object.object_id] = 'closed'
        object.joint_velocities = new_joint_velocities
        action_list = ["no_op" for i in range(80)]
        self.execute_steps(action_list, text=f"CLOSE {target}")
        return f"Close {target} successfully!"


    def pick(self, target):
        """Pick the target object in view."""
        inventory_object_id = self.object_memory.inventory_object_id
        if inventory_object_id != None:
            inventory_object = self.object_memory.id2object[inventory_object_id]
            return f"You have {inventory_object.name} in your hand currently. Cannot pick another object!"
        target_position = None
        if target not in self.object_memory.name2id:
            return f"{target} does not exist!"
        object_id = self.object_memory.name2id[target]
        if object_id not in self.object_memory.pickable_ids:
            return f"{target} cannot be picked up!"
        object = self.object_memory.id2object[object_id]
        target_position = object.position
        distance = self.get_distance_to_target(target_position)
        if distance > 2.0:
            return f"{target} is too far way!"
        action_list = self.sim.look_action(target_position=target_position)
        self.execute_steps(action_list, text=f"PICK {target}")
        if object.visible == False:
            return f"{target} is occluded and you have difficulty in picking up {target}!"
        object = self.sim._rigid_obj_mgr.get_object_by_id(object_id)
        position = object.translation 
        new_position = mn.Vector3(
                position.x,
                position.y-10000.0,
                position.z
            )
        object.translation = new_position
        object.motion_type = habitat_sim.physics.MotionType.STATIC
        self.execute_steps(['no_op'], text=f"PICK {target}")
        self.object_memory.inventory_object_id = object_id
        return f"Pick up {target} successfully!"


    def place(self, target) -> str:
        """Place the inventory object in/on the target receptacle."""
        inventory_object_id = self.object_memory.inventory_object_id
        if inventory_object_id == None:
            return f"You have no object in your hand! Cannot perform PLACE action."
        if target not in self.object_memory.name2id:
            return f"{target} does not exist!"
        receptacle_id = self.object_memory.name2id[target]
        if receptacle_id not in self.object_memory.receptacle_ids and receptacle_id not in self.object_memory.articulated_receptacle_ids:
            return f"{target} is not a receptacle!"
        receptacle = self.object_memory.id2object[receptacle_id]
        relation = receptacle.contain_type
        object = self.object_memory.id2object[inventory_object_id]
        
        recep_position = receptacle.position
        distance = self.get_distance_to_target(recep_position)
        if distance > 2.0:
            return f"{target} is too far way!"
        if receptacle_id in self.object_memory.articulated_receptacle_ids:
            if self.sim.articulated_object_state[receptacle_id] == 'closed':
                return f"Cannot perform PLACE action because {target} is currently closed!"
        target_position = self.propose_place_point(object=object.name, receptacle=receptacle.name, relation=relation)
        if target_position is None:
            return f"Cannot place {object.name} {relation} {receptacle.name}!"
        action_list = self.sim.look_action(target_position=target_position)
        self.execute_steps(action_list=action_list, text=f"PLACE {target}")
        
        sim_object = self.sim._rigid_obj_mgr.get_object_by_id(object.identifier)
        sim_object.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        sim_object.translation = target_position
        action_list = ['no_op' for i in range(30)]
        self.execute_steps(action_list=action_list, text=f"PLACE {target}")
        self.object_memory.inventory_object_id = None
        return f"Place {object.name} {relation} {receptacle.name} successfully!"

