import json
from typing import (Any, Dict, List, Sequence, Union)
from omegaconf import DictConfig
from env.agent import Agent
import habitat_sim
import magnum as mn
import numpy as np
from habitat_sim import Simulator as Sim
from habitat_sim.utils.common import quat_from_coeffs, quat_to_magnum
from env.utils import make_agent_cfg, make_sim_cfg, rotate_vector_along_axis, quaternion_to_z_rotation


class Simulator:
    """simulator related"""
    _resolution: List[int] # resolution of all sensors
    _fps: int # each step in the simulator equals to (1/fps) secs in the simulated world
    _config: DictConfig # used to initialize the habitat-sim simulator
    _simulator: Sim # habitat_sim simulator
    
    """agent related"""
    agents: List[Agent] # store agent states during simulation
    num_of_agents: int # number of agents in the simulator
    _agent_object_ids: List[int] # the object_ids of the rigid objects attached to the agents 
    _default_agent_id: int # 0 for the default agent
    holding_object: Dict # a dict(key: agent_id, value: object_id) that records which object the agent is holding
    grab_and_release_distance: float # the maximal distance for all agents to pick or place an object
    
    """object related"""
    _obj_template_mgr: Any # habitat-sim ObjectAttributesManager
    _articulated_obj_mgr: Any # habitat-sim ArticulatedObjectManager
    _rigid_obj_mgr: Any # habitat-sim RigidObjectManager 
    articulated_object_state: Dict # key: articulated_object_id, value: 'open' or 'closed'
    link_id2art_id: Dict # key: link_id, value: the articulated_object_id of the articulated object that the link belongs to
    articulated_id2semantic: Dict
    rigid_id2semantic: Dict
    articulated_object_placeable_points: Dict
    articulated_object_rotate_angle_z: Dict


    def __init__(self, sim_settings, agents_settings) -> None:
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []
        agent_configs = []
        for i in range(len(agents_settings)):
            agent_configs.append(make_agent_cfg(self._resolution, agents_settings[i]))
            self.agents.append(Agent(self, agents_settings[i], i))
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator = Sim(self._config)
        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies

        self._obj_template_mgr = self._simulator.get_object_template_manager()
        self._rigid_obj_mgr = self._simulator.get_rigid_object_manager()
        self._articulated_obj_mgr = self._simulator.get_articulated_object_manager()
        
        self.grab_and_release_distance = sim_settings['grab_and_release_distance']
        self._default_agent_id = sim_settings["default_agent"]
        self.holding_object = dict()
        for agent_id in range(self.num_of_agents):
            agent_object_path = agents_settings[agent_id]['agent_object']
            self.attach_object_to_agent(agent_object_path, agent_id)
            self.holding_object[agent_id] = None
        self.articulated_object_state = dict()
        self.link_id2art_id = dict()
        self.articulated_id2semantic = dict()
        self.rigid_id2semantic = dict()
        self.articulated_object_placeable_points = dict()
        self.articulated_object_rotate_angle_z = dict()
  
    
    def __del__(self):
        self._simulator.close()
    

    def get_all_rigid_objects(self):
        """return the list of all object handles of the rigid objects"""
        return self._rigid_obj_mgr.get_object_handles()


    def remove_rigid_object(self, object_handle):
        """remove an rigid_object by its handle"""
        self._rigid_obj_mgr.remove_object_by_handle(object_handle)


    def remove_articulated_object(self, object_handle):
        """remove an rigid_object by its handle"""
        self._articulated_obj_mgr.remove_object_by_handle(object_handle)


    def update_articulated_id_mapping(self):
        """maintain link_id2art_id, articulated_object_state"""
        for object_handle in self._articulated_obj_mgr.get_object_handles():
            articulated_object = self._articulated_obj_mgr.get_object_by_handle(object_handle) 
            articulated_object_id = articulated_object.object_id
            for link_object_id in articulated_object.link_object_ids:
                self.link_id2art_id[link_object_id] = articulated_object_id
            if articulated_object_id not in self.articulated_object_state:
                self.articulated_object_state[articulated_object_id] = 'closed'
                

    def load_articulated_object(self, urdf_path, semantic=None, position=[0.0, 1.5, 0.0], rotation=mn.Quaternion.rotation(mn.Deg(90.0), [0.0, 1.0, 0.0]), scale=1.0):
        """load an articulated object"""
        obj = self._articulated_obj_mgr.add_articulated_object_from_urdf(urdf_path, fixed_base=True, global_scale=scale)
        obj.translation = position
        obj.rotation = rotation
        obj.motion_type = habitat_sim.physics.MotionType.STATIC
        if semantic != None:
            self.articulated_id2semantic[obj.object_id] = semantic
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agents[self._default_agent_id].agent_height
        navmesh_settings.agent_radius = self.agents[self._default_agent_id].agent_radius
        navmesh_settings.include_static_objects = True
        self._simulator.recompute_navmesh(
            self._simulator.pathfinder,
            navmesh_settings,
        )
        obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        return obj


    def recompute_navmesh(self):
        """load an articulated object"""
        articulated_object_names = self._articulated_obj_mgr.get_object_handles()
        articulated_objects = [self._articulated_obj_mgr.get_object_by_handle(name) for name in articulated_object_names]
        for obj in articulated_objects:
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
        navmesh_settings = habitat_sim.NavMeshSettings()
        navmesh_settings.set_defaults()
        navmesh_settings.agent_height = self.agents[self._default_agent_id].agent_height
        navmesh_settings.agent_radius = self.agents[self._default_agent_id].agent_radius
        navmesh_settings.include_static_objects = True
        self._simulator.recompute_navmesh(
            self._simulator.pathfinder,
            navmesh_settings,
        )
        for obj in articulated_objects:
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC


    def awake_all_objects(self):
        """awake all objects for simulation"""
        rigid_handles = self._rigid_obj_mgr.get_object_handles()
        for handle in rigid_handles:
            obj = self._rigid_obj_mgr.get_object_by_handle(handle)
            obj.awake = True
        articulated_handles = self._articulated_obj_mgr.get_object_handles()
        for handle in articulated_handles:
            obj = self._articulated_obj_mgr.get_object_by_handle(handle)
            obj.awake = True


    def get_agent_state(self, agent_id=0):
        """get the agent state given agent_id"""
        state = self._simulator.agents[agent_id].get_state()
        return state

    def get_object_id_in_hand(self, agent_id=None): 
        """return the object_id of the object in hand, -1 if no object is in hand"""
        if agent_id is None:
            agent_id = self._default_agent_id
        return self.holding_object[agent_id]
    
    
    def set_object_id_in_hand(self, object_id, agent_id=None): 
        """grab an object"""
        if agent_id is None:
            agent_id = self._default_agent_id
        self.holding_object[agent_id] = object_id


    def get_path_finder(self):
        """get the habitat-sim pathfinder"""
        return self._simulator.pathfinder


    def get_island(self, position):
        """get the navmesh id of the position"""
        return self.get_path_finder().get_island(position)


    def unproject(self, agent_id=None):
        """get the crosshair ray of the agent"""
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors["rgb_1st"]
        view_point = mn.Vector2i(self._resolution[1]//2, self._resolution[0]//2)
        ray = sensor.render_camera.unproject(view_point, normalized=True)
        return ray


    def get_camera_info(self, camera_name, agent_id=None):
        if agent_id is None:
            agent_id = self._default_agent_id
        sensor = self._simulator.get_agent(agent_id)._sensors["rgb_1st"]
        camera = sensor.render_camera
        print('camera matrix: ', camera.camera_matrix)
        print('node: ', camera.node)
        print('projection matrix: ', camera.projection_matrix)
        return camera.camera_matrix, camera.projection_matrix



    def info_action(self, agent_id=0):
        """get the info of the object the agent is looking at"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        if hit_object_id == None:
            print("Too far away!")
            return
        if hit_object_id == -1: #the center viewpoint ray is hitting on the stage instead of an object
            print("Hitting on the stage!")
            print(hit_point)
            return
        if hit_object_id in self._agent_object_ids: #hitting on an agent
            print("Hitting on an agent!")
            return
        #the center viewpoint ray is hitting on an object
        print('hit_point: ', hit_point)

        if self._rigid_obj_mgr.get_library_has_id(object_id=hit_object_id): #if it is a rigid object
            object_in_view = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
            print('rigid object handle:', object_in_view.handle)
            print('rigid object position:', object_in_view.translation)
            print('rigid object rotation:', object_in_view.rotation)
        elif self._articulated_obj_mgr.get_library_has_id(object_id=hit_object_id): # hitting on the base of an articulated object
            object_base = self._articulated_obj_mgr.get_object_by_id(hit_object_id)
            print('articulated object base handle:', object_base.handle)
            print('articulated object base position:', object_base.translation)
            print('articulated object base rotation:', object_base.rotation)
            print('articulated object base transformation:', object_base.transformation)
        else:
            hit_object_id = self.link_id2art_id[hit_object_id]
            object_base = self._articulated_obj_mgr.get_object_by_id(hit_object_id)
            print('articulated object base handle:', object_base.handle)
            print('articulated object base position:', object_base.translation)
            print('articulated object base rotation:', object_base.rotation)
            print('articulated object base transformation:', object_base.transformation)


    def goto_action(self, target_position, agent_id=0):
        """return an action list to goto someplace"""
        action_list = []
        actions = self.reset_to_horizontal(agent_id=agent_id)
        action_list += actions[:-1]
        agent_state = self.get_agent_state(agent_id)
        agent_position = agent_state.position
        agent_island = self.get_island(agent_position)
        #project the target position to the agent's navmesh island
        path_finder = self.get_path_finder()
        target_on_navmesh = path_finder.snap_point(point=target_position, island_index=agent_island)
        follower = habitat_sim.GreedyGeodesicFollower(
            path_finder,
            self._simulator.agents[agent_id],
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right")
        try:
            action_list += follower.find_path(target_on_navmesh)
        except:
            pass
        return action_list

    
    def pick_action(self, agent_id=0):
        """atomic pick action"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id))
        if hit_object_id == None: #can neither pick nor place
            return "Not aiming at an object!"
        object_id_in_hand = self.get_object_id_in_hand(agent_id)

        if object_id_in_hand == None: #no object in hand, try to pick an object
            if hit_object_id == -1: #the center viewpoint ray is hitting on the stage instead of an object
                return "Aiming at the stage!"
            if hit_object_id in self._agent_object_ids: #hitting on an agent
                return "Cannot pick an agent!"
            #the center viewpoint ray is hitting on an object
            object_in_hand = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
            if object_in_hand == None:
                return "Cannot pick articulated object!"
            # the object is now legal for grabbing
            object_position = object_in_hand.translation
            new_object_position = mn.Vector3(
                object_position.x,
                object_position.y-10000.0,
                object_position.z
            )
            object_in_hand.translation = new_object_position #change the object altitude to hide it underground
            object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC #set it to static temporarily to avoid continuous fall
            self.set_object_id_in_hand(hit_object_id, agent_id)
            print(hit_object_id)
            return "Pick the object sucessfully!"
        else:
            return "The inventory is not empty!"


    def place_action(self, agent_id=0):
        """atomic place action"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        if hit_object_id == None: #can neither pick nor place
            return "Too far away!"
        object_id_in_hand = self.get_object_id_in_hand(agent_id)
        if object_id_in_hand == None:
            return "The inventory is empty. Cannnot perform place action!"
        object_in_hand = self._rigid_obj_mgr.get_object_by_id(object_id_in_hand)
        object_in_hand.motion_type = habitat_sim.physics.MotionType.DYNAMIC #change the grabbed object to DYNAMIC
        can_place = False
        old_object_position = object_in_hand.translation
        for i in range(10): #start from the hit point, try different object altitude
            #print("try: ", i)
            new_object_position = mn.Vector3(
                hit_point.x,
                hit_point.y+i*0.05,
                hit_point.z
            )
            object_in_hand.translation = new_object_position
            if not object_in_hand.contact_test(): #the object will not contact the collision world
                can_place = True
                break
        if can_place:
            object_in_hand.translation = new_object_position
            self.set_object_id_in_hand(None, agent_id)
            return "Place the object sucessfully!"
        else: #invalid place to release the object, freeze the object underground again
            object_in_hand.translation = old_object_position
            object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC
            return "Cannot place object here!"


    def grab_release_action(self, agent_id=0):
        """atomic grab and release action"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        # print("hit object: ", hit_object_id)
        # print("hit point: ", hit_point)
        if hit_object_id == None: #can neither pick nor place
            return "Too far away!"
        object_id_in_hand = self.get_object_id_in_hand(agent_id)
        if object_id_in_hand == None: #no object in hand, try to pick an object
            if hit_object_id == -1: #the center viewpoint ray is hitting on the stage instead of an object
                return "Hitting on the stage!"
            if hit_object_id in self._agent_object_ids: #hitting on an agent
                return "Cannot grab an agent!"
            #the center viewpoint ray is hitting on an object
            object_in_hand = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
            if object_in_hand == None:
                return "Cannot grab articulated object!"
            if object_in_hand.motion_type != habitat_sim.physics.MotionType.DYNAMIC:  
                return "Cannot grab object with non-dynamic motion type!"
            # the object is now legal for grabbing
            object_position = object_in_hand.translation
            new_object_position = mn.Vector3(
                object_position.x,
                object_position.y-10000.0,
                object_position.z
            )
            object_in_hand.translation = new_object_position #change the object altitude to hide it underground
            object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC #set it to static temporarily to avoid continuous fall
            self.set_object_id_in_hand(hit_object_id, agent_id)
            return "Grab the target sucessfully!"
        else: #agent is grabbing an object, try to place it in the scene
            object_in_hand = self._rigid_obj_mgr.get_object_by_id(object_id_in_hand)
            object_in_hand.motion_type = habitat_sim.physics.MotionType.DYNAMIC #change the grabbed object to DYNAMIC
            can_place = False
            old_object_position = object_in_hand.translation
            for i in range(10): #start from the hit point, try different object altitude
                #print("try: ", i)
                new_object_position = mn.Vector3(
                    hit_point.x,
                    hit_point.y+i*0.05,
                    hit_point.z
                )
                object_in_hand.translation = new_object_position
                if not object_in_hand.contact_test(): #the object will not contact the collision world
                    can_place = True
                    break
            if can_place:
                object_in_hand.translation = new_object_position
                self.set_object_id_in_hand(None, agent_id)
                return "Place the target sucessfully!"
            else: #invalid place to release the object, freeze the object underground again
                object_in_hand.translation = old_object_position
                object_in_hand.motion_type = habitat_sim.physics.MotionType.STATIC
                return "Cannot place object here!"
    

    def open_close_action(self, agent_id=0):
        """open or close a receptacle"""
        hit_object_id, hit_point = self.get_nearest_object_under_crosshair(
            self.unproject(agent_id)) #get the first object (or the stage) hit by the viewpoint of the agent
        if hit_object_id == None: #too far
            return "Too far away!"
        if hit_object_id == -1:
            return "Cannot interact with the stage!"
        object = self._rigid_obj_mgr.get_object_by_id(hit_object_id)
        if object is not None: # it is a rigid object not an articulated object
            return "Cannot interact with a rigid object!"
        object = self._articulated_obj_mgr.get_object_by_id(hit_object_id)
        if object is None: #hitting on a non-base link of an articulated object
            articulated_object_id = self.link_id2art_id[hit_object_id]
            object = self._articulated_obj_mgr.get_object_by_id(articulated_object_id)
        object_state = self.articulated_object_state[object.object_id]
        joint_velocities = object.joint_velocities
        if object_state == 'closed':
            new_joint_velocities = [5.0 for _ in range(len(joint_velocities))]
            self.articulated_object_state[object.object_id] = 'open'
        else:
            new_joint_velocities = [-5.0 for _ in range(len(joint_velocities))]
            self.articulated_object_state[object.object_id] = 'closed'
        object.joint_velocities = new_joint_velocities
        print('successfully interacted with the articulated object.')


    def look_action(self, target_position, agent_id=0):
        """return an action list to look at someplace"""
        camera_ray = self.unproject(agent_id)
        #get the camera's center pixel position, ray direction, and target direction
        camera_position = np.array(camera_ray.origin)
        camera_direction = np.array(camera_ray.direction)
        target_direction = np.array(target_position)-camera_position
        target_direction = target_direction / np.linalg.norm(target_direction)
        action_list = []
        #initialize the inner product
        max_product = np.dot(target_direction, camera_direction)
        y_axis = [0.0, 1.0, 0.0]
        #greedy algorithm for maximizing the inner product of the camera ray and the target direction
        #first try to turn left and right
        while True:
            step = None
            current_camera_direction = None
            for action in ['turn_left', 'turn_right']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'turn_left':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=degree/180*np.pi)
                if action == 'turn_right':
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=y_axis, radian=-degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            if step == None:
                break
            camera_direction = current_camera_direction
            action_list.append(step)
        # then try look up and down    
        while True:
            step = None
            current_camera_direction = None
            for action in ['look_up', 'look_down']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
                product = np.dot(new_camera_direction, target_direction)
                if product > max_product:
                    max_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            camera_direction = current_camera_direction
            action_list.append(step)
            if step == None:
                break
        return action_list
    

    def reset_to_horizontal(self, agent_id):
        """return an action list to look horizontally"""
        camera_ray = self.unproject(agent_id)
        #get the camera's direction
        camera_direction = np.array(camera_ray.direction)
        y_axis = [0.0, 1.0, 0.0]
        min_product = abs(np.dot(y_axis, camera_direction))
        #greedy algorithm for minimizing the abs(product)
        action_list = []
        while True:
            step = None
            current_camera_direction = None
            for action in ['look_up', 'look_down']:
                degree = self.agents[agent_id].action_space[action]
                if action == 'look_up':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=-degree/180*np.pi)
                if action == 'look_down':
                    axis = np.cross(y_axis, camera_direction)
                    new_camera_direction = rotate_vector_along_axis(vector=camera_direction, axis=axis, radian=degree/180*np.pi)
                product = abs(np.dot(new_camera_direction, y_axis))
                if product < min_product:
                    min_product = product
                    current_camera_direction = new_camera_direction
                    step = action
            camera_direction = current_camera_direction
            action_list.append(step)
            if step == None:
                break
        return action_list

    
    def perform_discrete_collision_detection(self):
        """perform discrete collision detection for the scene"""
        self._simulator.perform_discrete_collision_detection()
    

    def get_physics_contact_points(self):
        """return a list of ContactPointData ” “objects describing the contacts from the most recent physics substep"""
        return self._simulator.get_physics_contact_points()


    def is_agent_colliding(self, agent_id, action):
        """ check wether the action will cause collision. Used to avoid border conditions during simulation. """
        if action not in ["move_forward", "move_backward"]: #only move action will cause collision
            return False
        step_size = self.agents[agent_id].step_size
        agent_transform = self._simulator.agents[agent_id].body.object.transformation
        if action == "move_forward":
            position = - agent_transform.backward * step_size
        else:
            position = agent_transform.backward * step_size

        new_position = agent_transform.translation + position
        filtered_position = self.get_path_finder().try_step(
            agent_transform.translation,
            new_position)
        dist_moved_before_filter = (new_position - agent_transform.translation).dot()
        dist_moved_after_filter = (filtered_position - agent_transform.translation).dot()
        # we check to see if the the amount moved after the application of the filter
        # is _less_ than the amount moved before the application of the filter
        EPS = 1e-4
        collided = (dist_moved_after_filter + EPS) < dist_moved_before_filter
        return collided


    def get_nearest_object_under_crosshair(self, ray): 
        """ get the nearest object hit by the crosshair ray within the grab_release distance """
        ray_cast_results = self._simulator.cast_ray(ray, self.grab_and_release_distance)
        object_id = None
        hit_point = None
        if ray_cast_results.has_hits(): #the ray hit some objects
            first_hit = ray_cast_results.hits[0]
            object_id = first_hit.object_id
            hit_point = first_hit.point
        #object_id: None if no hit, -1 if hit on the stage, non-negative value if hit on an object
        return object_id, hit_point


    def step(self, actions: Union[str, dict, None]):
        """all agents perform actions in the environment and return observations."""
        self.update_articulated_id_mapping()
        self.awake_all_objects()
        if actions == None:
            actions = {self._default_agent_id: "no_op"}
        assert type(actions) in [str, dict]
        if type(actions) is str: #a single action for the default agent
            actions = {self._default_agent_id: actions}
        for agent_id in actions:
            action = actions[agent_id]
            # print(action)
            # print(actions[agent_id])
            assert action in self.agents[agent_id].action_space
            if action == "grab_release":
                self.grab_release_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "pick":
                self.pick_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "place":
                self.place_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "open_close":
                self.open_close_action(agent_id)
                actions[agent_id] = "no_op"
            if action == "info":
                self.info_action(agent_id)
                actions[agent_id] = "no_op"
            agent_position = self.get_agent_state().position
        observations = self._simulator.step(action=actions, dt=1/self._fps)
        return observations


    def load_object(self, object_config_path,
                    translation=[0.0, 0.0, 0.0],
                    rotation=[0.0, 0.0, 0.0, 1.0],
                    transformation=None,
                    semantic=None,
                    motion="DYNAMIC") -> Dict:
        """load a rigid object"""
        object_template_id = self._obj_template_mgr.load_configs(object_config_path)[0]
        obj = self._rigid_obj_mgr.add_object_by_template_id(object_template_id)
        if semantic != None:
            self.rigid_id2semantic[obj.object_id] = semantic
        rotation = quat_to_magnum(quat_from_coeffs(rotation))
        obj.translation = translation
        obj.rotation = rotation
        if transformation is not None:
            obj.transformation = transformation
        assert motion in ["DYNAMIC", "STATIC"]
        if motion == "DYNAMIC":
            obj.motion_type = habitat_sim.physics.MotionType.DYNAMIC
        else:
            obj.motion_type = habitat_sim.physics.MotionType.STATIC
        return obj


    def attach_object_to_agent(self, object_path, agent_id=0) -> Dict:
        """attach an rigid object to an agent"""
        if object_path is None:
            return
        object_template_id = self._obj_template_mgr.load_configs(object_path)[0]
        obj = self._rigid_obj_mgr.add_object_by_template_id(object_template_id, self._simulator.agents[agent_id].scene_node)
        obj.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        self._agent_object_ids[agent_id] = obj.object_id


    def initialize_agent(self, agent_id=0, position=[0.0, 0.0, 0.0], rotation=[0.0, 0.0, 0.0, 1.0]):
        """initialize an agent by its position and rotation"""
        agent_state = habitat_sim.AgentState()
        agent_state.position = position
        agent_state.rotation = rotation
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()

      
    def randomly_initialize_agent(self, agent_id=0):
        """randomly initialize an agent"""
        point = self.get_path_finder().get_random_navigable_point(max_tries=10, island_index=0)
        agent_state = habitat_sim.AgentState()
        agent_state.position = point
        agent_state.rotation = np.quaternion(1.0, 0.0, 0.0, 0.0)
        agent = self._simulator.initialize_agent(agent_id, agent_state)
        return agent.scene_node.transformation_matrix()


    def reconfigure(self, sim_settings, agents_settings):
        """reconfigure"""
        self._resolution = [sim_settings['height'], sim_settings['width']]
        self._fps = sim_settings['fps']
        self.agents = []

        agent_configs = []
        for i, single_agent_settings in enumerate(agents_settings):
            agent_configs.append(make_agent_cfg(self._resolution, single_agent_settings))
            self.agents.append(Agent(self, single_agent_settings, i))

        self.num_of_agents = len(self.agents)
        self._agent_object_ids = [None for _ in range(self.num_of_agents)] #by default agents have no rigid object bodies
        self._config = make_sim_cfg(sim_settings, agent_configs)
        self._simulator.reconfigure(self._config)

        self.grab_and_release_distance = sim_settings['grab_and_release_distance']
        self._obj_template_mgr = self._simulator.get_object_template_manager()
        self._rigid_obj_mgr = self._simulator.get_rigid_object_manager()
        self._default_agent_id = sim_settings["default_agent"]

        for agent_id in range(self.num_of_agents):
            agent_object_path = agents_settings[agent_id]['agent_object']
            self.attach_object_to_agent(agent_object_path, agent_id)


    def geodesic_distance(
        self,
        position_a: Union[Sequence[float], np.ndarray],
        position_b: Union[
            Sequence[float], Sequence[Sequence[float]], np.ndarray
        ],
        episode=None) -> float:
        """shortest distance from a to b"""
        if episode is None or episode._shortest_path_cache is None:
            path = habitat_sim.MultiGoalShortestPath()
            if isinstance(position_b[0], (Sequence, np.ndarray)): #multiple endpoints
                path.requested_ends = np.array(position_b, dtype=np.float32)
            else: #single endpoints
                path.requested_ends = np.array(
                    [np.array(position_b, dtype=np.float32)]
                )
        else:
            path = episode._shortest_path_cache
        path.requested_start = np.array(position_a, dtype=np.float32)
        self.get_path_finder().find_path(path) #Finds the shortest path between a start point and the closest of a set of end points (in geodesic distance) on the navigation mesh using MultiGoalShortestPath module. Path variable is filled if successful. Returns boolean success.
        if episode is not None:
            episode._shortest_path_cache = path
        return path.geodesic_distance
    

    def get_articulated_object_nav_point(self, object_id):
        articulated_object = self._articulated_obj_mgr.get_object_by_id(object_id)
        handle = articulated_object.handle
        raw_position = articulated_object.translation
        orientation = [0, 0, -1]
        rotate_angle = self.articulated_object_rotate_angle_z[handle]
        orientation[2] = np.math.cos(rotate_angle)
        orientation[0] = np.math.sin(rotate_angle)
        delta = [0.01, 0.01, 0.01]
        i = 0
        ori_delta = [delta[i] * orientation[i] for i in range(len(delta))]
        cur_position = raw_position
        while i < 100:
            cur_position += ori_delta
            if self.get_path_finder().is_navigable(np.array(cur_position, dtype=np.float32), max_y_delta=2.0):
                return cur_position
            i += 1
        return None
    

    def load_articulated_object_placeable_points(self, file='data/articulated_object/all_objects.json'):
        with open(file, 'r') as f:
            articulated_object_dict = json.load(f)
        articulated_object_handles = self._articulated_obj_mgr.get_object_handles()
        for handle in articulated_object_handles:
            if handle in articulated_object_dict:
                placeable_point = np.transpose(np.array(articulated_object_dict[handle]['position_base_self']))
                articulated_object = self._articulated_obj_mgr.get_object_by_handle(handle)
                matrix = articulated_object.transformation
                transformation = np.transpose(np.array([[matrix[0].x, matrix[0].y, matrix[0].z, matrix[0].w],
                                            [matrix[1].x, matrix[1].y, matrix[1].z, matrix[1].w],
                                            [matrix[2].x, matrix[2].y, matrix[2].z, matrix[2].w],
                                            [matrix[3].x, matrix[3].y, matrix[3].z, matrix[3].w],]))
                hit_points = np.matmul(transformation, placeable_point)
                self.articulated_object_placeable_points.update({handle: hit_points})


    def get_articulated_object_bbox(self, object):
        # v_bbs = []
        # for visual_scene_node in object.visual_scene_nodes:
        #     local_bb = visual_scene_node.compute_cumulative_bb().scaled(mn.Vector3(object.global_scale))
        #     #print(dir(visual_scene_node))
        #     translation = visual_scene_node.absolute_translation
        #     bb = local_bb.translated(translation) # bounding box of the visual scene node in the scene
            
        #     v_bbs.append(np.array(bb.min))
        #     v_bbs.append(np.array(bb.max))
        # bb_min = np.min(v_bbs, axis=0) # the min_range of the bbox
        # bb_max = np.max(v_bbs, axis=0) # the max_range of the bbox
        # center_pos = np.mean([bb_min, bb_max], axis=0)
        # return bb_min, bb_max, center_pos

        v_bbs = []
        for visual_scene_node in object.visual_scene_nodes:
            local_bb = visual_scene_node.compute_cumulative_bb()
            #print(dir(visual_scene_node))
            x0, y0, z0 = np.array(local_bb.min)
            x1, y1, z1 = np.array(local_bb.max)
            bbox_range_in_local = np.array([
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1]])
            #print(dir(visual_scene_node))
            rot = np.array(visual_scene_node.absolute_transformation_matrix())[:3, :3]
            #print(rot)
            bbox_range_in_world = (rot @ bbox_range_in_local.T).T + np.array(visual_scene_node.absolute_translation)
            bb_min = np.min(bbox_range_in_world, axis=0)
            bb_max = np.max(bbox_range_in_world, axis=0)
            v_bbs.append(bb_min)
            v_bbs.append(bb_max)
        bb_min = np.min(v_bbs, axis=0) # the min_range of the bbox
        bb_max = np.max(v_bbs, axis=0) # the max_range of the bbox
        center_pos = np.mean([bb_min, bb_max], axis=0)
        return bb_min, bb_max, center_pos
    
    def get_rigid_object_bbox(self, object):
        aabb = object.collision_shape_aabb
        x0, y0, z0 = np.array(aabb.min)
        x1, y1, z1 = np.array(aabb.max)
        bbox_range_in_local = np.array([
            [x0, y0, z0],
            [x0, y0, z1],
            [x0, y1, z0],
            [x0, y1, z1],
            [x1, y0, z0],
            [x1, y0, z1],
            [x1, y1, z0],
            [x1, y1, z1]])
        rot = np.array(object.rotation.to_matrix())
        bbox_range_in_world = (rot @ bbox_range_in_local.T).T + np.array(object.translation)
        bb_min = np.min(bbox_range_in_world, axis=0)
        bb_max = np.max(bbox_range_in_world, axis=0)
        center_pos = np.mean([bb_min, bb_max], axis=0)
        return bb_min, bb_max, center_pos
    

    def get_articulated_object_rotate_angle(self):
        rotation1 = mn.Quaternion.rotation(mn.Deg(-90.0), mn.Vector3.x_axis())
        rotation2 = mn.Quaternion.rotation(mn.Deg(90.0), mn.Vector3.y_axis())
        fixed_rotation = rotation2*rotation1
        for handle in self._articulated_obj_mgr.get_object_handles():
            object_ = self._articulated_obj_mgr.get_object_by_handle(handle)
            rotation = object_.rotation
            raw_rotation = rotation * fixed_rotation.inverted()
            w = raw_rotation.scalar
            x, y, z = raw_rotation.vector.x, raw_rotation.vector.y, raw_rotation.vector.z
            self.articulated_object_rotate_angle_z.update({handle: quaternion_to_z_rotation(w, x, y, z)})

