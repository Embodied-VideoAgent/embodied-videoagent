import numpy as np
import habitat_sim
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_similarity
import math


def make_sim_cfg(settings, agent_settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    #help(sim_cfg)
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    if settings["scene_dataset_config_file"] is not None:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.enable_physics = True
    if habitat_sim.__version__ != '0.3.0':
        sim_cfg.pbr_image_based_lighting = True
    sim_cfg.allow_sliding = settings["allow_sliding"]
    sim_cfg.navmesh_settings = habitat_sim.nav.NavMeshSettings()
    sim_cfg.navmesh_settings.set_defaults()
    sim_cfg.navmesh_settings.agent_radius = agent_settings[0].radius
    sim_cfg.navmesh_settings.agent_height = agent_settings[0].height
    sim_cfg.navmesh_settings.include_static_objects = settings[
        "navmesh_include_static_objects"]
    return habitat_sim.Configuration(sim_cfg, agent_settings)


def make_agent_cfg(resolution, settings):
    sensor_type = {
        'rgb' : habitat_sim.SensorType.COLOR,
        'depth' : habitat_sim.SensorType.DEPTH,
        'semantic': habitat_sim.SensorType.SEMANTIC
    }   
    sensor_specs = []
    def create_1st_camera_spec(**kwargs):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.resolution = resolution
        camera_sensor_spec.position = [0.0, settings["agent_height"], -settings["sensor_front_bias"]]
        for k in kwargs:
            setattr(camera_sensor_spec, k, kwargs[k])
        return camera_sensor_spec
    
    def create_3rd_camera_spec(**kwargs):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.resolution = resolution
        camera_sensor_spec.position = [0.0, 1.65*settings["agent_height"], -settings["sensor_front_bias"]+0.5*settings["agent_height"]]
        camera_sensor_spec.orientation = [-0.25*np.pi, 0.0, 0.0]
        for k in kwargs:
            setattr(camera_sensor_spec, k, kwargs[k])
        return camera_sensor_spec
    
    def create_articulated_agent_camera_spec(**kwargs):
        camera_sensor_spec = habitat_sim.CameraSensorSpec()
        camera_sensor_spec.resolution = resolution
        camera_sensor_spec.position = [0.0, 0.0, 0.0]
        camera_sensor_spec.orientation = [0.0, 0.0, 0.0]
        for k in kwargs:
            setattr(camera_sensor_spec, k, kwargs[k])
        return camera_sensor_spec
    
    if settings['articulated'] == True:
        for sensor in settings["sensors"]:
            sensor_specs.append(create_articulated_agent_camera_spec(
                uuid=sensor,
                sensor_type=sensor_type['rgb'],
                sensor_subtype=habitat_sim.SensorSubType.PINHOLE))
    else:   
        for sensor in settings["sensors"]:
            type, view = sensor.split('_')
            assert type in ['rgb', 'depth', 'semantic']
            assert view in ['1st', '3rd']

            if view == '1st':
                sensor_specs.append(create_1st_camera_spec(
                    uuid=type+'_'+view,
                    hfov=settings["hfov"],
                    sensor_type=sensor_type[type],
                    sensor_subtype=habitat_sim.SensorSubType.PINHOLE))
            else:
                sensor_specs.append(create_3rd_camera_spec(
                    uuid=type+'_'+view,
                    hfov=settings["hfov"],
                    sensor_type=sensor_type[type],
                    sensor_subtype=habitat_sim.SensorSubType.PINHOLE))
    
    
    action_space = {}
    if "no_op" in settings["action_space"]:
        action_space["no_op"] = habitat_sim.agent.ActionSpec("no_op")
    
    if "move_forward" in settings["action_space"]:
        action_space["move_forward"] = habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=settings["step_size"])
        )

    if "move_backward" in settings["action_space"]:
        action_space["move_backward"] = habitat_sim.agent.ActionSpec(
            "move_backward", habitat_sim.agent.ActuationSpec(amount=settings["step_size"])
        )

    if "turn_left" in settings["action_space"]:
        action_space["turn_left"] = habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=settings["turn_angle"])
        )

    if "turn_right" in settings["action_space"]:
        action_space["turn_right"] = habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=settings["turn_angle"])
        )
    
    if "look_up" in settings["action_space"]:
        action_space["look_up"] = habitat_sim.agent.ActionSpec(
            "look_up", habitat_sim.agent.ActuationSpec(amount=settings["tilt_angle"])
        )

    if "look_down" in settings["action_space"]:
        action_space["look_down"] = habitat_sim.agent.ActionSpec(
            "look_down", habitat_sim.agent.ActuationSpec(amount=settings["tilt_angle"])
        )

    # create agent specifications
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.radius = settings["agent_radius"]
    agent_cfg.height = settings["agent_height"]
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = action_space

    return agent_cfg


def rotate_vector_along_axis(vector, axis, radian):
    v = np.asarray(vector)
    axis = np.asarray(axis)
    axis = axis / np.linalg.norm(axis)
    r = R.from_rotvec(radian * axis)
    m = r.as_matrix()
    a = np.matmul(m, v.T)
    return a.T


def compute_cosine_similarity(target_embedding, embedding_list):
    target_embedding_tensor = target_embedding.reshape(1, -1)
    # Compute cosine similarity
    similarity_scores = cosine_similarity(target_embedding_tensor, embedding_list)
    return similarity_scores.reshape(-1)


def top_k_indices(scores, k):
    max_len = scores.shape[0]
    k = min(max_len, k)
    indices = np.argsort(scores)[-k:][::-1]
    return list(indices)



def quaternion_to_z_rotation(w, x, y, z):
    # Calculate yaw (rotation around Z-axis) from quaternion
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y**2 + z**2))
    # Convert radians to degrees if needed
    return yaw




