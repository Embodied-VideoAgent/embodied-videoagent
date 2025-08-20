import env.actions


class Agent:
    def __init__(self, sim, agent_settings, agent_id) -> None:
        self._sim = sim
        self.agent_id = agent_id
        self.articulated = agent_settings['articulated']
        self.agent_height = agent_settings['agent_height']
        self.agent_radius = agent_settings['agent_radius']

        self.action_space = dict()
        for action in agent_settings['action_space']:
            self.action_space[action] = None
            if action == 'move_forward' or 'move_backward':
                self.action_space[action] = agent_settings['step_size']
            if action == 'turn_left' or 'turn_right':
                self.action_space[action] = agent_settings['turn_angle']
            if action == 'look_up' or 'look_down':
                self.action_space[action] = agent_settings['tilt_angle']
