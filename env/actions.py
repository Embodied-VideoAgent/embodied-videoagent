import habitat_sim
import numpy as np


@habitat_sim.registry.register_move_fn(name="no_op", body_action=True)
class NoOperation(habitat_sim.SceneNodeControl):
    def __call__(self, scene_node, actuation_spec):
        pass 


@habitat_sim.registry.register_move_fn(name="move_backward", body_action=True)
class MoveBackward(habitat_sim.SceneNodeControl):
    def __call__(
        self, scene_node, actuation_spec):
        backward_ax = (
            np.array(scene_node.absolute_transformation().rotation_scaling())
            @ habitat_sim.geo.BACK
        )
        scene_node.translate_local(backward_ax * actuation_spec.amount)


    
