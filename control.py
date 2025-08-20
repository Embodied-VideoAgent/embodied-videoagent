import cv2
import numpy as np
import quaternion



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


def discrete_keyboard_control(sim):
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
        rgb = observations[0]["rgb_1st"]
        depth = observations[0]["depth_1st"]
        pos = sim._simulator.get_agent(0).state.sensor_states['depth_1st'].position
        rot = quaternion.as_rotation_matrix(sim._simulator.get_agent(0).state.sensor_states['depth_1st'].rotation)            
        rot = rot @ np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
        return rgb, depth, pos, rot




def discrete_keyboard_control_using_executor(executor):
    executor.execute_steps(action_list=["no_op"])
    while True:
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
        elif keystroke == ord(NO_OP_KEY):
            print("action: NO_OP")
            action = 'no_op'
        elif keystroke == ord(INFO_KEY):
            print("action: info")
            action = 'info'
        elif keystroke == ord(FINISH):
            print("action: FINISH")
            break
        else:
            print("INVALID KEY")
            continue
        executor.execute_steps(action_list=[action])
