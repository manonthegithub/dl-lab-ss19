import numpy as np

LEFT = 1
RIGHT = 2
STRAIGHT = 0
ACCELERATE =3
BRAKE = 4

LA = np.array([-1.0, 0.0, 0.0])
RA = np.array([1.0, 0.0, 0.0])
ACC = np.array([0.0, 1.0, 0.0])
BRA = np.array([0, 0, 0.2], dtype='float32')
STR = np.array([0.0, 0.0, 0.0])


def rgb2gray(rgb):
    """ 
    this method converts rgb images to grayscale.
    """
    gray = np.dot(rgb[...,:3], [0.2125, 0.7154, 0.0721])
    return gray.astype('float32') 


def action_to_id(a):
    """ 
    this method discretizes the actions.
    Important: this method only works if you recorded data pressing only one key at a time!
    """
    if all(a == LA): return LEFT               # LEFT: 1
    elif all(a == RA): return RIGHT             # RIGHT: 2
    elif all(a == ACC): return ACCELERATE        # ACCELERATE: 3
    elif all(a == BRA): return BRAKE             # BRAKE: 4
    elif all(a == STR): return STRAIGHT
    elif all(a * RA == RA): return RIGHT
    elif all(a * RA == LA): return LEFT

def id_to_action(action_id, max_speed=1.0):
    """ 
    this method makes actions continous.
    Important: this method only works if you recorded data pressing only one key at a time!
    """

    if action_id == LEFT:
        return np.array([-1.0, 0.0, 0.00])
    elif action_id == RIGHT:
        return np.array([1.0, 0.0, 0.00])
    elif action_id == ACCELERATE:
        return np.array([0.0, max_speed, 0.0])
    elif action_id == BRAKE:
        return np.array([0.0, 0.0, 0.2])
    elif action_id == STRAIGHT:
        return np.array([0.0, 0.0, 0.0])
    

class EpisodeStats:
    """
    This class tracks statistics like episode reward or action usage.
    """
    def __init__(self):
        self.episode_reward = 0
        self.actions_ids = []

    def step(self, reward, action_id):
        self.episode_reward += reward
        self.actions_ids.append(action_id)

    def get_action_usage(self, action_id):
        ids = np.array(self.actions_ids)
        return (len(ids[ids == action_id]) / len(ids))
