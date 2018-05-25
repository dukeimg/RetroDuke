from config import games
import numpy as np


class Game:
    def __init__(self, name):
        self.name = name

        self.config = games.get(name)
        self.frame_shape = self.config.get('frame_shape')
        self.action_space = self.config.get('action_space')
        self.render_format = self.config.get('render_format')
        self.process_image_format = self.config.get('process_image_format')
        self.network_configs = self.config.get('training')

        self.action_size = self.action_space.__len__()
        self.possible_actions = np.identity(self.action_size, dtype=int).tolist()

    # high — possible_actions instance
    # low — action_space instance
    def convert_action(self, high=None, low=None):
        if high:
            return self.action_space[self.possible_actions.index(high)]
        elif low:
            return self.possible_actions[self.action_space.index(low)]
