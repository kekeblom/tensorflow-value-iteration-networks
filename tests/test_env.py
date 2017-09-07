import unittest
from unittest import TestCase
import env
from env import GridWorld
import numpy as np

class EnvTestCase(TestCase):
    def setUp(self):
        self.env = GridWorld()
        self.env.reset()

    def test_not_at_goal(self):
        self.assertNotEqual(self.env.character_position, self.env.goal)

    def test_possible_moves(self):
        self.env = GridWorld(map_height=3, map_width=3)
        self.env.map = np.zeros((3, 3))
        self.env.map[0, 0] = env.CHARACTER
        self.env.character_position = (0, 0)
        moves = self.env._possible_moves()
        self.assertIn(env.RIGHT, moves)
        self.assertIn(env.DOWN, moves)

    def test_can_take_actions(self):
        prev_obs = self.env.map.copy()
        for _ in range(10):
            move = np.random.choice(self.env._possible_moves())
            obs, reward, done, _ = self.env.step(move)
            self.assertFalse((obs == prev_obs).all())
            prev_obs = obs.copy()


