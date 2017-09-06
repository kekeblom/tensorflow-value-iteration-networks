import numpy as np
import sys
from gym import Env, spaces, utils
from gym.envs import register

WALL_FRACTION = 0.3

UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

EMPTY = 0
WALL = 1
CHARACTER = 2
GOAL = 3

class GridWorld(Env):
    def __init__(self, map_height=8, map_width=8):
        super(GridWorld, self).__init__()
        self.map_height = map_height
        self.map_width = map_width
        self._reset()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=EMPTY, high=GOAL, shape=self.map.shape)
        self.goal_reward = self.map_height * self.map_width
        self.character_position = None
        self.steps = 0

    def _reset(self):
        self.map = np.zeros((self.map_height, self.map_width), np.float32)
        self.start = None
        self.goal = None
        self._set_start()
        self._set_goal()
        self._set_walls()
        self.character_position = self.start
        self.steps = 0
        return self.map

    def _step(self, action):
        self.steps += 1
        moves = self._possible_moves(self.character_position)
        reward = -0.1
        done = False
        if action in moves:
            new_position = self._position_after_move(self.character_position, action)
            self.map[self.character_position] = EMPTY
            if self.map[new_position[0]][new_position[1]] == GOAL:
                reward = self.goal_reward
                done = True

            self.map[new_position[0], new_position[1]] = CHARACTER
            self.character_position = new_position
        else:
            reward = -1.0

        if self.steps >= 200:
            reward = -50.0
            done = True

        return self.map.copy(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            return
        outfile = sys.stdout
        for i in range(self.map.shape[0]):
            for j in range(self.map.shape[1]):
                if self.map[i][j] == EMPTY:
                    outfile.write('.')
                elif self.map[i][j] == WALL:
                    outfile.write('#')
                elif self.map[i][j] == GOAL:
                    outfile.write(utils.colorize('G', "green", highlight=True))
                else:
                    outfile.write('C')
            outfile.write('\n')
        if mode != 'human':
            return outfile

    def _set_start(self):
        shape = self.map.shape
        row = np.random.randint(0, shape[0])
        column = np.random.randint(0, shape[1])
        self.map[row][column] = CHARACTER
        self.start = (row, column)

    def _set_goal(self):
        while True:
            row = np.random.randint(0, self.map.shape[0])
            column = np.random.randint(0, self.map.shape[1])
            if self.map[row][column] != CHARACTER:
                self.map[row][column] = GOAL
                self.goal = (row, column)
                break

    def _set_walls(self):
        wall_blocks = int(self.map.shape[0] * self.map.shape[1] * WALL_FRACTION)
        # make sure there exists a path to the goal
        path = self._find_path_to_goal()
        set_blocks = 0
        i = 0
        while set_blocks < wall_blocks:
            random_position = (np.random.randint(0, self.map.shape[0]), np.random.randint(0, self.map.shape[1]))
            if self._empty(random_position) and random_position not in path:
                self.map[random_position[0], random_position[1]] = WALL
                set_blocks += 1
            i += 1
            if i > (self.map.shape[0] * self.map.shape[1] * 2):
                break

    def _empty(self, position):
        return self.map[position[0], position[1]] == EMPTY

    def _find_path_to_goal(self):
        path = []
        current_position = self.start
        previous_move = None
        random_walk_length = np.random.randint(self.map_height + self.map_width, self.map_height * self.map_width)
        for _ in range(random_walk_length):
            path.append(current_position)
            possible_moves = self._possible_moves(current_position)
            move = np.random.choice(possible_moves)
            current_position = self._position_after_move(current_position, move)
        return path + self._find_path(current_position, self.goal)

    def _find_path(self, position, goal):
        path = []
        diff_y = position[0] - goal[0]
        diff_x = position[1] - goal[1]
        while diff_y != 0 and diff_x != 0:
            if diff_y > 0:
                position = (position[0] - 1, position[1])
                diff_y -= 1
                path.append(position)
            if diff_y < 0:
                position = (position[0] + 1, position[1])
                diff_y += 1
                path.append(position)
            if diff_x > 0:
                position = (position[0], position[1] - 1)
                diff_x -= 1
                path.append(position)
            if diff_x < 0:
                position = (position[0], position[1] + 1)
                diff_x += 1
                path.append(position)
        return path

    def _position_after_move(self, index, move):
        if move == UP:
            return (index[0] - 1, index[1])
        elif move == DOWN:
            return (index[0] + 1, index[1])
        elif move == LEFT:
            return (index[0], index[1] - 1)
        else:
            return (index[0], index[1] + 1)

    def _opposite(self, move):
        if move == UP:
            return DOWN
        elif move == DOWN:
            return UP
        elif move == LEFT:
            return RIGHT
        else:
            return LEFT

    def _possible_moves(self, position=None):
        if position is None:
            position = self.character_position
        moves = []
        if self._can_move(position, DOWN):
            moves.append(DOWN)
        if self._can_move(position, RIGHT):
            moves.append(RIGHT)
        if self._can_move(position, UP):
            moves.append(UP)
        if self._can_move(position, LEFT):
            moves.append(LEFT)
        return moves

    def _can_move(self, position, direction):
        try:
            if direction == UP:
                is_not_at_edge = position[0] > 0
                no_wall = self.map[position[0] - 1, position[1]] != WALL
                return is_not_at_edge and no_wall
            elif direction == DOWN:
                is_not_at_edge = position[0] < (self.map_height - 1)
                no_wall = self.map[position[0] + 1, position[1]] != WALL
                return is_not_at_edge and no_wall
            elif direction == LEFT:
                is_not_at_edge = position[1] > 0
                no_wall = self.map[position[0], position[1] - 1] != WALL
                return is_not_at_edge and no_wall
            elif direction == RIGHT:
                is_not_at_edge = position[1] < (self.map_width - 1)
                no_wall = self.map[position[0], position[1] + 1] != WALL
                return is_not_at_edge and no_wall
        except IndexError:
            return False

register("GridWorld-v0", entry_point="env:GridWorld")

if __name__ == '__main__':
    env = GridWorld()
    obs = env.reset()
    done = False
    for i in range(100):
        obs, reward, done, _ = env.step(env.action_space.sample())
        print(obs)



