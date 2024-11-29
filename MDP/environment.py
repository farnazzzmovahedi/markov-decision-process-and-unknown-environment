"""
Project Title: MDP

Department:
    Artificial Intelligence Department
    Faculty of Computer Engineering
    University of Isfahan
    November 1, 2024

Supervisor:
    Dr. Hossein Karshenas(h.karshenas@eng.ui.ac.ir) - Professor
    Pouria Sameti(pouria.sameti2002@mehr.ui.ac.ir) - Teaching Assistant


Project Overview:
    This project provides a flexible framework within the MDP (Markov Decision Process) setting.
    Students are encouraged to implement various algorithms, such as value iteration, policy
    iteration, and Q-learning, to explore different approaches to decision-making, rewards,
    penalties, and policy evaluation in a stochastic environment.

Objectives:
    - To provide students with practical experience in implementing reinforcement learning algorithms in a stochastic environment.
    - To explore modifications in the MDP framework that introduce intermediate states, rewarding or penalizing agents based on their choices.
    - To challenge students to strategize their approach in reaching the final state while meeting a cumulative score requirement.

Licensing Information:
    -You are free to use or extend these projects for educational purposes.
"""
from collections import defaultdict

import numpy as np
import pygame
import random
import copy

#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
COLORS = {
    'T': (135, 206, 235),  # Tile ground
    'P': (135, 206, 235),  # Pigs
    'Q': (135, 206, 235),  # Queen
    'G': (135, 206, 235),  # Goal
    'R': (135, 206, 235),  # Rock
}

GOOD_PIG_REWARD = 250
GOAL_REWARD = 400
QUEEN_REWARD = -400
DEFAULT_REWARD = (-1)

PIGS = 8
QUEENS = 2
ROCKS = 8


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################


class PygameInit:

    @classmethod
    def initialization(cls):
        grid_size = 64
        tile_size = 100

        pygame.init()
        screen = pygame.display.set_mode((grid_size * tile_size, grid_size * tile_size))
        pygame.display.set_caption("MDP Angry Birds")
        clock = pygame.time.Clock()

        return screen, clock


#######################################################
#                DONT CHANGE THIS PART                #
#######################################################
class AngryBirds:
    def __init__(self):
        self.__grid_size = 8
        self.__tile_size = 100
        self.__num_pigs = PIGS
        self.__num_queens = QUEENS
        self.__num_rocks = ROCKS
        self.__probability_dict = self.__generate_probability_dict()
        self.__base_grid = self.__generate_grid()
        self.__agent_pos = (0, 0)

        self.grid = copy.deepcopy(self.__base_grid)
        self.reward = 0
        self.done = False
        self.reward_map = self.reward_function()
        self.transition_table = self.__calculate_transition_model(self.__grid_size, self.__probability_dict,
                                                                  self.reward_map)

        self.__agent_image = pygame.image.load("Env/icons/angry-birds.png")
        self.__agent_image = pygame.transform.scale(self.__agent_image, (self.__tile_size, self.__tile_size))

        self.__pig_image = pygame.image.load('Env/icons/pigs.png')
        self.__pig_image = pygame.transform.scale(self.__pig_image, (self.__tile_size, self.__tile_size))
        self.__pig_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__pig_with_background.fill((135, 206, 235))
        self.__pig_with_background.blit(self.__pig_image, (0, 0))

        self.__egg_image = pygame.image.load('Env/icons/eggs.png')
        self.__egg_image = pygame.transform.scale(self.__egg_image, (self.__tile_size, self.__tile_size))
        self.__egg_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__egg_with_background.fill((135, 206, 235))
        self.__egg_with_background.blit(self.__egg_image, (0, 0))

        self.__queen_image = pygame.image.load('Env/icons/queen.png')
        self.__queen_image = pygame.transform.scale(self.__queen_image, (self.__tile_size, self.__tile_size))
        self.__queen_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__queen_with_background.fill((135, 206, 235))
        self.__queen_with_background.blit(self.__queen_image, (0, 0))

        self.__rock_image = pygame.image.load('Env/icons/rocks.png')
        self.__rock_image = pygame.transform.scale(self.__rock_image, (self.__tile_size, self.__tile_size))
        self.__rock_with_background = pygame.Surface((self.__tile_size, self.__tile_size))
        self.__rock_with_background.fill((135, 206, 235))
        self.__rock_with_background.blit(self.__rock_image, (0, 0))

    def __generate_grid(self):

        while True:
            filled_spaces = [(0, 0), (self.__grid_size - 1, self.__grid_size - 1)]
            grid = [['T' for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

            num_pigs = self.__num_pigs
            for _ in range(num_pigs):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'P'
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_queens):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'Q'
                        filled_spaces.append((r, c))
                        break

            for _ in range(self.__num_rocks):
                while True:
                    r, c = random.randint(0, self.__grid_size - 1), random.randint(0, self.__grid_size - 1)
                    if (r, c) not in filled_spaces:
                        grid[r][c] = 'R'
                        filled_spaces.append((r, c))
                        break

            grid[self.__grid_size - 1][self.__grid_size - 1] = 'G'
            if AngryBirds.__is_path_exists(grid=grid, start=(0, 0), goal=(7, 7)):
                break

        return grid

    def reset(self):
        self.grid = copy.deepcopy(self.__base_grid)
        self.__agent_pos = (0, 0)
        self.reward = 0
        self.done = False
        return self.__agent_pos

    def step(self, action):
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        neighbors = {
            0: [2, 3],
            1: [2, 3],
            2: [0, 1],
            3: [0, 1]
        }

        intended_probability = self.__probability_dict[self.__agent_pos][action]['intended']
        neighbors_probability = self.__probability_dict[self.__agent_pos][action]['neighbor']

        prob_dist = [0, 0, 0, 0]
        prob_dist[action] = intended_probability
        prob_dist[neighbors[action][0]] = neighbors_probability
        prob_dist[neighbors[action][1]] = neighbors_probability

        chosen_action = np.random.choice([0, 1, 2, 3], p=prob_dist)

        dx, dy = actions[chosen_action]
        new_row = self.__agent_pos[0] + dx
        new_col = self.__agent_pos[1] + dy

        if (0 <= new_row < self.__grid_size and 0 <= new_col < self.__grid_size and
                self.grid[new_row][new_col] != 'R'):
            self.__agent_pos = (new_row, new_col)

        current_tile = self.grid[self.__agent_pos[0]][self.__agent_pos[1]]
        reward = DEFAULT_REWARD

        if current_tile == 'Q':
            reward = QUEEN_REWARD
            self.grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'P':
            reward = GOOD_PIG_REWARD
            self.grid[self.__agent_pos[0]][self.__agent_pos[1]] = 'T'

        elif current_tile == 'G':
            reward = GOAL_REWARD
            self.done = True

        elif current_tile == 'T':
            reward = DEFAULT_REWARD

        probability = prob_dist[chosen_action]
        self.reward = reward
        next_state = self.__agent_pos
        is_terminated = self.done
        return next_state, probability, self.reward, is_terminated

    def render(self, screen):
        for r in range(self.__grid_size):
            for c in range(self.__grid_size):
                color = COLORS[self.grid[r][c]]
                pygame.draw.rect(screen, color, (c * self.__tile_size, r * self.__tile_size, self.__tile_size,
                                                 self.__tile_size))

                if self.grid[r][c] == 'P':
                    screen.blit(self.__pig_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'G':
                    screen.blit(self.__egg_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'Q':
                    screen.blit(self.__queen_with_background, (c * self.__tile_size, r * self.__tile_size))

                if self.grid[r][c] == 'R':
                    screen.blit(self.__rock_with_background, (c * self.__tile_size, r * self.__tile_size))

        for r in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (0, r * self.__tile_size), (self.__grid_size * self.__tile_size,
                                                                            r * self.__tile_size), 2)
        for c in range(self.__grid_size + 1):
            pygame.draw.line(screen, (0, 0, 0), (c * self.__tile_size, 0), (c * self.__tile_size,
                                                                            self.__grid_size * self.__tile_size), 2)

        agent_row, agent_col = self.__agent_pos
        screen.blit(self.__agent_image, (agent_col * self.__tile_size, agent_row * self.__tile_size))

    # def reward_function(self):
    #     # implement this function
    #
    #     """it returns a 8x8 matrix
    #     [[-1, -1, -1, -1, -1, -1, -1, -1],
    #     [-1, -1, -1, -1, -1, -1, -1, -1],
    #     [-1, -400, -1, -1, -1, 180, -1, -1],
    #     ...'"""
    #
    #     reward_map = [[0 for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]
    #     return reward_map
    def reward_function(self):
        """
        Generates a reward map based on the environment's grid, prioritizing pigs closer to the goal.
        """
        # Initialize a reward map with default rewards
        reward_map = [[DEFAULT_REWARD for _ in range(self.__grid_size)] for _ in range(self.__grid_size)]

        # Coordinates of the goal
        goal_pos = (self.__grid_size - 1, self.__grid_size - 1)

        # Iterate over each cell in the grid
        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                cell = self.grid[row][col]

                if cell == 'G':  # Goal
                    reward_map[row][col] = 1000
                elif cell == 'P':
                    distance_to_goal = abs(row - goal_pos[0]) + abs(col - goal_pos[1])

                    # Adjust reward scaling parameters
                    base_reward = 150  # Minimum reward for far pigs
                    alpha = 650  # Determines the range (800 - 150 = 650)
                    beta = self.__grid_size / 2  # Decay factor (can be tuned)

                    # Compute scaled reward
                    scaled_reward = base_reward + alpha * np.exp(-distance_to_goal / beta)
                    print("P", scaled_reward)

                    # # Ensure reward does not exceed goal reward
                    # if scaled_reward >= GOAL_REWARD:
                    #     scaled_reward = GOAL_REWARD - 1

                    reward_map[row][col] = scaled_reward
                elif cell == 'Q':  # Queen pig
                    reward_map[row][col] = -1000
                elif cell == 'T':
                    distance_to_goal = abs(row - goal_pos[0]) + abs(col - goal_pos[1])

                    # Adjusted parameters
                    base_penalty = -10  # Default penalty for far states
                    alpha = 10  # Spread multiplier for reducing the penalty
                    beta = 50  # Maximum impact of proximity
                    n = 3  # Exponent for sharper drop-off with distance

                    # Compute scaled penalty based on distance
                    scaled_reward = base_penalty + alpha * (beta / (1 + distance_to_goal ** n))

                    # Clamp rewards to stay in the range [-20, -2]
                    scaled_reward = max(min(scaled_reward, -2), base_penalty)

                    # Assign reward
                    reward_map[row][col] = scaled_reward

                    # Debugging output
                    print(f"T Cell at ({row}, {col}): Distance = {distance_to_goal}, Reward = {scaled_reward}")

                elif cell == 'R':  # Normal tile
                    reward_map[row][col] = -10

        return reward_map

    @classmethod
    def __calculate_transition_model(cls, grid_size, actions_prob, reward_map):
        actions = {
            0: (-1, 0),  # Up
            1: (1, 0),  # Down
            2: (0, -1),  # Left
            3: (0, 1)  # Right
        }

        neighbors = {
            0: [2, 3],  # Up -> Left and Right
            1: [2, 3],  # Down -> Left and Right
            2: [0, 1],  # Left -> Up and Down
            3: [0, 1]  # Right -> Up and Down
        }

        transition_table = {}

        for row in range(grid_size):
            for col in range(grid_size):
                state = (row, col)
                transition_table[state] = {}

                for action in range(4):
                    transition_table[state][action] = []

                    intended_move = actions[action]
                    next_state = (row + intended_move[0], col + intended_move[1])

                    if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                        reward = reward_map[next_state[0]][next_state[1]]
                        intended_probability = actions_prob[(next_state[0], next_state[1])][action]['intended']
                        transition_table[state][action].append((intended_probability, next_state, reward))
                    else:
                        intended_probability = actions_prob[state][action]['intended']
                        transition_table[state][action].append(
                            (intended_probability, state, reward_map[row][col]))

                    for neighbor_action in neighbors[action]:
                        neighbor_move = actions[neighbor_action]
                        next_state = (row + neighbor_move[0], col + neighbor_move[1])

                        if 0 <= next_state[0] < grid_size and 0 <= next_state[1] < grid_size:
                            reward = reward_map[next_state[0]][next_state[1]]
                            neighbor_probability = actions_prob[(next_state[0], next_state[1])][action]['neighbor']
                            transition_table[state][action].append((neighbor_probability, next_state, reward))
                        else:
                            neighbor_probability = actions_prob[state][action]['neighbor']
                            transition_table[state][action].append(
                                (neighbor_probability, state, reward_map[row][col]))

        return transition_table

    @classmethod
    def __is_path_exists(cls, grid, start, goal):
        grid_size = len(grid)
        visited = set()

        def dfs(x, y):
            if (x, y) == goal:
                return True
            visited.add((x, y))

            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if (0 <= nx < grid_size and 0 <= ny < grid_size and
                        (nx, ny) not in visited and grid[nx][ny] != 'R'):
                    if dfs(nx, ny):
                        return True
            return False

        return dfs(start[0], start[1])

    def __generate_probability_dict(self):
        probability_dict = {}

        for row in range(self.__grid_size):
            for col in range(self.__grid_size):
                state = (row, col)
                probability_dict[state] = {}

                for action in range(4):
                    intended_prob = random.uniform(0.60, 0.80)
                    remaining_prob = 1 - intended_prob
                    neighbor_prob = remaining_prob / 2

                    probability_dict[state][action] = {
                        'intended': intended_prob,
                        'neighbor': neighbor_prob}
        return probability_dict


def value_iteration(grid, transition_table, discount_factor=0.9, theta=1e-4):
    """
    Solves the MDP using Value Iteration algorithm.
    Args:
        env: Environment object with env.grid and env.transition_table.
        discount_factor: Discount factor for future rewards (gamma).
        theta: Threshold for stopping criteria.
    Returns:
        A policy dict mapping states to actions.
    """

    # Initialize the value function
    V = defaultdict(float)
    policy = {}

    while True:
        delta = 0  # To measure convergence
        for x in range(8):
            for y in range(8):
                state = (x, y)
                if grid[x][y] == "R":  # Skip rocks
                    continue

                # Calculate the value of each action
                action_values = []
                for action in range(4):  # 0=Up, 1=Down, 2=Left, 3=Right
                    value = 0
                    for prob, next_state, reward in transition_table[state][action]:
                        value += prob * (reward + discount_factor * V[next_state])
                    action_values.append(value)

                # Update the value function and policy
                best_action_value = max(action_values) if action_values else 0
                delta = max(delta, abs(V[state] - best_action_value))
                V[state] = best_action_value
                if action_values:
                    policy[state] = np.argmax(action_values)

        # Check for convergence
        if delta < theta:
            break

    return policy
