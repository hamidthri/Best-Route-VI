import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, gridSize, items):
        self.step_reward = -1
        self.invalid_move_penalty = -1  # Penalty for invalid moves
        self.m = gridSize[0]
        self.n = gridSize[1]
        self.grid = np.zeros(gridSize)
        self.items = items

        self.state_space = list(range(self.m * self.n))

        self.action_space = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.actions = ['U', 'D', 'L', 'R']

        self.P = self.int_P()

    # def int_P(self):
    #     P = {}
    #     for state in self.state_space:
    #         for action in self.actions:
    #             reward = self.step_reward
    #             n_state = state + self.action_space[action]

    #             if not self.is_valid_state(n_state):
    #                 reward = self.invalid_move_penalty
    #                 n_state = state  # Stay in the current state if move is invalid
    #             elif n_state in self.items.get('fire').get('loc'):
    #                 reward += self.items.get('fire').get('reward')
    #             elif n_state in self.items.get('water').get('loc'):
    #                 reward += self.items.get('water').get('reward')

    #             P[(state ,action)] = (n_state, reward)
    #     return P


    def is_valid_state(self, state):
        """Check if the state is within the grid boundaries."""
        if state < 0 or state >= self.m * self.n:
            return False
        row, col = divmod(state, self.m)
        return 0 <= row < self.m and 0 <= col < self.n

    def check_terminal(self, state):
        return state in self.items.get('fire').get('loc') + self.items.get('water').get('loc')


    def is_move_valid(self, n_state, oldState):
        """ Returns True if the move is valid, False otherwise. """
        if n_state not in self.state_space:
            return False
        if (oldState % self.m == 0 and n_state % self.m == self.m - 1) or \
        (oldState % self.m == self.m - 1 and n_state % self.m == 0):
            return False  # Prevent wrapping from one row to the other
        return True

    def int_P(self):
        P = {}
        for state in self.state_space:
            for action in self.actions:
                reward = self.step_reward
                n_state = state + self.action_space[action]

                # Check if the move is valid
                if not self.is_move_valid(n_state, state):
                    reward = self.invalid_move_penalty
                    n_state = state  # Stay in the current state if move is invalid

                # Check for specific items in the new state
                if n_state in self.items.get('fire').get('loc'):
                    reward += self.items.get('fire').get('reward')
                elif n_state in self.items.get('water').get('loc'):
                    reward += self.items.get('water').get('reward')

                P[(state, action)] = (n_state, reward)
        return P


def print_v(v, grid):
    v = np.reshape(v, (grid.n, grid.m))

    cmap = plt.cm.get_cmap('Greens', 10)
    norm = plt.Normalize(v.min(), v.max())
    rgba = cmap(norm(v))

    for w in grid.items.get('water').get('loc'):
        idx = np.unravel_index(w, v.shape)
        rgba[idx] = 0.0, 0.5, 0.8, 1.0

    for f in grid.items.get('fire').get('loc'):
        idx = np.unravel_index(f, v.shape)
        rgba[idx] = 1.0, 0.5, 0.1, 1.0

    fig, ax = plt.subplots()
    im = ax.imshow(rgba, interpolation='nearest')

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] != 0:
                text = ax.text(j, i, v[i, j], ha="center", va="center", color="w")

    plt.axis('off')
    # plt.savefig('deterministic_v.jpg', bbox_inches='tight', dpi=200)
    plt.show()


def print_policy(v, policy, grid):
    v = np.reshape(v, (grid.n, grid.m))
    policy = np.reshape(policy, (grid.n, grid.m))

    cmap = plt.cm.get_cmap('Greens', 10)
    norm = plt.Normalize(v.min(), v.max())
    rgba = cmap(norm(v))

    for w in grid.items.get('water').get('loc'):
        idx = np.unravel_index(w, v.shape)
        rgba[idx] = 0.0, 0.5, 0.8, 1.0

    for f in grid.items.get('fire').get('loc'):
        idx = np.unravel_index(f, v.shape)
        rgba[idx] = 1.0, 0.5, 0.1, 1.0

    fig, ax = plt.subplots()
    im = ax.imshow(rgba, interpolation='nearest')

    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            if v[i, j] != 0:
                text = ax.text(j, i, policy[i, j], ha="center", va="center", color="w")

    plt.axis('off')
    # plt.savefig('deterministic_policy.jpg', bbox_inches='tight', dpi=200)
    plt.show()
    
def interate_values(grid, v , policy, gamma, theta):
    converged = False
    while not converged:
        DELTA = 0
        for state in grid.state_space:
            if grid.check_terminal(state):
                v[state] = 0
            else:
                old_v = v[state]
                new_v = []
                for action in grid.actions:
                    (n_state, reward) = grid.P.get((state, action))
                    new_v.append(reward + gamma * v[n_state])

                v[state] = max(new_v)
                DELTA = max(DELTA, np.abs(old_v - v[state]))
        converged = True if DELTA < theta else False

    for state in grid.state_space:
        new_vs = []
        for action in grid.actions:
            (n_state, reward) = grid.P.get((state, action))
            new_vs.append(reward + gamma * v[n_state])

        new_vs = np.array(new_vs)
        best_action_idx = np.where(new_vs == new_vs.max())[0]
        policy[state] = grid.actions[best_action_idx[0]]

    return v, policy



if __name__ == '__main__':
    grid_size = (18, 18)
    items = {'fire': {'reward': -10, 'loc': [1]},
             'water': {'reward': 10, 'loc': [126]}}

    gamma = 1.0
    theta = 1e-10

    v = np.zeros(np.prod(grid_size))
    policy = np.full(np.prod(grid_size), 'n')

    env = GridWorld(grid_size, items)

    v, policy = interate_values(env, v, policy, gamma, theta)

    print_v(v, env)
    print_policy(v, policy, env)
