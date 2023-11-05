import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, gridSize, items):
        self.step_reward = -1
        self.m = gridSize[0]
        self.n = gridSize[1]
        self.grid = np.random.random(gridSize)
        # print(self.grid)
        self.items = items

        # self.state_space = (self.m, self.n)
        self.state_space = [(i, j) for i in range(self.m) for j in range(self.n)]
        # print(self.state_space)

        # self.action_space = {'U': -self.m, 'D': self.m, 'L': -1, 'R': 1}
        self.action_space = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.actions = ['U', 'D', 'L', 'R']

        self.P = self.int_P()

    def set_step_reward(self, state_x, state_y, action):
        # Calculate the future state based on the action
        # i, j = state
        i, j = state_x, state_y
        # print (i, j)
        # print (action)
        if action in self.action_space:
            move = self.action_space[action]
            # print(action)
            future_state = (i + move[0], j + move[1])
            # print(future_state)
        else:
            return -1  # Invalid action

        # Check if the future state is within the map's boundaries
        if 0 <= future_state[0] < self.m and 0 <= future_state[1] < self.n:
        # Check if the second element of future_state is "G"
          if self.grid[future_state[0]][future_state[1]] == "G":
              return 0  # Return a different reward for "G" condition
          else:
              # print(self.grid[future_state[0]][future_state[1]])
              return self.grid[future_state[0]][future_state[1]]  # Default reward
        else:
            return -1  # Default step reward for invalid moves
    def int_P(self):
        P = {}
        # print(self.state_space)
        for state_i, state_j in self.state_space:
            for action in self.actions:
                # reward = self.set_step_reward(state_i, state_j, action)
                reward = -1
                # n_state = state + self.action_space[action]
                move = self.action_space[action]
                n_state = (state_i + move[0], state_j + move[1])


                if n_state in self.items.get('fire').get('loc'):
                    reward += self.items.get('fire').get('reward')
                elif n_state in self.items.get('water').get('loc'):
                    reward += self.items.get('water').get('reward')
                elif self.check_move(n_state, state_i, state_j):
                    n_state = (state_i, state_j)

                P[(state_i, state_j ,action)] = (n_state, reward)
                # print("state")
                # print(state_i, state_j)
                # print ("action")
                # print(action)
                # print("next state")
                # print(n_state)
                # print("reward")
                # print(reward)
                # print(P[(state_i, state_j ,action)])
        return P

    def check_terminal(self, state):
        return state in self.items.get('fire').get('loc') + self.items.get('water').get('loc')

    def check_move(self, n_state, oldState_i, oldState_j):
        if n_state not in self.state_space:
            return True
        # elif oldState_i % self.m == 0 and n_state[0] % self.m == self.m - 1:
        #     return True
        # elif oldState_i % self.m == self.m - 1 and n_state[0] % self.m == 0:
        #     return True
        else:
            return False

def print_v(v, grid):
    v = np.reshape(v, (grid.m, grid.n))
    print(v.shape)

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
    i = 0
    while not converged:
        DELTA = 0
        for state in grid.state_space:
            i += 1
            # print (state)
            if  grid.check_terminal(state):
                v[state] = 0

            else:
                old_v = v[state]
                new_v = []
                for action in grid.actions:
                    (n_state, reward) = grid.P.get((state[0], state[1], action))
                    # print(reward)
                    # if reward == 9:
                    #     print(reward + gamma * v[n_state])
                    new_v.append(reward + gamma * v[n_state])
                    # print(new_v)

                v[state] = max(new_v)
                # print(old_v - v[state])
                DELTA = max(DELTA, np.abs(old_v - v[state]))
                converged = True if DELTA < theta else False

        for state in grid.state_space:
            i += 1
            new_vs = []

            for action in grid.actions:
                (n_state, reward) = grid.P.get((state[0], state[1], action))
                new_vs.append(reward + gamma * v[n_state])

            new_vs = np.array(new_vs)
            best_action_idx = np.where(new_vs == new_vs.max())[0]

            # Assign the index of the best action to the policy array
            policy[state[0], state[1]] = best_action_idx[0]

    print(i, 'iterations of state space')
    return v, policy


if __name__ == '__main__':

    grid_size = (2, 2)
    items = {'fire': {'reward': -10, 'loc': [12]},
             'water': {'reward': 10, 'loc': [65]}}

    gamma = 1.0
    theta = 1e-10

    v = np.zeros((grid_size[0], grid_size[1]))
    policy = np.full((grid_size[0], grid_size[1]), 'n')

    env = GridWorld(grid_size, items)

    v, policy = interate_values(env, v, policy, gamma, theta)

    print_v(v, env)
    print_policy(v, policy, env)