import numpy as np
import matplotlib.pyplot as plt

class GridWorld(object):
    def __init__(self, gridSize, items):
        self.step_reward = -1
        self.m = gridSize[0]
        self.n = gridSize[1]
        self.grid = np.random.randint(0, 50, gridSize)
        self.grid = np.array([[0,  -1, 0],
                              [0, -1, 0],
                              [0, -1, 0]])
        print((self.grid))
        self.items = items

        self.state_space = [(i, j) for i in range(self.m) for j in range(self.n)]
        self.action_space = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}
        self.actions = ['U', 'D', 'L', 'R']

        self.P = self.int_P()

    def set_step_reward(self, state_x, state_y, action):
        i, j = state_x, state_y
        if action in self.action_space:
            move = self.action_space[action]
            future_state = (i + move[0], j + move[1])
        else:
            return -1

        if 0 <= future_state[0] < self.m and 0 <= future_state[1] < self.n:
          if self.grid[future_state[0]][future_state[1]] == "G":
              return 0 
          else:
              return self.grid[future_state[0]][future_state[1]]
        else:
            return 1
    def int_P(self):
        P = {}
        for state_i, state_j in self.state_space:
            for action in self.actions:
                reward = -self.set_step_reward(state_i, state_j, action)
                move = self.action_space[action]
                n_state = (state_i + move[0], state_j + move[1])


                if np.array_equal(n_state, self.items.get('fire').get('loc')):
                    reward += self.items.get('fire').get('reward')
                elif np.array_equal(n_state, self.items.get('water').get('loc')):
                    reward += self.items.get('water').get('reward')
                elif self.check_move(n_state, state_i, state_j):
                    n_state = (state_i, state_j)
                P[(state_i, state_j ,action)] = (n_state, reward)
                # print(f"state: {state_i} {state_j}")
                # print(f"n state: {n_state}, reward: {reward}")

        # print(P)
        return P

    def check_terminal(self, state):
        return (state[0], state[1]) == (self.items.get('fire').get('loc').tolist()[0], self.items.get('fire').get('loc').tolist()[1]) or (state[0], state[1]) == (self.items.get('water').get('loc').tolist()[0], self.items.get('water').get('loc').tolist()[1])

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
    # print(v.shape)

    cmap = plt.cm.get_cmap('Greens', 10)
    norm = plt.Normalize(v.min(), v.max())
    rgba = cmap(norm(v))

    # for w in grid.items.get('water').get('loc').tolist():
    #     idx = np.unravel_index(w, v.shape)
    rgba[tuple(grid.items.get('water').get('loc').tolist())] = 0.0, 0.5, 0.8, 1.0

    # for f in grid.items.get('fire').get('loc'):
    #     idx = np.unravel_index(f, v.shape)
    rgba[tuple(grid.items.get('fire').get('loc').tolist())] = 1.0, 0.5, 0.1, 1.0

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
    print(policy)
    cmap = plt.cm.get_cmap('Greens', 10)
    norm = plt.Normalize(v.min(), v.max())
    rgba = cmap(norm(v))
    rgba[tuple(grid.items.get('water').get('loc').tolist())] = 0.0, 0.5, 0.8, 1.0
    rgba[tuple(grid.items.get('fire').get('loc').tolist())] = 1.0, 0.5, 0.1, 1.0
    fig, ax = plt.subplots()
    im = ax.imshow(rgba, interpolation='nearest')

    print(v)
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
    j = 0
    while not converged:
    # while (j < 10):
        DELTA = 0
        for state in grid.state_space:
            i += 1
            # print (state)
            # if grid.check_terminal(state):
            #     pass
                # v[state] = 0

            # else:
            old_v = v[state]
            new_v = []
            for action in grid.actions:
                (n_state, reward) = grid.P.get((state[0], state[1], action))
                # print(reward)
                # if reward == 9:
                #     print(reward + gamma * v[n_state])
                new_v.append(reward + gamma * v[n_state])

            v[state] = max(new_v)
            DELTA = max(DELTA, np.abs(old_v - v[state]))
            j += 1
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
        policy[state[0], state[1]] = grid.actions[best_action_idx[0]]

    print(i, 'iterations of state space')
    return v, policy


if __name__ == '__main__':

    grid_size = (3, 3)
    items = {'fire': {'reward': -10, 'loc': np.asarray([0, 0])},
             'water': {'reward': 100, 'loc': np.asarray([2, 2])}}

    gamma = 0.85
    theta = 1e-1

    v = np.zeros((grid_size[0], grid_size[1]))
    policy = np.full((grid_size[0], grid_size[1]), 'n', dtype=object)

    env = GridWorld(grid_size, items)

    v, policy = interate_values(env, v, policy, gamma, theta)

    print_v(v, env)
    print_policy(v, policy, env)