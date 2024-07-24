# Value Iteration

![output of value iteration algorithm](https://github.com/hamidthri/Best-Route-VI/blob/master/VI.png)



Implemented `value_iteration_best_route`, optimizing pathfinding in a grid-based environment using the value iteration algorithm.

## State and Action Transitions

In the GridWorld environment, agents can perform actions to navigate through a grid-based world. Each action leads to a transition between states in the grid. Here's a detailed explanation of how state and action transitions are handled:

### State Representation

- **State Indexing**: The state in the grid is represented as a single index in a 1D array, which is computed using the formula: `state = row * num_cols + col`. This allows us to manage the state space as a flat array.

### Actions

- **Action Set**: The agent can choose from four possible actions:
  - `'U'` (Up): Move to the cell directly above the current cell.
  - `'D'` (Down): Move to the cell directly below the current cell.
  - `'L'` (Left): Move to the cell directly to the left of the current cell.
  - `'R'` (Right): Move to the cell directly to the right of the current cell.

- **Action Effects**: Each action modifies the state index by adding or subtracting a value based on the direction:
  - `'U'`: Subtract `self.m` (number of columns) from the current state index.
  - `'D'`: Add `self.m` to the current state index.
  - `'L'`: Subtract `1` from the current state index.
  - `'R'`: Add `1` to the current state index.

### Transition Function

- **Transition Calculation**: The transition from a current state to a new state is calculated by updating the state index according to the chosen action. For example:
  - If the current state is `5` and the action is `'U'`, the new state is `5 - self.m`.

- **Validity Check**: The validity of a move is checked to ensure that the new state:
  - Remains within the grid boundaries.
  - Does not wrap around to the opposite edge (e.g., moving left from the leftmost column or moving up from the top row).

### Handling Invalid Moves

- **Invalid Moves**: If an action results in a state that is outside the grid boundaries or violates grid constraints (like moving off the edge), the agent stays in the current state. This is achieved by checking the validity of the new state and reverting to the current state if the move is invalid.

### Example

Consider a GridWorld of size `4x4` where the state is represented as a single index. Suppose the current state is `5` (which corresponds to the cell in the 2nd row and 1st column). Here’s how actions affect transitions:

- **Action `'U'`**: 
  - New state = `5 - 4 = 1` (Move Up to cell (1, 1)).

- **Action `'D'`**:
  - New state = `5 + 4 = 9` (Move Down to cell (2, 1)).

- **Action `'L'`**:
  - New state = `5 - 1 = 4` (Move Left to cell (1, 0)).

- **Action `'R'`**:
  - New state = `5 + 1 = 6` (Move Right to cell (1, 2)).

The GridWorld environment manages state transitions based on actions and ensures that all moves are valid within the grid. This allows agents to explore the grid and learn optimal policies through reinforcement learning algorithms like value iteration.

## Value Iteration

Value Iteration is a fundamental algorithm in reinforcement learning used to compute the optimal policy for a given Markov Decision Process (MDP). It iteratively updates the value function until it converges to the optimal values, from which the optimal policy can be derived. Here’s an overview of how the Value Iteration algorithm works:

### Overview

1. **Initialization**: 
   - **Value Function**: Start with an initial guess for the value function, usually setting all state values to zero.
   - **Policy**: The policy is initially arbitrary or undefined.

2. **Value Update**:
   - For each state in the grid, compute the value based on the expected return from each possible action.
   - The value function is updated using the Bellman equation, which accounts for the immediate reward and the discounted future rewards. The Bellman equation for value iteration is given by:

     ```
     V(s) <- max_a [ R(s, a) + γ * Σ_s' P(s' | s, a) * V(s') ]
     ```

     where:
     - `V(s)` is the value of state `s`.
     - `R(s, a)` is the reward received after taking action `a` in state `s`.
     - `γ` is the discount factor.
     - `P(s' | s, a)` is the transition probability of moving to state `s'` from state `s` given action `a`.
     - `V(s')` is the value of the next state `s'`.

3. **Convergence Check**:
   - Continue updating the value function until the change in value function (measured as `Δ`) is below a small threshold `θ`. This indicates that the value function has converged to the optimal values.

4. **Policy Extraction**:
   - Once the value function converges, derive the optimal policy by choosing the action that maximizes the expected value for each state:

     ```
     π(s) = argmax_a [ R(s, a) + γ * Σ_s' P(s' | s, a) * V(s') ]
     ```

     where `π(s)` is the optimal action for state `s`.

### Key Concepts

- **Discount Factor (γ)**: Determines the importance of future rewards. A value close to 1 makes future rewards nearly as important as immediate rewards, while a value close to 0 emphasizes immediate rewards more.

- **Convergence Threshold (θ)**: A small positive value that determines when the algorithm has sufficiently converged. If the maximum change in the value function across all states is less than `θ`, the algorithm stops.

- **Optimal Policy**: The policy derived after value iteration is optimal, meaning it maximizes the expected cumulative reward starting from any state.

### Advantages

- **Simplicity**: Value Iteration is relatively straightforward to implement and understand.
- **Optimality**: It guarantees finding the optimal policy if the algorithm converges.

### Limitations

- **Computational Complexity**: Value Iteration can be computationally expensive, especially for large state spaces, as it requires iterating over all states and actions repeatedly.
- **Memory Usage**: Storing the value function and policy for large state spaces can require significant memory.

### Summary

Value Iteration is a powerful and widely used algorithm in reinforcement learning for solving MDPs. It provides a systematic way to compute the optimal value function and policy by iteratively refining the value estimates until convergence. This method is particularly useful in environments with well-defined state and action spaces.
