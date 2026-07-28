# Chapter 8 – Planning and Learning with Tabular Methods

This chapter unifies **planning** and **learning** methods in Reinforcement Learning.

### Planning Methods
- Require a **model of the environment**.
- Examples:
  - Dynamic Programming (DP)
  - Heuristic Search

### Learning Methods
- Do **not** require a model.
- Learn directly from experience.
- Examples:
  - Monte Carlo (MC)
  - Temporal-Difference (TD)

## Similarity

Both planning and learning are based on:

- Computing **value functions**
- Looking ahead to future outcomes
- Performing a **backup**
- Updating estimated value functions

## Previous Connection

Earlier chapters showed that:
- Monte Carlo and TD are closely related.
- They can be unified using **Eligibility Traces**, e.g., **TD(λ)**.

## Goal of This Chapter

- Explore the relationship between **planning** and **learning**.
- Show how they can be integrated instead of treated as separate approaches.

# 8.1 Models and Planning

## Environment Model
- anything that an agent can use to predict how the environment will respond to its actions.
- A **model** predicts the environment's response to an action:
- Input: state `S` and action `A`
- Output: next state `S'` and reward `R`

## Types of Models

### Distribution Model
- Returns **all possible outcomes** with their probabilities.
- Used in **Dynamic Programming (DP)**.
- More informative but often harder to build.

### Sample Model
- Returns **one sampled outcome** according to the probability distribution.
- Easier to obtain in practice.
- Used in many simulation-based RL methods.


## Simulated Experience
Models generate artificial experience instead of interacting with the real environment.

- Sample model → one transition or one episode.
- Distribution model → all possible transitions or episodes.

## Planning
Planning is any computational process that:
- Uses a **model**(input).
- Produces or improves a **policy**.

```
Model → Planning → Policy
```

## Planning Approaches

### State-Space Planning
- Searches over **states**.
- Actions cause state transitions.
- Value functions are defined over states.
- Main focus of this book.

### Plan-Space Planning
- Searches over **plans** instead of states.
- Common in classical AI.
- Not emphasized in reinforcement learning.

## Unified View

All state-space planning methods follow:

```
Model
    ↓
Simulated Experience
    ↓
Backup Operations
    ↓
Value Functions
    ↓
Policy Improvement
```

## Planning vs Learning

| Planning | Learning |
|----------|----------|
| Uses simulated experience | Uses real experience |
| Requires a model | Does not require a model |
| Computes values via backups | Computes values via backups |

## Random-Sample One-Step Tabular Q-Planning

Algorithm:
1. Randomly choose `(S, A)`.
2. Query the sample model.
3. Receive `(R, S')`.
4. Update using one-step Q-learning:

$$
Q(S,A)\leftarrow Q(S,A)+\alpha\left[R+\gamma\max_aQ(S',a)-Q(S,A)\right]
$$

It converges under the same conditions as tabular Q-learning.

# 8.2 Integrated Planning, Acting, and Learning

## Online Planning
Planning can occur while the agent interacts with the environment.

New experience can:
- Improve the environment model.
- Improve the value function directly.
- Influence future planning.

## Two Uses of Real Experience

### Model Learning
- Updates the environment model.
- Makes planning more accurate.

### Direct Reinforcement Learning
- Updates value functions and policy directly from real experience.

Indirect RL refers to improving the policy through the learned model and planning.

## Direct vs. Indirect Methods

### Direct RL
- Simpler.
- Independent of model quality.
- Learns only from real experience.

### Indirect RL
- Uses experience more efficiently.
- Requires fewer interactions.
- Depends on the accuracy of the model.

## Dyna-Q Components

Dyna-Q integrates four processes:

- Acting
- Direct Reinforcement Learning
- Model Learning
- Planning

## Dyna Architecture

```text
Environment
      │
      ▼
Real Experience
   ├────────────► Direct RL
   │
   └────────────► Model Learning
                      │
                      ▼
                   Model
                      │
                      ▼
           Simulated Experience
                      │
                      ▼
                  Planning
                      │
                      ▼
            Update Value Function
```

## Search Control
Chooses which previously observed state-action pairs are simulated during planning.

## Dyna-Q Algorithm

For each real step:

1. Observe current state.
2. Select action using ε-greedy.
3. Execute action.
4. Update Q-value (Q-learning).
5. Update the model.
6. Repeat **n** planning steps:
   - Sample a previously observed state-action pair.
   - Query the model.
   - Apply the same Q-learning update.

# 8.3 When the Model Is Wrong

## Why Models Can Be Incorrect

A model may be inaccurate because:

- The environment is stochastic and only limited samples are available.
- Function approximation introduces errors.
- The environment changes over time.

An incorrect model leads to **suboptimal planning**.

## Optimistic Model Errors

If the model predicts better rewards or transitions than actually exist:

- The agent follows the optimistic policy.
- Real interaction quickly reveals the mistake.
- The model is corrected.

## Hidden Environmental Changes

A more difficult case occurs when:

- The environment becomes **better**.
- The current policy never explores the improved area.

As a result, the agent may never discover the change.

## Exploration vs. Exploitation in Planning

- **Exploration:** Try actions to improve the model.
- **Exploitation:** Follow the best policy according to the current model.

Planning must balance both objectives.

## Dyna-Q+

Dyna-Q+ encourages exploration by tracking how long each state-action pair has not been tried.

For a transition with reward `R` and elapsed time `τ`:

$$
R' = R + \kappa \sqrt{\tau}
$$

where:

- `τ` = time since the last real visit.
- `κ` = small exploration constant.

Older state-action pairs receive larger exploration bonuses.

# 8.4 Prioritized Sweeping

## Motivation

Randomly selecting state-action pairs for planning is inefficient because many backups have little or no effect.

Planning becomes more efficient by focusing on state-action pairs whose values are expected to change the most.



Instead of random backups:

- Start from states whose values have changed.
- Propagate value changes backward through predecessor states.
- Perform the most important backups first.

## Backward Propagation

When a state's value changes:

1. Update actions leading to that state.
2. Update predecessor states.
3. Continue propagating changes backward until convergence.

## Priority Queue

Maintain a priority queue of state-action pairs.

Priority is based on the magnitude of the expected value change:

$$
P = |R + \gamma \max_a Q(S',a) - Q(S,a)|
$$

Only pairs with:

$$
P > \theta
$$

are inserted into the queue.

Higher-priority pairs are processed first.

## Prioritized Sweeping Algorithm

For each real interaction:

1. Execute an action.
2. Update the model.
3. Compute backup priority.
4. Insert important pairs into the priority queue.
5. Repeat up to **n** planning backups:
   - Remove the highest-priority pair.
   - Perform the backup.
   - Compute priorities of predecessor pairs.
   - Insert significant predecessors into the queue.
# 8.5 Expected vs. Sample Updates

## Three Dimensions of One-Step Backups

One-step backups differ along **three binary dimensions**:

1. **State Value** ($v$) vs. **Action Value** ($q$)
2. **Optimal Policy** ($v^*, q^*$) vs. **Policy Evaluation** ($v^\pi, q^\pi$)
3. **Expected Backup** vs. **Sample Backup**

These three dimensions produce **eight possible one-step backup types**, seven of which correspond to well-known reinforcement learning algorithms (e.g., Value Iteration, Policy Evaluation, TD(0), Q-Learning, and Sarsa).

---

# Expected Backup

An **expected backup** considers **all possible successor states** and their probabilities.

$$
Q(s,a)=
\sum_{s',r}
\hat{p}(s',r|s,a)
\left[
r+\gamma\max_{a'}Q(s',a')
\right]
$$

### Advantages

- Uses the complete probability distribution.
- Produces an exact expected target (assuming the model is correct).
- No sampling error.

### Disadvantages

- Computationally expensive.
- Must evaluate **every possible successor state**.
- Cost increases with the branching factor.

---

# Sample Backup

A **sample backup** uses **only one sampled transition** from the model (or environment).

$$
Q(s,a)
\leftarrow
Q(s,a)
+
\alpha
\left[
R+\gamma\max_{a'}Q(S',a')
-
Q(s,a)
\right]
$$

### Advantages

- Computationally inexpensive.
- Requires only one successor state.
- Makes fast incremental updates.
- Well suited for large state spaces.

### Disadvantages

- Introduces sampling error.
- A single update is less accurate than an expected backup.

---

# Branching Factor

The **branching factor** ($b$) is the number of possible successor states for a given state-action pair.

$$
b = |\{\,s' \mid \hat{p}(s'|s,a) > 0\,\}|
$$

- One **expected backup** requires evaluating approximately **$b$ successor states**.
- One **sample backup** evaluates **only one successor state**.
- Therefore, an expected backup typically requires about **$b$ times more computation** than a sample backup.

---

# Why Sample Backups Can Be Better

Although an expected backup is more accurate, it is also much more expensive.

In large problems, the same computational budget can be used to perform **many sample backups** instead of **one expected backup**.

As a result:

- More state-action pairs are updated.
- Information spreads through the state space more quickly.
- Learning and planning often progress faster.

This is why algorithms such as **Dyna-Q** use **sample backups**, while **Prioritized Sweeping** often uses **expected backups** in stochastic environments.

---

# Comparison

| Expected Backup | Sample Backup |
|-----------------|---------------|
| Uses all possible successor states | Uses one sampled successor state |
| Computes the exact expected target | Computes an approximate target |
| No sampling error | Contains sampling error |
| High computational cost | Low computational cost |
| Better per individual update | Better when computation is limited |
| Preferred for small problems | Preferred for large problems |

---



> **Expected backups are more accurate, but sample backups are often more computationally efficient.**

# 8.6 Trajectory Sampling

## Two Backup Distribution Strategies

### 1. Uniform Sweeps

- Perform backups over the entire state (or state-action) space.
- Every state is updated equally.
- Computationally expensive for large problems.

### 2. Trajectory Sampling

- Generate simulated trajectories by following the current policy.
- Perform backups only on visited state-action pairs.
- Focus computation on relevant parts of the state space.

---

## On-Policy Distribution

Trajectory sampling follows the **on-policy distribution**:

- States are sampled according to the current policy.
- Experience is generated by simulating trajectories.
- Rewards and transitions come from the model.

---

## Advantages

- Avoids wasting computation on rarely visited states.
- Focuses planning on states likely to occur.
- Much more efficient in large state spaces.
- Easy to implement using model simulation.

---

## Disadvantages

- Frequently revisits the same states.
- May ignore unexplored regions.
- Long-term planning can become slower than uniform backups.

---

## Experimental Findings

- On-policy trajectory sampling improves planning speed initially.
- The advantage is larger when:
  - The state space is large.
  - The branching factor is small.
- Uniform backups may eventually outperform trajectory sampling in small problems.

---
# 8.7 Real-Time Dynamic Programming (RTDP)

## Idea

Real-Time Dynamic Programming (RTDP) is an **on-policy trajectory-sampling** version of **Value Iteration**.

Instead of performing full sweeps over every state, RTDP updates only the states that are actually visited during real or simulated trajectories.

---

# How RTDP Works

For each visited state:

1. Choose the **greedy action** according to the current value function.
2. Perform an **expected Value Iteration backup**.
3. Move to the next state.
4. Repeat until reaching the terminal (goal) state.

Unlike classical Dynamic Programming, RTDP does **not** update every state.

---

# Why RTDP?

Many states are never useful.

Some states:

- Cannot be reached from the start state.
- Cannot appear under an optimal policy.

Updating these states wastes computation.

RTDP focuses only on states that are likely to matter.

---

# Relevant vs. Irrelevant States

**Relevant States**

- Reachable from a start state.
- May appear under an optimal policy.
- Need accurate value estimates.

**Irrelevant States**

- Never visited by any optimal policy.
- Their values are unnecessary.
- RTDP may never update them.

---

# Relationship to Asynchronous Dynamic Programming

RTDP is an example of **Asynchronous Dynamic Programming**.

Instead of systematic sweeps:

- States are updated whenever they are visited.
- Update order depends on trajectories.

---

# Convergence Conditions

For stochastic shortest-path problems, RTDP converges to the optimal policy if:

1. Goal states have initial value **0**.
2. A policy exists that always reaches a goal.
3. Every non-goal transition has a **negative reward**.
4. Initial state values are **optimistic** (greater than or equal to the optimal values).

Under these conditions, RTDP converges with probability 1.

---

# Advantages

- Avoids unnecessary updates.
- Focuses computation on important states.
- Scales well to very large state spaces.
- Produces a near-optimal policy much earlier than classical Value Iteration.
- Suitable for online planning.

---

# Disadvantages

- Works best when trajectories visit all relevant states.
- Performance depends on the current policy.
- May ignore useful states if exploration is insufficient.

---

# RTDP vs. Classical Value Iteration

| Classical Value Iteration | RTDP |
|---------------------------|------|
| Updates every state | Updates only visited states |
| Full state sweeps | Trajectory sampling |
| High computational cost | Lower computational cost |
| Treats all states equally | Focuses on relevant states |
| Better for small problems | Better for very large problems |

---


> **RTDP performs Dynamic Programming only where the agent is likely to go.**

Instead of spending computation on every state, RTDP follows trajectories generated by the current greedy policy and updates only those states.

This makes planning much more efficient for large reinforcement learning problems.

# 8.8 Planning at Decision Time

## Two Ways of Using Planning

Planning can be used in two different ways:

1. **Background Planning**
2. **Decision-Time Planning**

---

# 1. Background Planning

Planning is performed continuously before action selection.

The agent gradually improves its:

- Policy
- Value Function

using simulated experience.

When the agent reaches a state, it simply looks up the best action.

Examples:

- Dynamic Programming
- Dyna

---

## Characteristics

- Planning occurs in the background.
- Not focused on the current state.
- Results are stored.
- Fast action selection.

---

# 2. Decision-Time Planning

Planning starts **after** reaching the current state.

For every decision:

1. Observe the current state.
2. Simulate future trajectories.
3. Evaluate candidate actions.
4. Select the best action.

Planning restarts from scratch at the next state.

---

## Characteristics

- Focuses only on the current state.
- Can search many steps ahead.
- Results are often discarded after selecting the action.

---

# Comparison

| Background Planning | Decision-Time Planning |
|---------------------|------------------------|
| Plans before acting | Plans after reaching a state |
| Improves the global policy | Chooses one action |
| Stores learned values | Usually discards results |
| Fast decision making | Slow but more informed decisions |

---

# Can They Be Combined?

Yes.

Many systems combine both approaches:

- Background planning improves the policy.
- Decision-time planning refines the current action when needed.

---

# Applications

**Background Planning**

- Robotics
- Autonomous driving
- Real-time control

**Decision-Time Planning**

- Chess
- Go
- Strategic games

---
# 8.9 Heuristic Search

## Definition

Heuristic Search is a **decision-time planning** method.

For every current state:

1. Build a search tree.
2. Evaluate the leaf nodes using the approximate value function.
3. Backup values toward the root.
4. Select the best action.
5. Discard the search tree.

---

Instead of improving the global value function, heuristic search focuses computation on selecting the **best action for the current state**.

---

# How It Works

1. Observe the current state.
2. Expand a search tree.
3. Evaluate leaf nodes.
4. Perform expected backups from leaves to the root.
5. Choose the action with the highest backed-up value.

---

# Characteristics

- Decision-time planning.
- Focuses on the current state.
- Searches multiple future steps.
- Usually does **not** store backup values.

---

# Relationship with Greedy Policy

A greedy policy looks **one step ahead**.

Heuristic Search extends this idea to **multiple future steps**.

> **Heuristic Search can be viewed as a multi-step greedy policy.**

---

# Advantages

- Better action selection.
- Focuses computation on important states.
- Works well with accurate models.
- Effective for large search spaces such as board games.

---

# Disadvantages

- Computationally expensive.
- Deeper search increases response time.
- Most computed values are discarded after action selection.

---

# Example: TD-Gammon

TD-Gammon:

- Learned an afterstate value function using Temporal Difference learning.
- Used heuristic search to choose moves.
- Deeper search produced stronger play but required more computation.

---

# Heuristic Search vs. Dyna

| Dyna | Heuristic Search |
|------|------------------|
| Background planning | Decision-time planning |
| Updates the value function permanently | Usually discards computed values |
| Learns continuously | Searches only for the current decision |
| Planning improves future decisions | Planning improves only the current decision |

---

# 8.10 Rollout Algorithms

## Idea

Rollout is a **Decision-Time Planning** algorithm based on **Monte Carlo Control**.

Instead of learning a complete value function, it estimates action values **only for the current state**.

---

## Basic Procedure

For every action in the current state:

1. Take that action.
2. Simulate many complete trajectories using a fixed **Rollout Policy**.
3. Compute the average return.
4. Estimate the action value.

Finally:

- Choose the action with the highest estimated value.
- Execute it.
- Repeat the entire process in the next state.

---

## Monte Carlo Estimation

For each action:

$$
q^\pi(s,a)
=
\text{Average(Returns of simulated trajectories)}
$$

The trajectories:

- start from the current state,
- begin with action $a$,
- then follow the rollout policy.

---

## Rollout Policy

The rollout policy is the policy used **after the first action** during simulation.

It can be:

- Random
- Heuristic
- Learned policy

A better rollout policy usually produces better decisions.

---

## Relation to Policy Improvement

Using the **Policy Improvement Theorem**:

If

$$
q^\pi(s,a)\ge v^\pi(s)
$$

then replacing the current action with $a$ cannot make the policy worse.

Therefore:

- Estimate $q^\pi(s,a)$ using Monte Carlo.
- Choose the action with the largest estimate.
- Obtain an improved policy for the current state.

---

## Characteristics

Unlike standard Monte Carlo Control:

- Does **not** learn the entire value function.
- Does **not** store long-term Q-values.
- Computes values only for the current decision.
- Discards them after selecting the action.

---

## Advantages

- Simple implementation.
- No need for value-function approximation.
- No need to evaluate every state-action pair.
- Works well with sample models.
- Often significantly improves a baseline policy.

---

## Disadvantages

- Computationally expensive.
- Must simulate many trajectories.
- Decision time increases with:
  - number of actions,
  - trajectory length,
  - rollout policy complexity,
  - number of simulations.

---

## Speed-up Techniques

- Parallel Monte Carlo simulations.
- Truncate trajectories and bootstrap with a value function.
- Prune actions that are unlikely to be optimal.

---

## Rollout vs Monte Carlo Control

| Rollout | Monte Carlo Control |
|---------|---------------------|
| Decision-time planning | Learning algorithm |
| Current state only | All states |
| No long-term memory | Stores value estimates |
| Values discarded after action | Values retained |
| Improves rollout policy | Learns optimal policy |

---

# 8.11 Monte Carlo Tree Search (MCTS)

Monte Carlo Tree Search (MCTS) is a **Decision-Time Planning** algorithm based on **Rollout**.

Unlike simple Rollout, MCTS **stores action-value estimates** from previous simulations and uses them to guide future simulations toward more promising trajectories.

---

## Main Characteristics

- Decision-time planning
- Based on Monte Carlo simulations
- Builds a search tree incrementally
- Focuses computation on promising states
- Widely used in Go, Chess, AlphaGo, and many planning problems

---

## Basic Procedure

Each time the agent reaches a new state:

1. Build (or reuse) a search tree rooted at the current state.
2. Run many Monte Carlo simulations.
3. Improve action-value estimates inside the tree.
4. Choose the best action.
5. Move to the next state.
6. Repeat.

---

## Tree Structure

The tree stores only states that are likely to be visited soon.

- Root = current state.
- Nodes = future states.
- Edges = actions.

The tree grows gradually as more simulations are performed.

---

## Tree Policy vs Rollout Policy

### Tree Policy

Used **inside the tree**.

Chooses actions using stored statistics.

Common choices:

- ε-Greedy
- UCB

Balances:

- Exploration
- Exploitation

---

### Rollout Policy

Used **outside the tree**.

Usually simple:

- Random
- Heuristic
- Fixed policy

---

## Four Steps of MCTS

### 1. Selection

Traverse the tree from the root to a leaf using the **Tree Policy**.

---

### 2. Expansion

Add one or more unexplored child nodes.

---

### 3. Simulation

Run a complete simulated trajectory.

- Inside tree → Tree Policy
- Outside tree → Rollout Policy

Obtain a Monte Carlo return.

---

### 4. Backup

Propagate the return back through the selected tree path.

Update action-value estimates stored on tree edges.

---

## Advantages

- Focuses simulations on promising trajectories.
- Efficient use of computation.
- No need to estimate the entire value function.
- Learns from previous simulations.
- Works well in huge search spaces.

---

## Disadvantages

- Requires many simulations.
- Computationally expensive.
- Performance depends on:
  - Tree Policy
  - Rollout Policy
  - Available computation time

---

## Rollout vs MCTS

| Rollout | MCTS |
|---------|------|
| Independent simulations | Guided simulations |
| No memory | Stores tree statistics |
| No search tree | Incrementally builds a tree |
| Values discarded | Values reused during planning |
| Simple | More efficient and powerful |

---

## Applications

- Go
- Chess
- Backgammon
- General Game Playing
- Robotics
- Sequential Decision Problems

---

## AlphaGo

AlphaGo combines:

- Monte Carlo Tree Search (MCTS)
- Deep Neural Networks
- Self-Play Reinforcement Learning

This combination enabled superhuman performance in Go.

---

