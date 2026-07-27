# Chapter 6 – Temporal-Difference Learning

Temporal-Difference (TD) Learning is one of the most important ideas in Reinforcement Learning. It combines the strengths of Dynamic Programming (DP) and Monte Carlo (MC) methods.

## Characteristics

### Like Monte Carlo
- Learns directly from experience.
- Does not require a model of the environment.

### Like Dynamic Programming
- Uses bootstrapping.
- Updates value estimates using other learned estimates.
- Does not wait until the end of an episode.

## Bootstrapping

Bootstrapping means updating the value of a state using the current estimate of the next state's value instead of waiting for the final return.

## Relationship Between Methods

- DP → requires a model and uses bootstrapping.
- MC → model-free, waits until episode ends.
- TD → model-free and bootstraps.

Later chapters introduce:
- **Chapter 7:** n-step methods (bridge between TD and MC)
- **Chapter 12:** TD(λ), which unifies TD and MC.

## Focus of This Chapter

The chapter begins with the **prediction (policy evaluation)** problem:

Estimate the value function:

$$
v_{\pi}(s)
$$

for a given policy $ \pi $.

For the control problem (finding the optimal policy), DP, TD, and MC all rely on **Generalized Policy Iteration (GPI)**. Their main difference lies in how they perform prediction.

# 6.1 TD Prediction 

## Prediction with Experience

Both Monte Carlo (MC) and Temporal-Difference (TD) methods estimate the value function from experience without requiring a model of the environment.

## Constant-α Monte Carlo

MC updates the value estimate only after the episode ends:

$$
V(S_t) \leftarrow V(S_t) + \alpha \left[G_t - V(S_t)\right]
$$

- Target: $G_t$ (actual return)
- Must wait until the complete return is known.

## TD(0)

TD updates immediately after observing the next reward and state:

$$
V(S_t) \leftarrow
V(S_t)+
\alpha
\left[
R_{t+1}
+\gamma V(S_{t+1})
-
V(S_t)
\right]
$$

- Target: $R_{t+1}+\gamma V(S_{t+1})$
- Learns after every transition.
- Does not wait until the episode ends.

## Difference

| Monte Carlo | TD(0) |
|--------------|--------|
| Waits for episode termination | Updates after one step |
| Uses actual return $G_t$ | Uses reward + estimated next value |
| No bootstrapping | Uses bootstrapping |

## TD(0) Algorithm

Initialize:

- Arbitrary values for all states.
- Terminal state value = 0.

For each step:

1. Select an action using policy $ \pi $.
2. Observe reward $R$ and next state $S'$.
3. Update:

$$
V(S)\leftarrow
V(S)+
\alpha
\left[
R+\gamma V(S')-V(S)
\right]
$$

4. Move to the next state.

---

## Bootstrapping

TD updates a state's value using the current estimate of the next state's value instead of waiting for the final return.

---

## Comparison

| Method | Target |
|---------|--------|
| Monte Carlo | $G_t$ |
| Dynamic Programming | $R+\gamma v_\pi(S')$ |
| TD | $R+\gamma V(S')$ |

TD combines:
- Monte Carlo sampling
- Dynamic Programming bootstrapping

---

## Sample Update

TD updates values using a **single sampled transition**, while DP uses the **expected value over all possible successor states**.

---

## TD Error

$$
\delta_t
=
R_{t+1}
+
\gamma V(S_{t+1})
-
V(S_t)
$$

The TD error measures how much the current prediction differs from a better one.

---

## Relationship to Monte Carlo

If value estimates remain fixed during an episode,

$$
G_t-V(S_t)
=
\sum_{k=t}^{T-1}
\gamma^{k-t}\delta_k
$$

Thus, the Monte Carlo error equals the discounted sum of TD errors.

# Example 6.1 – Driving Home

## Scenario

A driver predicts the remaining travel time while driving home. As new information becomes available (rain, traffic, highway exit, slow truck), the prediction is continuously updated.

## State Values

Each state represents a point during the trip.

The value of a state is the **expected remaining travel time**.

Since:

$$
\gamma = 1
$$

the return equals the actual remaining travel time.

## Monte Carlo View

- Wait until arriving home.
- Compare each prediction with the final travel time.
- Update values only after the episode ends.

## TD View

- Update predictions immediately after every new observation.
- Shift each estimate toward the next estimate.
- Learn continuously during the trip.

## result

Monte Carlo waits for the final outcome, while TD learns online from intermediate predictions.

This makes TD faster and more efficient in changing environments.



## Exercises

### Exercise 6.1

When value estimates are updated during an episode, the identity

$$
G_t-V(S_t)
=
\sum_{k=t}^{T-1}\gamma^{k-t}\delta_k
$$

is no longer exact. The exercise asks to derive the correction term caused by changing value estimates.

---

### Exercise 6.2

TD is often more efficient because it reuses previously learned value estimates.

Example:

- Only the beginning of the driving route changes.
- Most later state values are already accurate.
- TD immediately bootstraps from these estimates.
- Monte Carlo must still wait for the complete return.

---

## Comparison

| Method | Model | Bootstrapping | Wait for Episode |
|---------|-------|---------------|------------------|
| Monte Carlo | No | No | Yes |
| Dynamic Programming | Yes | Yes | No |
| TD | No | Yes | No |

**Main takeaway:** TD provides fast online learning by updating values after every transition while leveraging existing estimates.

# 6.2 Advantages of TD Prediction Methods 

## Advantages over Dynamic Programming

- TD is **model-free**.
- It does not require transition probabilities or reward models.

## Advantages over Monte Carlo

- Learns **online** after every transition.
- Does not wait until the end of an episode.
- Suitable for very long episodes.
- Works naturally in continuing (non-episodic) tasks.
- Learns from every transition, even if future actions are exploratory.

## Convergence

TD(0) has been proven to converge to the true value function $v_\pi$:

- In mean, with a sufficiently small constant step size.
- With probability 1, if the step size decreases according to stochastic approximation conditions.

Most proofs apply to the tabular setting, with some extensions to linear function approximation.

## TD vs Monte Carlo

Both methods converge asymptotically.

However, no general mathematical proof shows that one always converges faster.

Empirically, TD usually converges faster than constant-$\alpha$ Monte Carlo on stochastic tasks.

# 6.3 Optimality of TD(0) 

## Batch Updating

When only a finite amount of experience is available, the same training data can be processed repeatedly until convergence.

Procedure:

1. Compute all TD or MC increments over the entire dataset.
2. Sum the increments.
3. Update the value function once.
4. Repeat using the updated value estimates.

This process is called **batch updating**.

---

## Properties

- Batch TD(0) converges deterministically.
- The final solution is independent of the step size, provided \(\alpha\) is sufficiently small.
- Batch constant-\(\alpha\) MC also converges.
- However, the two methods converge to **different solutions**.

---

## Key Insight

Although online TD and MC do not reach their exact batch solutions, they generally move in the direction of those solutions.

Understanding these batch solutions helps explain the fundamental difference between TD and Monte Carlo learning.

# Example 6.3 – Random Walk under Batch Updating

## Experimental Setup

- Batch TD(0) and Batch Constant-\(\alpha\) Monte Carlo were applied to the Random Walk task.
- After each new episode, **all episodes observed so far** formed the training batch.
- The same batch was repeatedly processed until convergence.

---

## Evaluation

The learned value function was compared with the true value function using:

- Root Mean Square (RMS) Error
- Averaged over the five states
- Averaged over 100 independent runs

---

## Results

- Batch TD consistently achieved lower RMS error than Batch Monte Carlo.
- TD produced better predictions despite Monte Carlo minimizing training-set error.

---

## Monte Carlo Solution

Batch Monte Carlo converges to the **sample average of observed returns**.

This minimizes the **mean squared error on the training data**.

---

## Why TD Performs Better

TD learns the underlying Markov structure through bootstrapping.

Although Monte Carlo is optimal for the observed data, TD produces estimates that generalize better to future experience.

# 6.3 Optimality of TD(0) 

## Batch Monte Carlo

Batch MC converges to the estimates that minimize the **training-set mean squared error (MSE)**.

It simply averages the observed returns for each state.

---

## Batch TD(0)

Batch TD converges to the **certainty-equivalence estimate**.

This is the value function that would be exactly correct for the **maximum-likelihood (ML) model** built from the observed data.

### Maximum-Likelihood Model

Transition probabilities are estimated as

$$
P(j|i)=\frac{\text{Number of transitions }i\rightarrow j}{\text{Total transitions leaving }i}
$$

Expected rewards are estimated by averaging the observed rewards for each transition.

---

## Certainty-Equivalence Estimate

After building the ML model, solve the Bellman equations on that model.

The resulting value function is called the **certainty-equivalence estimate** because it assumes the estimated model is the true environment.

Batch TD(0) converges to this solution.

---

## Why TD Often Performs Better

- Monte Carlo minimizes **training error**.
- TD estimates the underlying Markov process.
- TD therefore generalizes better to future experience.

---

## Computational Complexity

Let

$$
n=|S|
$$

where $|S|$ is the number of states.

| Method | Complexity |
|---------|------------|
| Build ML model | $O(n^2)$ memory |
| Solve Bellman equations | $O(n^3)$ computation |
| TD learning | $O(n)$ memory |

---

# 6.4 Sarsa: On-Policy TD Control 


Move from **prediction** to **control** using Temporal-Difference learning.

Instead of estimating the state-value function,

$$
V(s),
$$

Sarsa estimates the action-value function

$$
Q_\pi(s,a).
$$

---

## Sarsa Update Rule

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma Q(S_{t+1},A_{t+1})
-
Q(S_t,A_t)
\right]
$$

If the next state is terminal,

$$
Q(S_{t+1},A_{t+1})=0.
$$

---

## Why the Name "Sarsa"?

The update uses the five-tuple

$$
(S_t,\;A_t,\;R_{t+1},\;S_{t+1},\;A_{t+1})
$$

which gives the algorithm its name:

**State → Action → Reward → State → Action**

---

## On-Policy Learning

- The behavior policy and target policy are the same.
- The next action is selected using the current policy (e.g., $\varepsilon$-greedy).

---

## Convergence Conditions

Sarsa converges to the optimal policy if:

- Every state-action pair is visited infinitely often.
- The learning rate satisfies the stochastic approximation conditions.
- The policy becomes greedy in the limit (e.g., $\varepsilon = 1/t$).

---

## Windy Gridworld

## Environment

- Standard Gridworld with a **Start** and **Goal**.
- A vertical wind pushes the agent upward.
- Wind strength varies by column.
- Available actions:
  - Up
  - Down
  - Left
  - Right

---

## Reward

Each step receives

$$
R=-1
$$

until the goal is reached.

The objective is to minimize the number of steps.

---

## Learning Parameters

- $\varepsilon = 0.1$
- $\alpha = 0.5$
- Initial values:

$$
Q(s,a)=0
$$

for all state-action pairs.

---

## Results

- Sarsa gradually learns a shorter path.
- Around 8000 time steps, the greedy policy becomes optimal.
- Due to $\varepsilon$-greedy exploration, the average episode length remains about **17 steps**, although the optimal path requires only **15 steps**.

---

## Why Monte Carlo Is Difficult Here

Monte Carlo requires episodes to terminate.

Some policies may never reach the goal, causing episodes to continue indefinitely.

Sarsa updates after every transition, allowing it to quickly abandon poor policies.

---

# 6.5 Q-learning: Off-Policy TD Control

## Q-learning Update Rule

Q-learning updates the action-value function using

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\max_a Q(S_{t+1},a)
-
Q(S_t,A_t)
\right]
$$

Unlike Sarsa, the update uses the **maximum estimated value** of the next state instead of the action actually taken.

---

## Why Q-learning Is Off-Policy

- **Behavior policy:** generates experience (e.g., $\varepsilon$-greedy).
- **Target policy:** always assumes the greedy action.

Therefore,

- behavior policy $\neq$ target policy.

---

## Algorithm

1. Initialize $Q(s,a)$ arbitrarily.
2. Choose actions using the behavior policy.
3. Observe reward and next state.
4. Update

$$
Q(S,A)
\leftarrow
Q(S,A)
+
\alpha
\left[
R
+
\gamma
\max_a Q(S',a)
-
Q(S,A)
\right].
$$

5. Repeat until the terminal state.

---

## Backup Idea

The update backs up from

$$
(S,A)
$$

to

$$
\max_a Q(S',a),
$$

considering all possible actions in the next state.

---

# Example 6.6: Cliff Walking

## Environment

- Undiscounted episodic Gridworld.
- Four actions: Up, Down, Left, Right.
- Every move gives

$$
R=-1.
$$

- Entering the cliff gives

$$
R=-100
$$

and immediately returns the agent to the start state.

---

## Q-learning Behavior

Q-learning learns the **optimal policy**, which follows the shortest path along the edge of the cliff.

However, because actions are selected using an $\varepsilon$-greedy behavior policy, exploratory actions may cause the agent to fall into the cliff.

---

## Sarsa Behavior

Sarsa learns the value of the policy that is actually executed.

Since it accounts for exploration during learning, it prefers a **longer but safer path** away from the cliff.

---

## Why the Difference?

### Q-learning

Target:

$$
R+\gamma\max_a Q(S',a)
$$

Assumes the next action is always greedy.

---

### Sarsa

Target:

$$
R+\gamma Q(S',A')
$$

Uses the action actually selected by the current behavior policy.

---

## result

- Q-learning optimizes the greedy target policy.
- Sarsa optimizes the behavior policy.
- During learning, Sarsa often achieves better online performance because it considers exploration risk.

---

## Exercises

### Exercise 6.11

Q-learning is **off-policy** because the behavior policy generates experience while the target policy is greedy.

### Exercise 6.12

If action selection is completely greedy,

$$
A'=\arg\max_a Q(S',a),
$$

then Sarsa and Q-learning perform identical updates.

---


- Q-learning learns the optimal policy.
- Sarsa learns the policy being executed.
- Exploration affects Sarsa's updates but not Q-learning's targets.
- Sarsa is often safer during learning.
- Both converge to the optimal policy as

$$
\varepsilon \rightarrow 0.
$$

# 6.6 Expected Sarsa

## Motivation

Expected Sarsa replaces the random next action used in Sarsa with the **expected value** over all possible next actions according to the current policy.

---

## Update Rule

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
R_{t+1}
+
\gamma
\sum_a
\pi(a|S_{t+1})
Q(S_{t+1},a)
-
Q(S_t,A_t)
\right]
$$

---

## Comparison of TD Targets

| Algorithm | TD Target |
|------------|-----------|
| Sarsa | $R+\gamma Q(S',A')$ |
| Q-learning | $R+\gamma\max_aQ(S',a)$ |
| Expected Sarsa | $R+\gamma\sum_a\pi(a|S')Q(S',a)$ |

---

## Advantages

- Uses the expected next value instead of a sampled action.
- Eliminates variance caused by random action selection.
- Produces more stable learning.
- Usually performs slightly better than Sarsa.

---

## Computational Cost

Expected Sarsa requires evaluating all possible actions in the next state, making it slightly more expensive than Sarsa.

---

## Experimental Results

Expected Sarsa was compared with Sarsa and Q-learning on the Cliff Walking task.

Results showed that:

- Expected Sarsa retained Sarsa's advantage over Q-learning.
- It also outperformed Sarsa across a wide range of step sizes ($\alpha$).

---

## Deterministic Environments

In deterministic environments, randomness comes only from the behavior policy.

Because Expected Sarsa averages over all possible actions, it can safely use

$$
\alpha = 1
$$

without degrading long-term performance.

Sarsa usually requires a smaller $\alpha$ for stable learning.

---

## Relationship to Q-learning

Expected Sarsa computes

$$
\sum_a
\pi(a|S')Q(S',a).
$$

If the target policy is greedy,

$$
\pi(a|S')=
\begin{cases}
1,& a=\arg\max_aQ(S',a),\\
0,&\text{otherwise},
\end{cases}
$$

then

$$
\sum_a
\pi(a|S')Q(S',a)
=
\max_aQ(S',a),
$$

which is exactly the Q-learning target.

Thus, **Q-learning is a special case of Expected Sarsa.**

---

## Advantages

- Lower variance than Sarsa.
- Better empirical performance.
- Stable for large step sizes in deterministic tasks.
- Can operate as either on-policy or off-policy.
- Generalizes Q-learning.

---

# 6.7 Maximization Bias and Double Learning

## Maximization Bias

Many TD control algorithms use

$$
\max_a Q(s,a)
$$

to estimate

$$
\max_a q(s,a).
$$

Because action-value estimates are noisy, this introduces a **positive bias**:

$$
\max_a Q(s,a)
>
\max_a q(s,a).
$$

This phenomenon is called **maximization bias**.

---

## Why It Happens

Even if all true action values are identical,

$$
q(s,a)=0,
$$

their estimates fluctuate above and below zero.

Taking the maximum of these noisy estimates usually produces an overly optimistic value.

---

## Example 6.7

The MDP has two choices from state $A$:

- **Right:** immediate terminal reward

$$
0.
$$

- **Left:** transitions to state $B$, where many actions produce rewards sampled from

$$
N(-0.1,1).
$$

The expected return of choosing **Left** is

$$
-0.1,
$$

so **Right** is always optimal.

However, Q-learning often prefers **Left** because

$$
\max_aQ(B,a)
$$

becomes positively biased.

---

- Maximization over noisy estimates creates optimistic value estimates.
- Q-learning can choose suboptimal actions because of this bias.
- The bias becomes larger when more actions are available.

## Double Learning

Maximization bias occurs because the same estimates are used for

1. selecting the best action, and
2. evaluating its value.

Double Learning solves this by maintaining two independent value estimates:

$$
Q_1,\qquad Q_2.
$$

---

## Action Selection

Choose the greedy action using one estimator:

$$
A^*
=
\arg\max_a Q_1(a),
$$

but evaluate it using the other estimator:

$$
Q_2(A^*)
=
Q_2
\left(
\arg\max_aQ_1(a)
\right).
$$

The roles of $Q_1$ and $Q_2$ are alternated.

---

## Double Q-learning Update

When updating $Q_1$,

$$
Q_1(S,A)
\leftarrow
Q_1(S,A)
+
\alpha
\left[
R
+
\gamma
Q_2
\left(
S',
\arg\max_aQ_1(S',a)
\right)
-
Q_1(S,A)
\right].
$$

The update for $Q_2$ is symmetric.

---

## Behavior Policy

Action selection is typically based on

$$
Q_1+Q_2
$$

or their average using an $\varepsilon$-greedy policy.

---

## Advantages

- Removes most maximization bias.
- Produces more accurate value estimates.
- Improves stability.
- Requires roughly twice the memory.

---

# 6.8 Games, Afterstates, and Other Special Cases

## Afterstate

An **afterstate** is the state immediately **after the agent's action** but **before the environment or opponent responds**.

```
State → Action → Afterstate → Environment Response → Next State
```

Unlike a standard state-value function, an **afterstate value function** evaluates these intermediate states.

---

## Why Afterstates Are Useful

Afterstates are beneficial when the immediate effects of actions are completely known.

Examples:

- Tic-Tac-Toe
- Chess
- Queueing systems

Many different **state–action pairs** can lead to the **same afterstate**.

Instead of learning separate values for each pair,

$$
Q(s,a),
$$

the agent learns a single value

$$
V(\text{afterstate}),
$$

allowing knowledge to transfer immediately across equivalent situations.

---

## Advantages

- Reduces redundant learning.
- Shares experience across multiple state–action pairs.
- Speeds convergence.
- Produces more efficient learning.
