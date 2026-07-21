# Chapter 4: Dynamic Programming

## What is Dynamic Programming (DP)?
- **Dynamic Programming (DP)** is a collection of algorithms used to compute **optimal policies** when the environment is **fully known** and modeled as a **Markov Decision Process (MDP)**.
- In Reinforcement Learning, classical DP is mainly of **theoretical importance** because it:
  - requires a **perfect model** of the environment,
  - has **high computational cost**.
- Most RL algorithms can be viewed as approximating the results of DP:
  - **without** knowing the model,
  - and **with much less computation**.

## Assumptions
From this chap onward, the environment is assumed to be a **finite MDP**:
- Finite state set $S$
- Finite action set $A$
- Finite reward set $R$
- Known transition dynamics:
  $$p(s', r \mid s, a)$$
- For episodic tasks, $S^+$ includes the terminal state.

> **Note:** DP can also be applied to continuous state/action spaces, but exact solutions are rarely possible. A common approach is to discretize (quantize) the spaces and then apply finite-state DP methods.

## DP Idea
- The central idea of DP (and RL in general) is to use **value functions** to evaluate states or state-action pairs.
- Once the **optimal value functions** $v_*$ or $q_*$ are computed, the **optimal policy** can be obtained by choosing the action with the highest value.

## Bellman Optimality Equations
The optimal value functions satisfy:

$$v_*(s) = \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t=s, A_t=a]$$

$$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') \mid S_t=s, A_t=a]$$

These equations express the **Principle of Optimality**:
> The value of a state (or state-action pair) equals the **immediate reward** plus the **best possible discounted future value**.

## How DP Works
- DP algorithms are created by converting the Bellman equations into **iterative update rules**.
- Starting from an initial estimate of the value function, the values are repeatedly updated until they **converge** to the optimal values.
# 4.1 Policy Evaluation (Prediction)

## Goal
- **Policy Evaluation** computes the state-value function $v_\pi$ for a given policy $\pi$.
- This is also called the **prediction problem**, since it predicts the expected return when following $\pi$.

## Bellman Expectation Equation
For every state $s \in S$:

$$
v_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t=s]
$$

Using the recursive definition of return:

$$
v_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma v_\pi(S_{t+1}) \mid S_t=s]
$$

Expanding over actions and transitions:

$$
v_\pi(s) = \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a) \left[ r + \gamma v_\pi(s') \right]
$$

- If $\gamma < 1$ (or termination is guaranteed), then $v_\pi$ exists and is unique.
- With a known model, this forms a system of $|S|$ linear equations.

## Iterative Policy Evaluation
- Instead of solving the equations directly, DP repeatedly updates the value function.
- Initialize $v_0$ arbitrarily (terminal states are initialized to $0$).
- Update using the Bellman expectation backup:

$$
v_{k+1}(s) = \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a) \left[ r + \gamma v_k(s') \right]
$$
- $v_{k+1}(s)$ is the new value of s
- Repeating this update makes $v_k$ converge to $v_\pi$.
- The optimal value function is the **fixed point** of the Bellman expectation equation.

## Expected Updates
- DP performs **expected updates**, averaging over **all possible next states** using the model.
- Unlike sample-based methods, DP does not learn from a single observed transition.
- One complete pass over all states is called a **sweep**.

## Implementation
- Two common implementations:
  - **Two-array:** keep old and new values separately.
  - **In-place:** overwrite values immediately.
- In-place updates usually converge faster because newly updated values are reused immediately.
- The update order may affect the convergence speed.

## Stopping Criterion
Stop when the largest value change after a sweep is sufficiently small:

$$
\max_{s \in S} |v_{k+1}(s) - v_k(s)| < \theta
$$

where $\theta > 0$ is a small threshold.
## Iterative Policy Evaluation Algorithm

**Input:**
- Policy $\pi$
- Threshold $\theta > 0$

**Initialize:**
- Initialize $V(s)$ arbitrarily for all states.
- Set $V(\text{terminal}) = 0$.

**Repeat**
1. Set $\Delta \leftarrow 0$.
2. For each state $s \in S$:
   - Store the old value:
     $$
     v \leftarrow V(s)
     $$
   - Update using the Bellman expectation equation:
     $$
     V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s',r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right]
     $$
   - Compute the change:
     $$
     \Delta \leftarrow \max(\Delta, \left| v - V(s) \right|)
     $$

**Until**

$$
\Delta < \theta
$$

**Output:** Approximate value function $V \approx v_\pi$.

# 4.2 Policy Improvement


## Action-Value for Policy Improvement
To check whether changing the action at state $s$ is beneficial, compute:

$$
q_\pi(s,a)=\mathbb{E}[R_{t+1}+\gamma v_\pi(S_{t+1}) \mid S_t=s,\,A_t=a]
$$

- If

$$
q_\pi(s,a) > v_\pi(s),
$$

then choosing action $a$ is better than following the current policy at state $s$.

## Policy Improvement Theorem
- If, for every state,

$$
q_\pi(s,\pi'(s)) \ge v_\pi(s),
$$

then the new policy $\pi'$ is **at least as good as** the old policy $\pi$.
- If the inequality is strict for at least one state, then $\pi'$ is **strictly better**.

## Greedy Policy
A new policy is obtained by choosing the action with the highest action-value:

$$
\pi'(s)=\arg\max_a q_\pi(s,a)
$$

- This is called the **greedy policy** with respect to $v_\pi$.
- A greedy policy is guaranteed to be **no worse** than the original policy.

## Optimality
- If the greedy policy does **not** improve the current policy, then the current policy is already **optimal**.
- In this case,

$$
v_\pi = v^*
$$

and the Bellman Optimality Equation is satisfied.

## Stochastic Policies
- The same ideas apply to **stochastic policies**.
- If multiple actions have the same maximum value, the probability can be divided among them.
- Actions that are **not** optimal must receive **zero probability**.


# 4.3 Policy Iteration

## The Policy Iteration Loop
- Once a policy $\pi$ has been evaluated to find $v_\pi$, we can perform policy improvement to obtain a better policy $\pi'$.
- This cycle of evaluation and improvement can be repeated, creating a sequence of monotonically improving policies and value functions:

$$
\pi_0 \xrightarrow{\text{Evaluation}} v_{\pi_0} \xrightarrow{\text{Improvement}} \pi_1 \xrightarrow{\text{Evaluation}} v_{\pi_1} \xrightarrow{\text{Improvement}} \pi_2 \dots \xrightarrow{\text{Improvement}} \pi_* \xrightarrow{\text{Evaluation}} v_*
$$

- Since a finite MDP has only a finite number of deterministic policies, this process is guaranteed to converge to an optimal policy $\pi_*$ and optimal value function $v_*$ in a finite number of iterations.
- **Speedup Tip:** When evaluating a new policy, starting the iterative evaluation loop with the value function of the *previous* policy ($V$) drastically reduces the number of iterations required to converge.

---

## Policy Iteration Algorithm

**1. Initialization**
- Initialize $V(s) \in \mathbb{R}$ and $\pi(s) \in A(s)$ arbitrarily for all $s \in S$.
- Set $V(\text{terminal}) = 0$.

**2. Policy Evaluation**
- **Repeat**
  - $\Delta \leftarrow 0$
  - For each $s \in S$:
    - $v \leftarrow V(s)$
    - $V(s) \leftarrow \sum_{s', r} p(s', r \mid s, \pi(s)) \left[ r + \gamma V(s') \right]$
    - $\Delta \leftarrow \max(\Delta, |v - V(s)|)$
- **Until** $\Delta < \theta$ (a small threshold $\theta > 0$)

**3. Policy Improvement**
- $\text{policy-stable} \leftarrow \text{true}$
- For each $s \in S$:
  - $\text{old-action} \leftarrow \pi(s)$
  - $\pi(s) \leftarrow \arg\max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V(s') \right]$
  - If $\text{old-action} \neq \pi(s)$, then $\text{policy-stable} \leftarrow \text{false}$
- If $\text{policy-stable}$ is $\text{true}$, then stop and return $V \approx v_*$ and $\pi \approx \pi_*$; else go to Step 2.

---
# 4.4 Value Iteration

- A drawback of **Policy Iteration** is that each iteration requires **Policy Evaluation**, which may need many sweeps to converge.
- **Value Iteration** solves this by stopping policy evaluation after **only one sweep**.
- It combines **Policy Evaluation** and **Policy Improvement** into a single update.

The update rule is obtained directly from the **Bellman Optimality Equation**:

$$
v_{k+1}(s)
=
\max_a
\mathbb{E}
\left[
R_{t+1}
+
\gamma v_k(S_{t+1})
\mid
S_t=s,A_t=a
\right]
$$

or equivalently,

$$
v_{k+1}(s)
=
\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma v_k(s')
\right].
$$

- Unlike **Policy Evaluation**, the update takes the **maximum over all actions**.
- Starting from any initial value function $v_0$, the sequence $\{v_k\}$ converges to the optimal value function $v_*$.

---

## Value Iteration Algorithm

**Initialization**
- Initialize $V(s)$ arbitrarily.
- Set $V(\text{terminal})=0$.

**Repeat**
- $\Delta \leftarrow 0$
- For each state $s$:
  - Save old value: $v \leftarrow V(s)$
  - Update:

$$
V(s)
=
\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V(s')
\right]
$$

  - Update

$$
\Delta
=
\max(\Delta,\;|v-V(s)|)
$$

- Stop when

$$
\Delta < \theta
$$

where $\theta$ is a small threshold.

Finally extract the optimal policy:

$$
\pi_*(s)
=
\arg\max_a
\sum_{s',r}
p(s',r|s,a)
\left[
r+\gamma V(s')
\right].
$$

---

# 4.5 Asynchronous Dynamic Programming


- A major drawback of classical Dynamic Programming (DP) methods is that they require **full sweeps over the entire state space**.
- When the number of states is very large, even a single sweep can be computationally infeasible.
- **Asynchronous Dynamic Programming** removes this restriction by updating states in **any order**, instead of sweeping through all states systematically.

---

## Asynchronous Updates

- Updates are performed **in-place**, meaning the new value immediately replaces the old one.
- States can be updated:
  - in any order, uusing whatever values of other states happen to be available.
  - and some states may be updated many times before others are updated once.

- However, to guarantee convergence:
  - **every state must continue to be updated indefinitely**.
  - No state can be permanently ignored.


---

## Asynchronous Value Iteration

Instead of updating every state in one sweep, only **one state** is updated at each step:

$$
V(s_k) \leftarrow
\max_a \sum_{s',r}
p(s',r|s_k,a)
\left[r+\gamma V(s')\right]
$$

where only the selected state $s_k$ is updated.

### Convergence
If

$$
0 \le \gamma < 1
$$

and **every state is selected infinitely often** (even in a random order), the value function converges to the optimal value:

$$
V \rightarrow v^*
$$

---

## Advantages

- No need to wait for a complete sweep before making progress.
- More flexible than synchronous DP.
- States that are more important can be updated more frequently.
- Computational effort can be focused on relevant regions of the state space.
- Different DP updates (policy evaluation and value iteration) can even be mixed together.

---

## Real-Time Learning

A major advantage is that planning and interaction can occur simultaneously.

- While the agent interacts with the environment,
- DP updates can be applied to the states the agent actually visits.
- The updated value estimates immediately improve future decisions.

This allows computation to focus on the most relevant states instead of wasting time updating rarely visited ones.

---

# 4.6 Generalized Policy Iteration (GPI)

## Idea
- **Generalized Policy Iteration (GPI)** is the general framework in which **Policy Evaluation** and **Policy Improvement** interact.
- Unlike Policy Iteration, these two processes do **not** need to be completed separately; they can be interleaved in any order or frequency.
- Examples:
  - **Policy Iteration:** full evaluation + full improvement.
  - **Value Iteration:** one evaluation sweep + improvement.
  - **Asynchronous DP:** evaluation and improvement are mixed even more frequently.

---

## Two Main Processes

### 1. Policy Evaluation
- Updates the value function so that it becomes consistent with the current policy:

$$
V \rightarrow v_\pi
$$

### 2. Policy Improvement
- Updates the policy to become greedy with respect to the current value function:

$$
\pi \rightarrow \text{greedy}(V)
$$

These two processes are repeated until both stop changing.

---

## Convergence

When both processes stabilize:

- The value function is consistent with the current policy.
- The policy is greedy with respect to the current value function.

Therefore,

$$
V = v^*, \qquad \pi = \pi^*
$$

and the Bellman Optimality Equation is satisfied.

---

## Competition and Cooperation

The two processes both **compete** and **cooperate**:

- **Competition:** Improving the policy makes the old value function inaccurate, while updating the value function may make the current policy no longer greedy.
- **Cooperation:** Repeating both processes gradually drives the system toward the optimal policy and optimal value function.

---

# 4.7 Efficiency of Dynamic Programming

## Main Idea
- Although **Dynamic Programming (DP)** is not practical for extremely large state spaces, it is one of the **most efficient exact methods** for solving finite MDPs.

---

## Computational Efficiency

- Suppose:
  - $n$ = number of states
  - $k$ = number of actions

- The worst-case running time of DP algorithms is **polynomial** in $n$ and $k$.

- Even though the total number of deterministic policies is

$$
k^n
$$

DP does **not** search all possible policies.

- Therefore, DP is **exponentially faster** than brute-force policy search, which would need to examine all $k^n$ policies.

---

## Comparison with Other Methods

### Direct Policy Search
- Requires checking every possible policy.
- Time complexity grows exponentially.
- Much slower than DP.

### Linear Programming
- Can also solve MDPs.
- May have stronger theoretical convergence guarantees.
- However, becomes impractical for much smaller state spaces (roughly **100× smaller**) than DP.

---

## Curse of Dimensionality

- The main limitation of DP is the **curse of dimensionality**:
  - As the number of state variables increases,
  - the number of states often grows exponentially.

- This is **a property of the problem itself**, not a weakness of DP.

- Compared to competing methods, DP actually handles large state spaces **better**.

---

## Practical Performance

- Modern computers can solve MDPs with **millions of states** using DP.

- The two most common DP algorithms are:
  - **Policy Iteration**
  - **Value Iteration**

- There is **no universal winner** between them.
- In practice, both usually converge much faster than their theoretical worst-case bounds, especially with a good initial policy or value function.

---

## Asynchronous DP

For very large state spaces, **Asynchronous DP** is often preferred because:

- It updates only selected states instead of sweeping over the entire state space.
- Requires much less memory.
- Focuses computation on states that are important for the optimal policy.
- Can find good (or optimal) policies much faster than synchronous DP when only a small subset of states is relevant.

