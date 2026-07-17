
Chapter 3 introduces **finite Markov decision processes (finite MDPs)** as the formal framework for reinforcement learning.

### Main Idea
Unlike bandit problems, where the agent only chooses among actions and receives immediate rewards, MDPs include:
- **states**
- **state-dependent actions**
- **transitions to future states**
- **delayed rewards**

| Concept | In Bandits | In MDPs |
|--------|------------|---------|
| Action | Present | Present |
| Reward | Immediate only | Immediate and future |
| Goal | Find the best action | Find the best action in each state |
| Value function | `q*(a)` | `q*(s,a)`, `v*(s)` |
| Effect of decision | Only on current reward | On reward and future states |
| Decision type | One-step | Sequential |

bandits can be formulized as one state MDPs.

So, actions affect not only the immediate reward but also future situations and future rewards.
## 3.1 The Agent–Environment Interface

### Main Idea
In reinforcement learning, an **agent** learns by interacting with an **environment**:

- the **agent** is the learner and decision maker
- the **environment** is everything outside the agent

At each discrete time step:

1. the agent observes the current state $S_t$
2. chooses an action $A_t$
3. the environment returns:
   - a reward $R_{t+1}$
   - the next state $S_{t+1}$

The agent's goal is to choose actions so as to maximize reward over time.

---

### Trajectory
The interaction generates a trajectory (or sequence) of the form:

$$
S_0, A_0, R_1, S_1, A_1, R_2, S_2, \dots
$$

Important convention:

- there is no $R_0$
- the first reward is $R_1$, because it is produced **after** taking $A_0$

---

### Finite MDP
A **finite MDP** is one in which:

- the state set $\mathcal{S}$ is finite
- the action set $\mathcal{A}$ is finite
- the reward set $\mathcal{R}$ is finite

Because these sets are finite, the next-state and reward variables have well-defined **discrete probability distributions**.

---

### MDP Dynamics
The environment is fully described by the **dynamics function**:

$$
p(s', r \mid s, a)
\overset{\text{def}}{=}
\Pr\{S_t=s',\ R_t=r \mid S_{t-1}=s,\ A_{t-1}=a\}
$$

This means:

- the agent is currently in state $s$
- it takes action $a$
- then the environment moves to state $s'$
- and gives reward $r$
- with probability $p(s',r \mid s,a)$

This is the most complete one-step description of the environment.

---

### Probability Condition
For each fixed state-action pair $(s,a)$, the probabilities over all possible next states and rewards must sum to 1:

$$
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} p(s', r \mid s, a) = 1
$$

So for each $(s,a)$, $p(\cdot,\cdot \mid s,a)$ defines a valid probability distribution.

---

### Markov Property
An MDP assumes the **Markov property**:

- the future depends only on the current state and action
- it does **not** depend on the full past history, once the current state is known

Informally:

> the state must contain all past information that is relevant for predicting the future.

So the restriction is really on the **state representation**:
if the state is rich enough, then the process is Markov.

---

### Why the Dynamics Function Is Important
The four-argument function

$$
p(s',r \mid s,a)
$$

completely characterizes the one-step behavior of the environment.

From it, we can compute other useful quantities such as:

- state-transition probabilities
- expected rewards
- expected rewards conditioned on the next state

That is why this function is the core model of an MDP.

---

### State Transition Probability
If we only care about the next state and ignore the reward, we sum over all possible rewards:

$$
p(s' \mid s, a) = \sum_{r \in \mathcal{R}} p(s', r \mid s, a)
$$

This gives the probability of moving to state $s'$ after taking action $a$ in state $s$.

Note: the book uses the same symbol $p$ for both $p(s',r \mid s,a)$ and $p(s' \mid s,a)$, which is a slight abuse of notation.

---

### Expected Reward for a State-Action Pair
The expected immediate reward for taking action $a$ in state $s$ is:

$$
r(s, a)
=
\mathbb{E}[R_t \mid S_{t-1}=s,\ A_{t-1}=a]
$$

Using the joint dynamics:

$$
r(s,a)=\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}} r\, p(s', r \mid s, a)
$$

So this is the average reward obtained from $(s,a)$.

---

### Expected Reward Conditioned on the Next State
Sometimes we also define the expected reward when the next state is known:

$$
r(s,a,s')
=
\mathbb{E}[R_t \mid S_{t-1}=s,\ A_{t-1}=a,\ S_t=s']
$$

This can be written as:

$$
r(s,a,s')
=
\sum_{r \in \mathcal{R}} r \,
\frac{p(s',r \mid s,a)}{p(s' \mid s,a)}
$$

whenever $p(s' \mid s,a) > 0$.

This is useful when the reward depends on both the action and the specific next state reached.


## 3.2 Goals and Rewards

### Main Idea
In reinforcement learning, the agent’s goal is defined through a special signal called the **reward**, sent from the environment to the agent.

At each time step, the reward is a scalar value:

$$
R_t \in \mathcal{R}
$$

The agent’s objective is to maximize the **total cumulative reward** it receives over time.

---

### Reward Hypothesis
The central idea of reinforcement learning is the **reward hypothesis**:

> All goals and purposes can be formulated as the maximization of the expected value of the cumulative sum of a scalar reward signal.

This is one of the most distinctive features of reinforcement learning.

---

### Long-Term Objective
The agent does **not** try to maximize only immediate reward.  
Instead, it seeks to maximize **long-term cumulative reward**.

This means that an action with low immediate reward may still be good if it leads to better future rewards.

---

### Example

#### Escaping a Maze
A robot may receive a reward of **-1 at each time step until escape**, encouraging it to escape as quickly as possible.

#### Playing Chess or Checkers
A natural reward scheme is:
- `+1` for winning
- `-1` for losing
- `0` for drawing and for all nonterminal position

The agent always learns to **maximize reward**.

Therefore, if we want the agent to achieve our intended goal, we must design the reward signal so that maximizing reward also leads to the desired behavior.

---

### Reward design
A reinforcement learning agent does **not** automatically understand what we really want.  
It only optimizes the reward signal we provide.

So, **reward design is crucial**:
- a good reward encourages the right behavior
- a poor reward may lead to unwanted behavior

In reinforcement learning, goals are expressed through rewards.  
The agent’s task is to maximize the expected cumulative reward.
## 3.3 Returns and Episodes

### Goal of the Agent
In reinforcement learning, the agent's objective is to maximize the **expected return**.  
That means the agent is not trying to maximize only the immediate reward, but the total future reward it can accumulate over time.

This is an important distinction:

- **Reward**: the immediate feedback received at a single time step
- **Return**: the total accumulated reward from a given time step onward

---

### Return
The **return**, denoted by $G_t$, is the total future reward following time step $t$.

For **episodic tasks**, the return is defined as the sum of rewards until the terminal state:

$$
G_t = R_{t+1} + R_{t+2} + \cdots + R_T
$$

where:

- $R_{t+1}, R_{t+2}, \dots$ are future rewards
- $T$ is the final time step of the episode

So, $G_t$ tells us how much total reward the agent will receive from time $t$ until the episode ends.

---

### Episodic Tasks
An **episodic task** is a task that naturally breaks into episodes.

Each episode:

- starts in some initial state,
- continues through a sequence of states, actions, and rewards,
- ends in a **terminal state**.

Examples include:

- playing a game of chess

A key point is that the episode length does **not** need to be fixed.  
The terminal time $T$ may vary from one episode to another.

After termination, the environment resets and a new episode begins.

---

### Continuing Tasks
Some tasks do not have a natural ending. These are called **continuing tasks**.

Examples include:

- controlling an industrial process,
- monitoring a communication network,
- operating a long-running robot.

In such tasks, the interaction goes on indefinitely, so the simple sum of all future rewards may become infinite.

For example, if the agent receives reward $+1$ forever, then:

$$
G_t = 1 + 1 + 1 + \cdots
$$

which diverges.

So for continuing tasks, we need a different definition of return.

---

### Discounted Return
To handle continuing tasks, reinforcement learning usually uses the **discounted return**:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots
= \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

where:

- $\gamma$ is the **discount rate**
- $0 \le \gamma \le 1$

Discounting reduces the weight of rewards that are farther in the future.

---

### Meaning of the Discount Factor
The value of $\gamma$ determines how much the agent cares about future rewards.

- If $\gamma = 0$, the agent is **myopic** and cares only about the immediate reward:
  $$
  G_t = R_{t+1}
  $$

- If $\gamma$ is close to 1, the agent places more value on future rewards and behaves more **far-sightedly**.

Thus, $\gamma$ controls the trade-off between immediate and delayed reward.

---

### Why Discounting is Useful
Discounting is important for both mathematical and conceptual reasons.

#### 1. It keeps return finite
If rewards are bounded and $\gamma < 1$, then the discounted return remains finite even for infinitely long tasks.

#### 2. It expresses preference for earlier rewards
In many situations, receiving a reward sooner is better than receiving the same reward later.

#### 3. It reflects uncertainty about the future
Rewards far in the future may be less certain, so it is reasonable to value them less.

---

### Constant Reward Example
Even though discounted return contains infinitely many terms, it can still be finite.

If the reward is always $+1$, then:

$$
G_t = 1 + \gamma + \gamma^2 + \gamma^3 + \cdots
$$

This is a geometric series, so:

$$
G_t = \sum_{k=0}^{\infty} \gamma^k = \frac{1}{1-\gamma}, \qquad \text{for } \gamma < 1
$$

This shows how discounting prevents infinite returns from blowing up.

---

### Recursive Form of Return
A very important relation is that the return can be written recursively as:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

This equation says:

- the return now
- equals the immediate reward
- plus the discounted return from the next time step onward

This recursive structure is fundamental in reinforcement learning and becomes the basis for many later methods.

---

### Return at the End of an Episode
To make the recursive formula valid even at termination, we define:

$$
G_T = 0
$$

That means once the terminal state is reached, there is no future reward left to collect.

So if the episode ends immediately after time step $t+1$, then:

$$
G_t = R_{t+1} + \gamma G_{t+1} = R_{t+1}
$$

because $G_{t+1} = 0$ after termination.

---

### Episodic vs. Continuing Tasks
The key difference is:

- **Episodic tasks** end in a terminal state, so return can be defined as a finite sum.
- **Continuing tasks** do not end naturally, so discounted return is usually used.

In episodic tasks, if termination is guaranteed, using $\gamma = 1$ can still be valid because the total number of rewards is finite.  
In continuing tasks, however, we usually require $\gamma < 1$ to ensure the return stays finite.

---
## 3.4 Unified Notation for Episodic and Continuing Tasks

This section introduces a single notation that can be used for both **episodic** and **continuing** reinforcement learning tasks.

### Why Is Unified Notation Needed?

In this book, some problems are episodic and some are continuing but weoften consider both.It is therefore useful to establish one notation that enables us to talk precisely about both cases simultaneously.

---

### Precise Notation for Episodic Tasks

In episodic tasks, interaction is divided into separate episodes.  
Each episode starts counting time again from zero, so a fully precise notation would be:

- $S_{t,i}$: state at time $t$ in episode $i$
- $A_{t,i}$: action at time $t$ in episode $i$
- $R_{t,i}$: reward at time $t$ in episode $i$
- $\pi_{t,i}$: policy at time $t$ in episode $i$
- $T_i$: terminal time of episode $i$

However, in practice we usually do not need to distinguish between different episodes.  
Most of the time, we are either discussing one particular episode or stating something that is true for all episodes.

Therefore, the episode index is usually omitted, and we simply write:

- $S_t$
- $A_t$
- $R_t$
- $T$

This is a mild abuse of notation, but it makes the presentation much simpler.

---

### Absorbing State

To unify episodic and continuing tasks, the end of an episode is modeled as entering a special **absorbing state**.

This absorbing state has two properties:

- it transitions only to itself,
- it always produces reward $0$.

So after termination, the reward sequence continues as:

$$
0, 0, 0, \dots
$$

This convention does **not** change the return.  
It only allows episodic tasks to be treated mathematically like continuing tasks.

---

### Explanation

Suppose an episode produces the rewards:

$$
+1,\ +1,\ +1
$$

and then terminates.

With the absorbing-state convention, the full reward sequence becomes:

$$
+1,\ +1,\ +1,\ 0,\ 0,\ 0,\dots
$$

Now the return is the same whether:

- we sum only up to the terminal time, or
- we sum over the full infinite sequence.

This remains true even when discounting is used, because all rewards after termination are zero.

---

### Unified Return Notation

Using this convention, the return can be written in the general infinite-horizon form:

$$
G_t \doteq \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

This form works for:

- **episodic tasks**, because rewards after termination are all zero,
- **continuing tasks**, because the interaction truly continues forever.

An equivalent finite-sum form is:

$$
G_t = \sum_{k=t+1}^{T} \gamma^{\,k-t-1} R_k
$$

where:

- $T$ is the terminal time for episodic tasks,
- $T = \infty$ for continuing tasks.

The exponent $k - t - 1$ means that the reward $R_k$ is received $k - t - 1$ steps after time $t$.

---

### Conditions on $T$ and $\gamma$

This notation allows:

- $T < \infty$ with $\gamma = 1$, which is valid for episodic tasks,
- $T = \infty$ with $\gamma < 1$, which is standard for continuing tasks.

But in general, we should **not** take both $T = \infty$ and $\gamma = 1$ at the same time, because the return may then diverge.

So:

- finite horizon allows undiscounted return,
- infinite horizon usually requires discounting.

---
## 3.5 Policies and Value Functions

This section introduces the concepts of **policy** and **value functions**, which are central to reinforcement learning.  
A policy describes how the agent behaves, and value functions describe how good states or actions are under that behavior.

---

### Policy

A **policy** is a mapping from states to probabilities of selecting each possible action:

$$
\pi(a \mid s) \doteq \Pr(A_t = a \mid S_t = s)
$$

That is, if the agent is in state $s$, then $\pi(a \mid s)$ is the probability of choosing action $a$.

For each state $s$, the policy defines a probability distribution over the available actions:

$$
\sum_{a \in \mathcal{A}(s)} \pi(a \mid s) = 1
$$

A policy may be:

- **deterministic**, meaning one action is always selected in a given state,
- **stochastic**, meaning actions are selected according to probabilities.

---

### State-Value Function

The **state-value function** of a policy $\pi$, denoted by $v_\pi(s)$, is the expected return when starting from state $s$ and then following policy $\pi$:

$$
v_\pi(s) \doteq \mathbb{E}_\pi [G_t \mid S_t = s]
$$

Using the definition of return:

$$
v_\pi(s)
=
\mathbb{E}_\pi
\left[
\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\;\middle|\;
S_t = s
\right]
$$

So $v_\pi(s)$ measures how good it is to be in state $s$ if the agent continues to behave according to policy $\pi$.

---

### Action-Value Function

The **action-value function** of a policy $\pi$, denoted by $q_\pi(s,a)$, is the expected return when starting from state $s$, taking action $a$, and then following policy $\pi$ thereafter:

$$
q_\pi(s,a) \doteq \mathbb{E}_\pi [G_t \mid S_t = s,\; A_t = a]
$$

Equivalently,

$$
q_\pi(s,a)
=
\mathbb{E}_\pi
\left[
\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
\;\middle|\;
S_t = s,\; A_t = a
\right]
$$

Thus, $q_\pi(s,a)$ measures how good it is to take action $a$ in state $s$ under policy $\pi$.

---

### Difference Between the Environment Model and Value Functions

It is important not to confuse the environment dynamics with value functions.

The quantity

$$
p(s', r \mid s, a)
$$

describes the **environment**. It gives the probability that after taking action $a$ in state $s$, the agent transitions to state $s'$ and receives reward $r$.

By contrast:

- $v_\pi(s)$ evaluates a **state** under policy $\pi$,
- $q_\pi(s,a)$ evaluates a **state-action pair** under policy $\pi$.

So:

- $p(s', r \mid s, a)$ tells us what the environment does,
- $v_\pi$ and $q_\pi$ tell us how good states and actions are.

Also note that $p$ is a property of the environment, whereas $v_\pi$ and $q_\pi$ depend on the policy.

---

### Expected Immediate Reward

If the current state is $S_t = s$ and the action is chosen according to policy $\pi$, then the expected next reward is:

$$
\mathbb{E}[R_{t+1} \mid S_t = s]
=
\sum_{a} \pi(a \mid s)
\sum_{s'} \sum_{r} p(s', r \mid s, a)\, r
$$

---

### Relation Between $v_\pi$ and $q_\pi$

The value of a state is the expected value of the actions chosen from that state:

$$
v_\pi(s)
=
\sum_{a \in \mathcal{A}(s)} \pi(a \mid s)\, q_\pi(s,a)
$$

This means that the state value is the average of the action values, weighted by the probabilities assigned by the policy.

---

### Relation Between $q_\pi$ and $v_\pi$

The action-value function can be expressed in terms of the state-value function:

$$
q_\pi(s,a)
=
\sum_{s' \in \mathcal{S}} \sum_{r \in \mathcal{R}}
p(s', r \mid s, a)
\left[
r + \gamma v_\pi(s')
\right]
$$

This equation says that the value of taking action $a$ in state $s$ equals:

- the immediate reward,
- plus the discounted value of the next state,
- averaged over all possible next states and rewards.

---

### Bellman Equation for $v_\pi$

A fundamental result is the **Bellman equation** for the state-value function:

$$
v_\pi(s)
=
\sum_{a} \pi(a \mid s)
\sum_{s', r} p(s', r \mid s, a)
\left[
r + \gamma v_\pi(s')
\right]
$$

This is the **Bellman expectation equation** for $v_\pi$.

It expresses a recursive relationship:  
the value of the current state depends on the immediate reward and the values of successor states.

This equation follows from the recursive form of return:

$$
G_t = R_{t+1} + \gamma G_{t+1}
$$

---

### Bellman Equation for $q_\pi$

There is also a Bellman equation for the action-value function:

$$
q_\pi(s,a)
=
\sum_{s',r} p(s',r \mid s,a)
\left[
r + \gamma \sum_{a'} \pi(a' \mid s') q_\pi(s',a')
\right]
$$

Using the relation between $v_\pi$ and $q_\pi$, this can also be written as:

$$
q_\pi(s,a)
=
\sum_{s',r} p(s',r \mid s,a)
\left[
r + \gamma v_\pi(s')
\right]
$$

---

### Terminal States

For episodic tasks, terminal states have value zero:

$$
v_\pi(\text{terminal}) = 0
$$

because after termination there are no future rewards to collect.

This convention helps keep the same equations valid for both episodic and continuing tasks.

---

### Estimating Value Functions from Experience

Value functions do not have to be known exactly from the environment model.  
They can also be estimated from experience.

For example:

- $v_\pi(s)$ can be estimated by averaging returns observed after visiting state $s$,
- $q_\pi(s,a)$ can be estimated by averaging returns observed after taking action $a$ in state $s$.

This idea is the basis of **Monte Carlo methods** and later **temporal-difference methods**.

When the number of states is very large, storing a separate value for every state may be impractical.  
In that case, **function approximation** is used.

---
## 3.6 Optimal Policies and Optimal Value Functions


### Optimal Policy

A policy $\pi$ is better than or equal to another policy $\pi'$ if

$$
v_\pi(s) \geq v_{\pi'}(s), \quad \forall s \in \mathcal{S}
$$

This means that for every state, policy $\pi$ gives an expected return at least as high as policy $\pi'$.

An optimal policy is a policy that is better than or equal to all other policies. It is denoted by

$$
\pi_*
$$

For finite MDPs, there is always at least one optimal policy.

The optimal policy is not necessarily unique but the optimal value is always unique.

---

### Optimal State-Value Function

The optimal state-value function is defined as

$$
v_*(s) \doteq \max_\pi v_\pi(s)
$$

This means that $v_*(s)$ is the maximum expected return achievable from state $s$ if the agent behaves optimally from that point onward.

In words:

> How good is state $s$ if I make the best possible decisions from now on?

---

### Optimal Action-Value Function

The optimal action-value function is defined as

$$
q_*(s,a) \doteq \max_\pi q_\pi(s,a)
$$

This means that $q_*(s,a)$ is the expected return if:

- the agent is currently in state $s$
- it takes action $a$ now
- and then follows the best possible policy afterward

> $q_*(s,a)$ does not mean that $a$ is necessarily the optimal action.  
> It means if you take action $a$ first, and then behave optimally afterward, this is the resulting expected return.

---

### Relationship Between $v_*$ and $q_*$

The optimal value of a state is the value of its best action:

$$
v_*(s) = \max_a q_*(s,a)
$$

So:

- $q_*(s,a)$ evaluates each possible action in state $s$
- $v_*(s)$ chooses the best one

This is one of the most important relationships in the section.

---

### Recursive Form of $q_*$

The optimal action-value function can be written as

$$
q_*(s,a) = \mathbb{E}\left[ R_{t+1} + \gamma v_*(S_{t+1}) \mid S_t = s, A_t = a \right]
$$

Using the environment dynamics, this becomes

$$
q_*(s,a) = \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma v_*(s')\right]
$$

This says:

- action $a$ produces an immediate reward
- then the environment moves to a next state $s'$
- from that next state onward, the agent behaves optimally

So the value of $(s,a)$ is the immediate reward plus the discounted optimal future value.

---

### Bellman Optimality Equation for $v_*$

The Bellman optimality equation for the optimal state-value function is

$$
v_*(s) = \max_a \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma v_*(s')\right]
$$

This equation says:

- consider every possible action in state $s$
- for each action, compute its expected immediate reward plus discounted future optimal value
- choose the action that gives the highest result

So the optimal value of a state is the return obtained by taking the best possible action, where "best" means best in the long run, not just best immediately.

---

### Bellman Optimality Equation for $q_*$

The Bellman optimality equation for the optimal action-value function is

$$
q_*(s,a) = \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma \max_{a'} q_*(s',a')\right]
$$

This means:

- take action $a$ now in state $s$
- receive reward $r$ and move to next state $s'$
- from $s'$ onward, choose the best possible action

So the value of $(s,a)$ is the immediate reward plus the discounted value of the best next action.

---

### Greedy Policy with Respect to $v_*$

Once $v_*$ is known, an optimal policy can be constructed by choosing actions greedily with respect to $v_*$.

That means choosing any action that achieves

$$
\max_a \sum_{s',r} p(s',r \mid s,a)\left[r + \gamma v_*(s')\right]
$$

Such a policy is optimal.

This works because $v_*$ already includes the value of the entire future. So even though the choice looks greedy at one step, it is globally optimal.

---

### Greedy Policy with Respect to $q_*$

If $q_*$ is known, then the optimal policy is even easier to obtain:

$$
\pi_*(s) \in \arg\max_a q_*(s,a)
$$

This means:

- choose any action with the largest optimal action-value
- if several actions tie, any of them may be chosen
- therefore multiple optimal policies may exist

This is one major reason why $q_*$ is so useful.

#### example 3.23:
q*(high, search)
= α [ r_search + γ max_a q*(high, a) ]
+ (1 - α) [ r_search + γ max_a q*(low, a) ]

q*(high, wait)
= r_wait + γ max_a q*(high, a)

q*(low, search)
= β [ r_search + γ max_a q*(low, a) ]
+ (1 - β) [ -3 + γ max_a q*(high, a) ]

q*(low, wait)
= r_wait + γ max_a q*(low, a)

q*(low, recharge)
= γ max_a q*(high, a)

---


## 3.7 Optimality and Approximation

This section explains that although optimal policies and optimal value functions are important theoretical goals, in practice agents rarely achieve exact optimality. For most problems of interest, computing an optimal policy is too expensive in terms of time and computation.

Even if the agent has a complete and accurate model of the environment, solving the Bellman optimality equation exactly is usually infeasible. The main reason is that real problems often involve very large state spaces, limited computation per time step, and limited memory.

Memory is also a major constraint. In small finite problems, value functions or policies can be stored in tables, with one entry for each state or state-action pair. This is called the tabular case, and methods based on this representation are called tabular methods.

In many practical tasks, however, the number of states is far too large for tabular storage. In those cases, value functions, policies, or models must be approximated using more compact parameterized representations. This is the idea of function approximation.

reinforcement learning naturally works with approximation. The agent is usually forced to settle for solutions that are only approximately optimal.

At the same time, reinforcement learning has an important advantage: not all states matter equally. Some states are encountered very frequently, while others are extremely rare. Making poor decisions in rare states may have little effect on total reward, while making good decisions in common states can lead to strong overall performance.

The example of TD-Gammon illustrates this idea. The program may make poor decisions in many board positions, but if those positions almost never occur in serious play, its overall performance can still be excellent.

Because reinforcement learning is online, the agent can focus more learning effort on frequently encountered states and less effort on rare ones. This ability to concentrate learning where it matters most is one of the main features that distinguishes reinforcement learning from other approximate methods for solving MDPs.