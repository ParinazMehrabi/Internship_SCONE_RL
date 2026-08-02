# Chapter 10: On-policy Control with Approximation

## Main Idea

This chapter extends reinforcement learning control methods by using function approximation for action-value functions.

Instead of storing:
$q(s,a)$

we approximate:
$$\hat{q}(s,a,w) \approx q^*(s,a)$$

where $w$ is a finite-dimensional weight vector.

---

## Semi-gradient Sarsa

Semi-gradient Sarsa extends semi-gradient TD(0) from state values to action values.

TD error:
$$\delta = R + \gamma \hat{q}(S', A', w) - \hat{q}(S, A, w)$$

Weight update:
$$w \leftarrow w + \alpha \delta \nabla_w \hat{q}(S, A, w)$$

Only the current estimate is differentiated, therefore it is called semi-gradient.

---

## On-policy Control

The chapter focuses on on-policy methods. The same policy is used for:
- generating experience
- learning the optimal policy

Action selection uses:
$\epsilon$-greedy

---

## Episodic Case

For episodic tasks:
- Function approximation can be directly extended from state values to action values.
- Linear Sarsa and n-step Sarsa are introduced.
- Mountain Car is used as an example.

---

## Continuing Case

For continuing environments, discounting becomes problematic.

Using:
$\gamma < 1$
is not suitable because it artificially reduces future rewards.

Using:
$\gamma = 1$
may cause infinite returns.

Therefore, a new formulation is introduced.

---

## Average Reward Formulation

Instead of maximizing discounted return, we maximize long-term average reward:
$r(\pi)$

The objective becomes:
Maximize average reward per time step.

---

## Differential Value Functions

In average-reward problems, ordinary value functions are replaced by:

**Differential value functions**

They measure how much better or worse a state is compared with the long-term average reward.


# 10.1 Episodic Semi-gradient Control


This chapter extends semi-gradient prediction from **state values** to **action values**.

Previously:

$$
S \rightarrow U
$$

Now:

$$
(S, A) \rightarrow U
$$

Approximate action-value function:

$$
\hat{q}(s,a,w) \approx q_\pi(s,a)
$$

where \(w\) is the parameter (weight) vector.

---

# General Gradient Descent Update

The general update rule is

$$
w_{t+1}
=
w_t
+
\alpha
\left[
U_t-\hat q(S_t,A_t,w_t)
\right]
\nabla \hat q(S_t,A_t,w_t)
$$

where

- $U_t$: target value
- $\alpha$: learning rate
- $\nabla \hat q$: gradient with respect to the weights

---

# One-step Semi-gradient Sarsa

The TD error is

$$
\delta
=
R_{t+1}
+
\gamma
\hat q(S_{t+1},A_{t+1},w)
-
\hat q(S_t,A_t,w)
$$

Weight update:

$$
w
\leftarrow
w
+
\alpha
\delta
\nabla
\hat q(S_t,A_t,w)
$$

---

# Control

Policy evaluation is combined with policy improvement.

Greedy action:

$$
A^*
=
\arg\max_a
\hat q(S,a,w)
$$

Behavior policy:

- ε-greedy

---

# Algorithm: Episodic Semi-gradient Sarsa for Estimating

Goal:

$$
\hat q \approx q^*
$$

## Input

A differentiable action-value function:

$$
\hat q : \mathcal{S} \times \mathcal{A} \times \mathbb{R}^d \rightarrow \mathbb{R}
$$

Algorithm parameters:

- Learning rate:

$$
\alpha > 0
$$

- Exploration parameter:

$$
\epsilon > 0
$$

Initialize the weight vector

$$
w \in \mathbb{R}^d
$$

arbitrarily (e.g.,$w=0$).

---

## Algorithm

```text
Loop for each episode

    Initialize state S
    Choose action A using ε-greedy policy

    Loop for each step

        Take action A

        Observe reward R and next state S'

        If S' is terminal

            w ← w + α [R − q̂(S,A,w)] ∇q̂(S,A,w)

            Break

        Choose next action A' using ε-greedy policy

        w ← w + α [R + γq̂(S',A',w) − q̂(S,A,w)] ∇q̂(S,A,w)

        S ← S'

        A ← A'
```

---

## Terminal Update

When the next state is terminal:

$$
w
\leftarrow
w
+
\alpha
\left[
R-\hat q(S,A,w)
\right]
\nabla
\hat q(S,A,w)
$$

---

## Non-terminal Update

Otherwise compute the TD error:

$$
\delta
=
R
+
\gamma
\hat q(S',A',w)
-
\hat q(S,A,w)
$$

Then update the weights:

$$
w
\leftarrow
w
+
\alpha
\delta
\nabla
\hat q(S,A,w)
$$

---

## Action Selection

Choose the next action using an ε-greedy policy:

$$
A'
=
\epsilon\text{-greedy}
\left(
\hat q(S',\cdot,w)
\right)
$$

The greedy action is

$$
A^*
=
\arg\max_a
\hat q(S',a,w)
$$

while ε-greedy occasionally selects a random action for exploration.

# 10.2 Semi-gradient n-step Sarsa


The one-step Semi-gradient Sarsa algorithm is extended by replacing the
one-step TD target with an n-step return.

---

# n-step Return

For

$$
t+n<T
$$

the n-step return is

$$
G_{t:t+n}
=
R_{t+1}
+
\gamma R_{t+2}
+
\cdots
+
\gamma^{n-1}R_{t+n}
+
\gamma^n
\hat q(S_{t+n},A_{t+n},w)
$$

If

$$
t+n\ge T
$$

then

$$
G_{t:t+n}=G_t
$$

(Monte Carlo return).

---

# Weight Update

$$
w
\leftarrow
w
+
\alpha
\left[
G_{t:t+n}
-
\hat q(S_t,A_t,w)
\right]
\nabla
\hat q(S_t,A_t,w)
$$

---

# Algorithm

1. Initialize weights.
2. Select the initial action using ε-greedy.
3. Store states, actions, and rewards.
4. Compute

$$
\tau=t-n+1
$$

5. If

$$
\tau\ge0
$$

compute the n-step return.

6. Update the weights.

7. Continue until

$$
\tau=T-1
$$

---

One-step target:

$$
R+\gamma \hat q(S',A')
$$

n-step target:

$$
R_1+\gamma R_2+\cdots+\gamma^{n-1}R_n+\gamma^n\hat q
$$

---