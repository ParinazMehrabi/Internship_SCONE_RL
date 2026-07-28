# Chapter 7 – n-step Bootstrapping

## Main Idea

n-step TD methods unify:

- Monte Carlo (MC)
- One-step Temporal Difference (TD)

They form a spectrum:

```
One-step TD  <------ n-step TD ------>  Monte Carlo
      n=1                         n=Episode Length
```

Intermediate values of **n** often produce the best performance.

---

## Why n-step Methods?

One-step TD forces the same time interval for:

- Action updates
- Bootstrapping

In many applications:

- Actions should be updated frequently.
- Bootstrapping should cover a longer period with meaningful state changes.

n-step methods separate these two ideas by allowing bootstrapping over multiple steps.

---

## Relationship with Eligibility Traces

n-step methods are the foundation of **Eligibility Traces** (Chapter 12).

This chapter studies n-step bootstrapping independently before introducing eligibility traces.

---

## Topics Covered

1. Prediction
   - Estimate the state-value function $v_\pi$ for a fixed policy.

2. Control
   - Extend n-step ideas to action values and control algorithms.

---
# 7.1 n-step TD Prediction

n-step TD methods generalize both **Monte Carlo (MC)** and **One-step TD**.

- **n = 1** → One-step TD
- **n = Episode Length (∞)** → Monte Carlo
- Intermediate values of **n** combine the advantages of both methods.

---

# n-step Return

## Monte Carlo Target (Full Return)

$$
G_t
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2R_{t+3}
+\cdots
+
\gamma^{T-t-1}R_T
$$

Uses **all future rewards** until the end of the episode.

---

## One-step Return

$$
G_{t:t+1}
=
R_{t+1}
+
\gamma V_t(S_{t+1})
$$

Uses:

- One real reward
- Bootstrapped estimate of the next state

---

## Two-step Return

$$
G_{t:t+2}
=
R_{t+1}
+
\gamma R_{t+2}
+
\gamma^2V_{t+1}(S_{t+2})
$$

Uses:

- Two real rewards
- Value estimate after two steps

---

## General n-step Return

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
\gamma^nV_{t+n-1}(S_{t+n})
$$

If the episode terminates before n steps,

$$
G_{t:t+n}=G_t
$$

(the ordinary Monte Carlo return).

---

# n-step TD Update Rule

After enough future rewards are observed:

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\left(
G_{t:t+n}
-
V(S_t)
\right)
$$

Same TD update rule, but with an n-step target.

---

# Important Algorithm Ideas

### Initialization

- Initialize all state values $V(s)$.
- Choose:
  - learning rate $\alpha$
  - step size $n$

---

### During Each Episode

For every time step:

1. Take action using policy \(\pi\).
2. Observe reward and next state.
3. Store recent states and rewards.
4. Wait until n future rewards become available.
5. Compute n-step return.
6. Update the state that occurred n steps earlier.

---

### Beginning of Episode

No updates are made during the first

$$
n-1
$$

steps because future rewards are not yet available.

---

### End of Episode

Extra updates are performed after termination so every visited state is updated exactly once.

---

### Memory Optimization

Only the latest

$$
n+1
$$

states and rewards need to be stored.

A circular buffer (mod $n+1$) is sufficient.

Memory complexity:

$$
O(n)
$$

---

# Error Reduction Property

Expected n-step returns are guaranteed to reduce the prediction error.

$$
\max_s
\left|
E_\pi[G_{t:t+n}|S_t=s]
-
v_\pi(s)
\right|
\le
\gamma^n
\max_s
\left|
V(s)-v_\pi(s)
\right|
$$

Consequences:

- Expected target is closer to the true value.
- n-step TD converges under standard conditions.
- Larger n reduces worst-case prediction error.

---

# Advantages of n-step TD

Bridges TD and Monte Carlo

Flexible choice of bootstrapping depth

Better bias-variance trade-off

Uses more real rewards than one-step TD

Converges under standard assumptions

Often performs better than both TD(0) and Monte Carlo

---
# 7.2 n-step Sarsa

n-step Sarsa extends **one-step Sarsa** to use **n-step returns** for control.

- **n = 1** → One-step Sarsa (Sarsa(0))
- **n = Episode Length** → Monte Carlo Control
- Intermediate values of **n** balance bootstrapping and sampled returns.

Unlike n-step TD Prediction, this method estimates **action values** $Q(s,a)$ instead of state values $V(s)$.

The behavior policy is typically **$\varepsilon$-greedy** with respect to $Q$.

---

# n-step Return for Sarsa

The target is

$$
G_{t:t+n}
=
R_{t+1}
+\gamma R_{t+2}
+\cdots
+\gamma^{n-1}R_{t+n}
+\gamma^nQ(S_{t+n},A_{t+n})
$$

for

$$
t+n<T
$$

If the episode terminates before $n$ steps,

$$
G_{t:t+n}=G_t
$$

(the ordinary Monte Carlo return).

---

# Update Rule

The action-value update is

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\left[
G_{t:t+n}
-
Q(S_t,A_t)
\right]
$$

Only the state-action pair $(S_t,A_t)$ is updated.

---

# Algorithm

## Initialization

- Initialize all $Q(s,a)$ arbitrarily.
- Initialize policy $\pi$ as $\varepsilon$-greedy with respect to $Q$ (or use a fixed policy).
- Choose:
  - learning rate $\alpha$
  - exploration parameter $\varepsilon$
  - step size $n$

---

## During Each Episode

1. Observe initial state $S_0$.
2. Select action $A_0$ using $\pi$.
3. Execute action.
4. Observe reward $R_{t+1}$ and next state $S_{t+1}$.
5. If not terminal:
   - Select $A_{t+1}$ using the current policy.
6. Compute

$$
\tau=t-n+1
$$

7. If $\tau\ge0$:
   - Compute the n-step return.
   - Update $Q(S_\tau,A_\tau)$.
8. If policy learning is enabled:
   - Update policy to remain $\varepsilon$-greedy.

---

# Memory Requirement

Only the latest

$$
n+1
$$

states, actions and rewards must be stored.

A circular buffer (mod $n+1$) is sufficient.

Memory complexity:

$$
O(n)
$$

---

# Expected Sarsa Extension

Instead of bootstrapping from one sampled action,

Expected Sarsa bootstraps from the **expected value** under the target policy.

The n-step return becomes

$$
G_{t:t+n}
=
R_{t+1}
+\cdots
+\gamma^{n-1}R_{t+n}
+\gamma^n\bar{V}(S_{t+n})
$$

where

$$
\bar{V}(s)
=
\sum_a
\pi(a|s)Q(s,a)
$$

If the state is terminal,

$$
\bar{V}(s)=0
$$

---

# Backup Diagrams

Spectrum of methods:

```text
Sarsa(0)
      ↓
2-step Sarsa
      ↓
3-step Sarsa
      ↓
...
      ↓
n-step Sarsa
      ↓
Monte Carlo Control
```

Expected Sarsa differs only in the final backup:

- Sarsa:
  - Uses one sampled action.

- Expected Sarsa:
  - Uses the expectation over all possible actions.

---

# Advantages

- Generalizes one-step Sarsa.
- Faster reward propagation.
- Better sample efficiency.
- Balances bias and variance.
- Naturally extends TD control to n-step updates.
- Works with on-policy learning.

---

# Comparison

| Method | Bootstrap From | Uses Sampled Action? |
|---------|----------------|----------------------|
| Sarsa(0) | $Q(S_{t+1},A_{t+1})$ | Yes |
| n-step Sarsa | $Q(S_{t+n},A_{t+n})$ | Yes |
| Expected Sarsa | $\sum_a\pi(a|s)Q(s,a)$ | No |

---

# 7.3 n-step Off-policy Learning

Off-policy learning estimates the value of a **target policy** $\pi$ while following a different **behavior policy** $b$.

Typically:

- Target policy ($\pi$): Greedy
- Behavior policy ($b$): Exploratory (e.g., $\varepsilon$-greedy)

To correct for the difference between the two policies, **Importance Sampling** is used.

---

# Importance Sampling Ratio

The importance sampling ratio over multiple steps is

$$
\rho_{t:h}
=
\prod_{k=t}^{\min(h,T-1)}
\frac{\pi(A_k|S_k)}
{b(A_k|S_k)}
$$

Properties:

- $\rho=1$ if $\pi=b$
- $\rho=0$ if any action is impossible under $\pi$
- Larger ratios give more weight to trajectories that are more likely under $\pi$

---

# Off-policy n-step TD

The update becomes

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
\rho_{t:t+n-1}
\left(
G_{t:t+n}
-
V(S_t)
\right)
$$

Compared to on-policy n-step TD, the TD error is weighted by the importance sampling ratio.

---

# Off-policy n-step Sarsa

The update rule is

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
\rho_{t+1:t+n}
\left(
G_{t:t+n}
-
Q(S_t,A_t)
\right)
$$

Notice that the ratio starts at $t+1$ because the selected action $A_t$ is already fixed and does not need correction.

---

# Algorithm

1. Initialize $Q(s,a)$.
2. Initialize target policy $\pi$ (usually greedy).
3. Generate experience using behavior policy $b$.
4. Store states, actions, and rewards.
5. Compute

$$
\tau=t-n+1
$$

6. Compute the importance sampling ratio.
7. Compute the n-step return.
8. Update $Q(S_\tau,A_\tau)$.
9. If learning the target policy, keep $\pi$ greedy with respect to $Q$.

---

# Off-policy Expected Sarsa

Expected Sarsa replaces the sampled final action with its expected value.

The n-step return is

$$
G_{t:t+n}
=
R_{t+1}
+\cdots
+\gamma^{n-1}R_{t+n}
+\gamma^n\bar V(S_{t+n})
$$

where

$$
\bar V(s)
=
\sum_a
\pi(a|s)Q(s,a)
$$

The importance sampling ratio becomes

$$
\rho_{t+1:t+n-1}
$$

(one fewer factor than n-step Sarsa).

Reason:

The final action is replaced by an expectation and therefore does not require correction.

---

# 7.4 Per-decision Methods with Control Variates

Ordinary off-policy n-step methods are correct but often suffer from **high variance** because of importance sampling.

This section introduces **per-decision importance sampling** together with **control variates** to reduce variance while preserving unbiasedness.

---

# Recursive n-step Return

The ordinary n-step return can be written recursively as

$$
G_{t:h}
=
R_{t+1}
+
\gamma G_{t+1:h},
\qquad t<h<T
$$

with

$$
G_{h:h}=V_{h-1}(S_h).
$$

---

# Problem with Ordinary Importance Sampling

A simple off-policy correction would multiply the entire return by

$$
\rho_t
=
\frac{\pi(A_t|S_t)}
{b(A_t|S_t)}.
$$

However:

- Large $\rho$ produces very high variance.
- If $\rho=0$, the return becomes zero, causing an undesirable update toward zero.

---

# Control Variate Return

Instead, define

$$
G_{t:h}
=
\rho_t
\left(
R_{t+1}
+
\gamma G_{t+1:h}
\right)
+
(1-\rho_t)V(S_t).
$$

Properties:

- If $\rho=1$, this reduces to the ordinary on-policy return.
- If $\rho=0$, then

$$
G_{t:h}=V(S_t),
$$

so no update occurs.

---

# Why Control Variates?

The additional term

$$
(1-\rho_t)V(S_t)
$$

is called a **control variate**.

Its purpose is to

- reduce variance,
- keep the estimator unbiased.

Because

$$
E[\rho]=1,
$$

the expected contribution of the control variate is zero.

---

# Learning Rule

The standard n-step TD update is still used:

$$
V(S_t)
\leftarrow
V(S_t)
+
\alpha
(G_{t:h}-V(S_t)).
$$

The importance sampling correction is already embedded inside the return.

---

# Action-value Version

For action values,

the first action is **not corrected**, since it has already been taken.

The recursive return becomes

$$
G_{t:h}
=
R_{t+1}
+
\gamma
\rho_{t+1}
\left(
G_{t+1:h}
-
Q(S_{t+1},A_{t+1})
\right)
+
\gamma
\bar V(S_{t+1}).
$$

where

$$
\bar V(s)
=
\sum_a
\pi(a|s)Q(s,a).
$$

This formulation is closely related to Expected Sarsa.

---

# Advantages

- Lower variance than ordinary importance sampling.
- No bias is introduced.
- Stable when $\rho=0$.
- Generalizes the on-policy algorithm.
- More data-efficient in practice.

---

# Limitations

- More mathematically complex.
- Still relies on importance sampling.
- Off-policy learning remains slower than on-policy learning.

---

# Idea

Ordinary off-policy update:

$$
G
=
\rho(R+\gamma G)
$$

Improved update with control variate:

$$
G
=
\rho(R+\gamma G)
+
(1-\rho)V.
$$

The second term prevents unnecessary updates when the sampled action is inconsistent with the target policy.

---

# 7.6 n-step Q(σ)

## Goal

Unify all multi-step action-value methods into one algorithm.

---

## Sampling Parameter

$$
\sigma_t \in [0,1]
$$

- $\sigma=1$ → full sampling
- $\sigma=0$ → full expectation

---

## Special Cases

| σ | Algorithm |
|---|-----------|
| 1 | n-step Sarsa |
| 0 | Tree Backup |
| Mixed | Hybrid |
| Last step expectation | Expected Sarsa |

---

## General Return

$$
G_{t:h}
=
R_{t+1}
+
\gamma
\left(
\sigma_{t+1}\rho_{t+1}
+
(1-\sigma_{t+1})
\pi(A_{t+1}|S_{t+1})
\right)
\left(
G_{t+1:h}
-
Q(S_{t+1},A_{t+1})
\right)
+
\gamma
\bar V(S_{t+1})
$$

---

## Update Rule

$$
Q(S_t,A_t)
\leftarrow
Q(S_t,A_t)
+
\alpha
(G-Q)
$$

---