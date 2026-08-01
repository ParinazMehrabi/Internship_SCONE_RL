# Chapter 9 - On-policy Prediction with Approximation
## Objective

The goal of this chapter is to estimate the **state-value function** using **function approximation** instead of a lookup table.

The approximate value function is represented as:

$$
\hat{v}(s, w) \approx v_\pi(s)
$$

where:

- **s** = state
- **w** = weight vector (parameters of the function)
- **v̂(s, w)** = approximate state value
- **vπ(s)** = true state value under policy π

---

## From Tabular Methods to Function Approximation

### Tabular Methods

- Store one value for each state.
- Updating one state only changes that state's value.
- Works well only when the state space is small.

---

### Function Approximation

Instead of storing a value for every state, we learn a parameterized function.

```
Value(s) = f(s, w)
```

The agent only learns the weight vector **w**.

---

## Weight Vector

The weight vector contains the parameters of the approximation function.

Example:

```
w = [0.5, -1.2, 3.7]
```

Learning means finding the best values for these weights.

---

## Why Use Function Approximation?

Usually,

```
d << |S|
```

where:

- **d** = number of weights
- **|S|** = number of states

Instead of storing millions of state values, we only store a relatively small number of parameters.

Benefits:

- Lower memory usage
- Better scalability
- Applicable to large state spaces

---

## Generalization

One of the biggest advantages of function approximation.

### Tabular Methods

```
Update State A

↓

Only State A changes.
```

### Function Approximation

```
Update State A

↓

Weights change

↓

Many related states change.
```

Advantages:

- Learns faster
- Shares information across similar states
- Can estimate values for unseen states

Disadvantages:

- Updating one state may unintentionally affect others
- Learning becomes harder to analyze and control

---


# 9.1 - Value-function Approximation

Every value-function update in Reinforcement Learning can be viewed as a **supervised learning training example**.

Instead of directly updating a table entry, we train a function approximator using input–output pairs.

---

# General Update Form

Every update is written as:

```
s → u
```

where:

- **s** = state (input)
- **u** = update target (desired output)

The function approximator learns:

```
State → Target Value
```

---

# Examples of Update Targets

### Monte Carlo

$$
S_t → G_t
$$

Target = actual return.

---

### TD(0)

$$S_t → R_{t+1} + γ v̂(S_{t+1}, w)$$


Target = reward + estimated value of the next state.

---

### n-step TD


$$S_t → G_{t:t+n}$$


Target = n-step return.

---

### Dynamic Programming (DP)

$$s → Eπ[R + γv̂(S')]$$


Updates any state using the expected value under policy π.

---

# From RL to Supervised Learning

Each RL update becomes one training example.

Example:

| Input | Target |
|-------|--------|
| State A | 5 |
| State B | 3 |
| State C | 7 |

The function approximator learns the mapping.

---

# Function Approximation

Possible models include:

- Linear function approximation
- Neural networks
- Decision trees
- Regression models

The learned function is used as the estimated value function.

---

# Why Not Every ML Method Works

Many supervised learning algorithms assume:

- A fixed dataset
- Multiple training passes

RL usually does **not** satisfy these assumptions.

---

# Online Learning

In Reinforcement Learning:

```
Experience

↓

Update

↓

New Experience

↓

Update
```

Learning must occur continuously while interacting with the environment.

---

# Nonstationary Target Functions

The target values change over time because:

- Policies may change (GPI).
- Bootstrapping methods (TD, DP) update their own targets.

Therefore, RL requires algorithms that can adapt to changing targets.

---

# 9.2 - The Prediction Objective (VE)

## Why we need an objective?

In tabular methods:

- Each state has its own independent value.
- Updating one state does not affect others.
- The true value function can be learned exactly.

With function approximation:

- One update changes many states.
- The number of parameters is much smaller than the number of states.
- It is impossible to make every state's value exact.

Therefore, an optimization objective is required.

---

# State Importance Distribution

A probability distribution is defined over the state space:

```
μ(s) ≥ 0

Σ μ(s) = 1
```

It represents how important each state's prediction error is.

---

# Mean Squared Value Error (VE)

The prediction objective is

\[
VE(w)=
\sum_{s\in S}
\mu(s)
\left(v_\pi(s)-\hat{v}(s,w)\right)^2
\]

where:

- **vπ(s)** = true value
- **v̂(s,w)** = predicted value
- **μ(s)** = state importance

Goal:

```
Minimize VE
```

---

# Root VE

The square root of VE is called **Root VE**.

It provides a more interpretable measure of prediction error and is commonly used in plots.

---

# On-policy Distribution

A common choice for μ(s):

```
Fraction of time spent in each state
```

This is called the **on-policy distribution**.

- Continuing tasks → stationary distribution.
- Episodic tasks → depends on the starting-state distribution.

---

# On-policy Distribution in Episodic Tasks

In episodic tasks, the state distribution $\mu(s)$ depends on the starting states of the episodes.

### Key Definitions:
- $h(s)$: The probability that an episode begins in state $s$.
- $\eta(s)$: The expected (average) number of time steps spent in state $s$ during a single episode.

### The Recursive Relation for $\eta(s)$:
The average time spent in state $s$ comes from two sources:
1. The episode starting directly in $s$.
2. Transitions from all possible preceding states $\bar{s}$ into $s$.

$$
\eta(s) = h(s) + \sum_{\bar{s}} \eta(\bar{s}) \sum_{a} \pi(a|\bar{s}) p(s|\bar{s}, a) \quad \forall s \in \mathcal{S}
$$

### The On-policy Distribution $\mu(s)$:
To get the probability distribution, we normalize the expected visits $\eta(s)$ so that they sum to one:

$$
\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}
$$

### Handling Discounting ($\gamma < 1$):
If discounting is used, it is treated as a form of "soft termination." This means transitions into future states are weighted less. 
To account for this, we add the factor $\gamma$ to the recursive term:

$$
\eta(s) = h(s) + \gamma \sum_{\bar{s}} \eta(\bar{s}) \sum_{a} \pi(a|\bar{s}) p(s|\bar{s}, a)
$$

---

# The Ideal Goal: Global Optimum

The ultimate objective in function approximation is to find a weight vector $\mathbf{w}^*$ that minimizes the **Value Error (VE)**.

- **Objective:** $VE(\mathbf{w}^*) \le VE(\mathbf{w})$ for all possible weight vectors $\mathbf{w}$.
- **Definition:** This $\mathbf{w}^*$ is called the **Global Optimum**.

In complex function approximators (like deep neural networks), finding the global optimum is often difficult, and we might settle for a **Local Optimum**. However, for linear function approximation, the global optimum is typically achievable.

---

# Is VE the Best Objective?

Not necessarily.

The ultimate goal of Reinforcement Learning is to find a better policy, not simply minimize prediction error.

However, VE is currently the most practical objective for value prediction.

---

# Optimization Goals

## Global Optimum

Find

```
w*
```

such that

```
VE(w*) ≤ VE(w)
```

for every possible weight vector.

---

## Local Optimum

A solution that minimizes VE only within a neighborhood.

Most nonlinear models (e.g., neural networks) can only guarantee convergence to a local optimum.

---

# Divergence

Some RL algorithms may fail to converge.

In those cases,

```
VE → ∞
```

This is called **divergence**.

---


# 9.3: Stochastic-Gradient and Semi-Gradient Methods

Function approximation learns the value function by updating a weight vector using **Stochastic Gradient Descent (SGD)**. Instead of storing a discrete table of values, the algorithm adjusts a parameterized, differentiable function $\hat{v}(s, \mathbf{w})$ based on streaming experience.

---

## Weight Vector & Approximator
* **Weight Vector:** A $d$-dimensional column vector of real values:
  $$\mathbf{w} = (w_1, w_2, \dots, w_d)^T$$
* **Approximate Value Function:** $\hat{v}(s, \mathbf{w})$ must be differentiable with respect to $\mathbf{w}$ for all $s \in \mathcal{S}$.

---

## Ideal SGD Update (With True Values)
Initially, assume each training example consists of a state and its true value under policy $\pi$:
$$S_t \rightarrow v_\pi(S_t)$$

The SGD update minimizes the squared error on the observed example by taking a step in the direction of the negative gradient:

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t - \frac{1}{2} \alpha \nabla \left[ v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t) \right]^2
$$

Evaluating the derivative using the chain rule yields:

$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ v_\pi(S_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$

Where:
* $\alpha > 0$ is the step-size parameter (learning rate).
* $\nabla f(\mathbf{w})$ is the gradient vector of partial derivatives:
  $$
  \nabla f(\mathbf{w}) \doteq \left( \frac{\partial f(\mathbf{w})}{\partial w_1}, \frac{\partial f(\mathbf{w})}{\partial w_2}, \dots, \frac{\partial f(\mathbf{w})}{\partial w_d} \right)^T
  $$

---

## Why Stochastic & Why Small Steps?
1. **Stochastic:** The update is performed using only a **single sample/example** at a time rather than a full batch over the entire state space. This enables efficient online learning.
2. **Small Step Size ($\alpha$):** 
   * Function approximators have limited resources (degrees of freedom); no single $\mathbf{w}$ can perfectly fit all states.
   * Moving all the way to eliminate error on one state destroys approximations for other states. Small steps balance errors across the state distribution $\mu$.
   * Under standard stochastic approximation conditions (Robbins-Monro conditions on decreasing $\alpha$), SGD is guaranteed to converge to a **local optimum**:
     $$
     \sum_{t=1}^{\infty} \alpha_t = \infty \quad \text{and} \quad \sum_{t=1}^{\infty} \alpha_t^2 < \infty
     $$

---

## General SGD with Approximate Targets ($U_t$)
In actual RL, the true value $v_\pi(S_t)$ is unknown. We replace it with an approximate target $U_t \in \mathbb{R}$:
$$S_t \rightarrow U_t$$

The general SGD update equation becomes:

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left[ U_t - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$

### Unbiased Target Guarantee
If $U_t$ is an **unbiased estimate** of $v_\pi(S_t)$, meaning:

$$
\mathbb{E}[U_t \mid S_t = s] = v_\pi(s) \quad \forall t
$$

then $\mathbf{w}_t$ is guaranteed to converge to a **local optimum** under decreasing $\alpha$.

---

## Gradient Monte Carlo Method
In Monte Carlo methods, the target is the actual return following state $S_t$:
$$U_t \doteq G_t$$

* Since $G_t$ is by definition an unbiased estimate of $v_\pi(S_t)$ ($\mathbb{E}[G_t \mid S_t = s] = v_\pi(s)$), **Gradient Monte Carlo is guaranteed to converge to a local optimum**.

### Algorithm Pseudocode: Gradient Monte Carlo Algorithm
```text
Input: the policy π to be evaluated
Input: a differentiable function v̂: S × R^d → R
Algorithm parameter: step size α > 0
Initialize value-function weights w ∈ R^d arbitrarily (e.g., w = 0)

Loop forever (for each episode):
Generate an episode S_0, A_0, R_1, S_1, A_1, ..., R_T, S_T using π
Loop for each step of episode, t = 0, 1, ..., T - 1:
w ← w + α * [ G_t - v̂(S_t, w) ] * ∇v̂(S_t, w)

```

# Semi-gradient Methods & State Aggregation

## Why Semi-gradient?
In true Gradient Descent, the **Target** must be independent of the weights $\mathbf{w}$. However, in bootstrapping methods, the target depends on the current weight vector.

*   **Example (TD target):** $U_t = R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t)$
*   **The Problem:** When we change $\mathbf{w}_t$ to reduce the error, the target itself moves. 
*   **Definition:** "Semi-gradient" methods ignore the effect of weights on the target and only calculate the gradient with respect to the prediction $\hat{v}(S_t, \mathbf{w}_t)$.

---

## Semi-gradient TD(0) Update
For a bootstrapping target $U_t$, the update rule is:

$$
\mathbf{w}_{t+1} \doteq \mathbf{w}_t + \alpha \left[ U_t - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$

Specifically for **Semi-gradient TD(0)**:
$$
\mathbf{w}_{t+1} = \mathbf{w}_t + \alpha \left[ R_{t+1} + \gamma \hat{v}(S_{t+1}, \mathbf{w}_t) - \hat{v}(S_t, \mathbf{w}_t) \right] \nabla \hat{v}(S_t, \mathbf{w}_t)
$$

### Advantages:
1.  **Speed:** Typically learns significantly faster than Monte Carlo.
2.  **Online/Continual:** Can learn step-by-step without waiting for the episode's end.
3.  **Continuing Tasks:** Suitable for tasks that never terminate.

---

## State Aggregation
State aggregation is the simplest form of function approximation.

*   **Mechanism:** States are partitioned into disjoint groups (bins). Each group has one component in the weight vector $\mathbf{w}$.
*   **Gradient in State Aggregation:** 
    *   $\nabla \hat{v}(S_t, \mathbf{w}_t) = 1$ for the component corresponding to $S_t$'s group.
    *   $\nabla \hat{v}(S_t, \mathbf{w}_t) = 0$ for all other components.
*   **Result:** Updating one state only affects the value of its own group.

---

## Gradient vs. Semi-gradient Comparison

| Feature | Gradient Methods (e.g., MC) | Semi-gradient Methods (e.g., TD) |
| :--- | :--- | :--- |
| **Target ($U_t$)** | Independent of $\mathbf{w}$ | Depends on current $\mathbf{w}$ |
| **Gradient** | Full Gradient | Partial Gradient (ignores target) |
| **Convergence** | Guaranteed to local optimum | No general guarantee (except linear case) |
| **Bias** | Unbiased | Biased (due to bootstrapping) |
| **Learning** | Offline (end of episode) | Online (step-by-step) |

---

# linear Function Approximation


The approximate value function is represented as a **linear combination of features**.

---

# Feature Vector

Each state is represented by


$$x(s)
=
[x_1(s),x_2(s),...,x_d(s)]^T
$$

---

# Weight Vector

The parameters are


$$w=
[w_1,w_2,\ldots,w_d]^T

$$
---

# Linear Approximation


$$\hat v(s,w)
=
w^Tx(s)
=
\sum_i w_i x_i(s)$$


The prediction is simply the dot product between the feature vector and the weight vector.

---

# Gradient

For linear approximation,


$$\nabla\hat v(s,w)
=
x(s)
$$

The gradient equals the feature vector.

---

# SGD Update

General update:


$$w
\leftarrow
w
+
\alpha
(Target-Prediction)
x(s)$$


---

# Gradient Monte Carlo

Target:


$$U_t=G_t$$


Update:


$$w
\leftarrow
w
+
\alpha
(G_t-\hat v)
x(s)$$


Gradient Monte Carlo converges to the minimum VE under standard assumptions.

---

# Linear TD(0)

Target:


$$R+\gamma\hat v(S')$$


Update:


$$w
\leftarrow
w
+
\alpha
(R+\gamma\hat v(S')-\hat v(S))
x(S)$$


This algorithm is called **Linear Semi-gradient TD(0)**.

---

# TD Fixed Point

Linear TD converges to a stable fixed point.

The fixed point is generally **not identical** to the minimum VE, but it is stable and well-defined.

---

# Advantages

- Fast computation
- Simple gradients
- Strong theoretical guarantees
- Foundation of many modern RL algorithms

---
# 9.4 - Linear Methods

## Linear Function Approximation

One of the most important special cases of function approximation is when the approximate value function is linear in the weight vector.

$$
\hat v(s,w)=w^T x(s)
=
\sum_{i=1}^{d} w_i x_i(s)
\tag{9.8}
$$

The approximate value function is called **linear in the weights**. It can still be nonlinear with respect to the state $s$, depending on how the features are defined.

---

## Feature Vector

Each state $s$ is represented by a real-valued feature vector:

$$
x(s)
=
[x_1(s),x_2(s),\ldots,x_d(s)]^T
$$

where $d$ is the number of features.

---

## Weight Vector

The parameter vector is:

$$
w=
[w_1,w_2,\ldots,w_d]^T
$$

The number of components in $w$ is the same as the number of components in $x(s)$.

---

## Features

Each component of the feature vector is the value of a feature function:

$$
x_i:\mathcal{S}\rightarrow\mathbb{R}
$$

A feature is the complete function $x_i$, while $x_i(s)$ is the value of that feature for state $s$.

---

## Basis Functions

For linear methods, features are called **basis functions** because they form a linear basis for the set of approximate value functions.

Constructing a $d$-dimensional feature vector is equivalent to selecting $d$ basis functions.

The quality of the approximation strongly depends on the selected features.

---

# Linear SGD

## Linear Gradient

For linear approximation,

$$
\hat v(s,w)=w^T x(s)
$$

the gradient with respect to the weights is:

$$
\nabla_w \hat v(s,w)=x(s)
$$

Thus, the gradient is especially simple: it is exactly the feature vector of the current state.

---

## General SGD Update

The general SGD update becomes:

$$
w_{t+1}
=
w_t
+
\alpha
\left[
U_t-\hat v(S_t,w_t)
\right]
x(S_t)
$$

where:

- $U_t$ is the target value.
- $\alpha>0$ is the step-size parameter.
- $S_t$ is the sampled state.
- $x(S_t)$ is the feature vector of the sampled state.

---

## Why Linear Methods?

Linear methods are particularly useful because:

- Their gradients are very simple.
- They are easy to analyze mathematically.
- The value function is linear in the parameters.
---

## Gradient Monte Carlo

For Gradient Monte Carlo:

$$
U_t=G_t
$$

With linear function approximation, Gradient Monte Carlo converges to the global minimum of the value error objective $VE$, provided that the step size decreases according to the usual stochastic approximation conditions:

$$
\sum_{t=0}^{\infty}\alpha_t=\infty
$$

and

$$
\sum_{t=0}^{\infty}\alpha_t^2<\infty
$$

---

# Linear Semi-gradient TD(0)

## TD(0) Update

Using the shorthand:

$$
x_t=x(S_t)
$$

the linear semi-gradient TD(0) update is:

$$
w_{t+1}
=
w_t
+
\alpha
\left[
R_{t+1}
+
\gamma w_t^T x_{t+1}
-
w_t^T x_t
\right]
x_t
\tag{9.9}
$$

Equivalently:

$$
w_{t+1}
=
w_t
+
\alpha
\left[
R_{t+1}x_t
-
x_t(x_t-\gamma x_{t+1})^T w_t
\right]
$$

Linear semi-gradient TD(0) converges under suitable conditions, but not because of ordinary SGD convergence results. Its convergence requires a separate theorem.

---

## Expected TD Update

Once the continuing system has reached steady state:

$$
\mathbb{E}[w_{t+1}\mid w_t]
=
w_t+\alpha(b-Aw_t)
\tag{9.10}
$$

where:

$$
b
=
\mathbb{E}[R_{t+1}x_t]
\in\mathbb{R}^{d}
$$

and:

$$
A
=
\mathbb{E}
\left[
x_t(x_t-\gamma x_{t+1})^T
\right]
\in\mathbb{R}^{d\times d}
\tag{9.11}
$$

---

## TD Fixed Point

If TD converges, it must converge to a weight vector $w_{\mathrm{TD}}$ satisfying:

$$
b-Aw_{\mathrm{TD}}=0
$$

Therefore:

$$
Aw_{\mathrm{TD}}=b
$$

and, if $A$ is invertible:

$$
w_{\mathrm{TD}}
=
A^{-1}b
\tag{9.12}
$$

This solution is called the **TD fixed point**.

The TD fixed point is generally not the same as the weight vector that minimizes $VE$.

---

# Proof of Convergence of Linear TD(0)

## Expected Update in Matrix Form

The expected update can be rewritten as:

$$
\mathbb{E}[w_{t+1}\mid w_t]
=
(I-\alpha A)w_t+\alpha b
\tag{9.13}
$$

The matrix $A$ multiplies the current weight vector $w_t$.

Therefore, the stability of the learning dynamics depends on $A$, not on $b$.

---

## Diagonal Case

Suppose $A$ is diagonal.

If any diagonal element of $A$ is negative, then the corresponding diagonal element of:

$$
I-\alpha A
$$

is greater than $1$ for $\alpha>0$.

The corresponding component of $w_t$ is amplified repeatedly, causing divergence.

If all diagonal elements of $A$ are positive, then $\alpha$ can be chosen sufficiently small so that all diagonal elements of:

$$
I-\alpha A
$$

lie between $0$ and $1$.

In this case, the dynamic part of the update shrinks the weights toward a stable point.

---

## Positive Definite Matrix

In general, stability is assured when $A$ is positive definite:

$$
y^T A y>0
\qquad
\forall y\neq0
$$

Positive definiteness guarantees:

- Stable expected TD dynamics.
- Existence of a unique TD fixed point.
- Existence of $A^{-1}$.
- Convergence toward $w_{\mathrm{TD}}$ under appropriate assumptions.

---

# Matrix Form of $A$

## Continuing On-policy Case

For linear TD(0) in a continuing task with:

$$
\gamma<1
$$

the matrix $A$ can be written as:

$$
A
=
X^T D(I-\gamma P)X
$$

where:

- $X$ is the $|\mathcal{S}|\times d$ feature matrix, with $x(s)^T$ as the row corresponding to state $s$.
- $D$ is the $|\mathcal{S}|\times|\mathcal{S}|$ diagonal matrix whose diagonal entries are $\mu(s)$.
- $\mu(s)$ is the stationary on-policy distribution.
- $P$ is the state-transition probability matrix under policy $\pi$.
- $I$ is the identity matrix.

---

## Key Matrix

The key inner matrix is:

$$
D(I-\gamma P)
$$

Its properties determine whether $A$ is positive definite and whether TD is stable.

---

## Stationary Distribution

For the stationary distribution $\mu$:

$$
\mu=P^T\mu
$$

Equivalently:

$$
\mu^T P=\mu^T
$$

---

## Column Sums

Let $\mathbf{1}$ be the column vector whose entries are all $1$.

The row vector containing the column sums of a matrix $M$ is:

$$
\mathbf{1}^T M
$$

For the key matrix:

$$
\mathbf{1}^T D(I-\gamma P)
=
\mu^T(I-\gamma P)
$$

Using:

$$
\mu^T P=\mu^T
$$

we obtain:

$$
\mathbf{1}^T D(I-\gamma P)
=
\mu^T-\gamma\mu^T P
$$

$$
=
\mu^T-\gamma\mu^T
$$

$$
=
(1-\gamma)\mu^T
$$

Because:

$$
\gamma<1
$$

and the stationary probabilities are positive for all visited states:

$$
\mu(s)>0
$$

all column sums are positive.

The row sums are also positive because $P$ is a stochastic matrix and $\gamma<1$.

Therefore:

$$
D(I-\gamma P)
$$

is positive definite, which implies that:

$$
A=X^TD(I-\gamma P)X
$$

is positive definite when the features have full column rank.

---

## Convergence Result

For continuing on-policy prediction with:

$$
\gamma<1
$$

Linear semi-gradient TD(0) is stable because $A$ is positive definite.

With additional technical assumptions and an appropriate decreasing step-size schedule, Linear TD(0) converges with probability one to:

$$
w_{\mathrm{TD}}=A^{-1}b
$$

---

# Quality of the TD Fixed Point

At the TD fixed point, the value error is bounded by:

$$
VE(w_{\mathrm{TD}})
\leq
\frac{1}{1-\gamma}
\min_w VE(w)
$$

Thus, TD's asymptotic error is at most:

$$
\frac{1}{1-\gamma}
$$

times the smallest possible value error.

When $\gamma$ is close to $1$, this bound may be large.

Therefore, TD can have worse asymptotic accuracy than Monte Carlo.

However, TD methods usually have much lower variance and often learn faster.

---


# n-step Semi-gradient TD

The n-step semi-gradient TD update is:

$$
w_{\tau+n}
=
w_{\tau+n-1}
+
\alpha
\left[
G_{\tau:\tau+n}
-
\hat v(S_\tau,w_{\tau+n-1})
\right]
\nabla_w\hat v(S_\tau,w_{\tau+n-1})
\tag{9.15}
$$

For linear approximation:

$$
\nabla_w\hat v(S_\tau,w_{\tau+n-1})
=
x(S_\tau)
$$

---

## n-step Return

For a nonterminal bootstrap state:

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
\gamma^n\hat v(S_{t+n},w_{t+n-1})
\tag{9.16}
$$

If the episode terminates before time $t+n$, the bootstrap term is omitted.

---

# 9.5 Feature Construction for Linear Methods

The performance of linear methods in Reinforcement Learning is heavily dependent on **State Representation**. While linear models are computationally efficient and offer convergence guarantees, their success relies on how well the features capture the environment's dynamics.

## The Problem of Feature Interaction
In a linear model, the contribution of one feature is independent of others. However, in many tasks, the "value" of a feature depends on the context of another feature.

The upcoming subsections will explore specific techniques:
1. Polynomials
2. Fourier Basis
3. Coarse Coding
4. Tile Coding
5. Radial Basis Functions

# 9.5.1 Polynomial Features

## Main Idea

Polynomial basis functions extend the original state representation by adding higher-order and interaction terms.

Instead of using only the raw state variables,

$$
x(s)=(s_1,s_2)
$$

we construct richer feature vectors such as

$$
x(s)=
(1,s_1,s_2,s_1s_2,s_1^2,s_2^2,\ldots)
$$

The value function remains linear in the weights:

$$
\hat v(s,w)=w^Tx(s)
$$

although it becomes nonlinear with respect to the original state variables.

## General Polynomial Basis

For a state with $k$ dimensions,

$$
x_i(s)=
\prod_{j=1}^{k}
s_j^{c_{i,j}}
$$

where

$$
c_{i,j}\in\{0,1,\ldots,n\}
$$

The total number of polynomial features is

$$
(n+1)^k
$$

## Advantages

- Simple
- Easy to implement
- Captures feature interactions
- Widely used in regression

## Disadvantages

- Exponential growth of features
- Suffers from the curse of dimensionality

# 9.5.2 Fourier Basis


Fourier basis represents the value function as a weighted sum of cosine basis functions.

For one-dimensional states:

$$
x_i(s)=\cos(i\pi s)
$$

where

$$
i=0,\ldots,n
$$

---

## Multi-dimensional Fourier Basis

For

$$
s=(s_1,\ldots,s_k)
$$

the feature is

$$
x_i(s)=\cos(\pi s^T c_i)
$$

where

$$
c_i=(c_{i1},...,c_{ik})
$$

Each vector $c_i$ determines the frequency along every dimension.

---

## Advantages

- Strong approximation capability
- Easy feature construction
- Captures interactions between state variables
- Frequency selection is intuitive

---

## Step-size Recommendation

For each feature:

$$
\alpha_i=
\frac{\alpha}
{\sqrt{c_{i1}^2+\cdots+c_{ik}^2}}
$$

Lower learning rates are used for high-frequency features.

---

## Disadvantages

- Number of features grows exponentially:

$$
(n+1)^k
$$

- Global generalization
- Gibbs (ringing) effects near discontinuities

---

## Comparison with Polynomial Basis

The book reports that Fourier basis generally:

- learns faster,
- achieves lower prediction error,
- is more stable than polynomial basis.

For this reason, Fourier basis is generally preferred over polynomial basis in online reinforcement learning.

# 9.5.3 Coarse Coding

Represent each state using overlapping binary features.

Each feature corresponds to a region of the state space.

If the state lies inside the region:

Feature = 1

Otherwise:

Feature = 0

---

## Generalization

Updating one state changes the weights of all active features.

Therefore, nearby states sharing those features are updated as well.

---

## Feature Size

Small receptive fields

- Narrow generalization

Large receptive fields

- Broad generalization

---

## points

Feature width mainly affects early learning.

The final approximation quality depends much more on the total number of features.

# 9.5.4 Tile Coding

Tile coding divides the continuous state space into multiple overlapping tilings.

Each tiling is a partition of the state space.

Each partition element is called a tile.

---

## Representation

Each state activates exactly one tile in each tiling.

If there are $n$ tilings, then exactly $n$ binary features are active.

---

## Learning

For linear approximation:

$$
\hat{v}(s,w)=w^Tx(s)
$$

Only the weights of active tiles are updated.

---

## Step Size

A common choice is

$$
\alpha=\frac{1}{n}
$$

or

$$
\alpha=\frac{1}{10n}
$$

where $n$ is the number of tilings.

---

## Advantages

- Fast computation
- Sparse binary features
- Efficient memory usage
- Strong and controllable generalization

---

## Offset Strategy

Uniform offsets may create diagonal artifacts.

Asymmetric offsets provide smoother and more uniform generalization.

---

## Hashing

Hashing maps many tiles into a smaller memory table.

This greatly reduces memory usage while maintaining good performance.

# Chapter 9.6 - Selecting Step-Size Parameters Manually

## The Challenge
Selecting the step-size parameter $\alpha$ is crucial for SGD performance. 
- **Theoretical methods:** Often result in learning that is too slow for practical use.
- **Recursive Least Squares (LSTD):** Requires $O(d^2)$ parameters, which is computationally prohibitive for large-scale problems.

## Intuition from Tabular Case
In a tabular setting, the relationship between $\alpha$ and learning speed is clear:
- $\alpha = 1$: Eliminates error in a single update (One-trial learning).
- $\alpha = 1/10$: Approaches the mean of targets after ~10 experiences.
- **General Rule:** If $\alpha = 1/\tau$, the estimate converges to the mean after approximately $\tau$ experiences.

## Rule of Thumb for Linear SGD
For general linear function approximation, the recommended formula to achieve convergence in $\tau$ experiences with similar feature vectors is:

$$
\alpha \doteq \frac{1}{\tau E[x^T x]}
$$

Where:
- $\tau$: Desired number of experiences for convergence.
- $x$: Random feature vector.
- $E[x^T x]$: The expected squared norm (length) of the feature vector.

## Application to Tile Coding
In Tile Coding, the squared norm $x^T x$ is always equal to the number of tilings ($n$), because exactly $n$ features are active (equal to 1) at any time.

$$
\alpha = \frac{1}{\tau \cdot n}
$$

---

# 9.7 - Nonlinear Function Approximation: Artificial Neural Networks

Artificial Neural Networks (ANNs) are powerful nonlinear function approximators. They are capable of learning highly complex mappings between inputs and outputs and form the foundation of modern Deep Reinforcement Learning.

---

# Neural Network Architecture

A typical feedforward neural network consists of:

- Input Layer
- One or more Hidden Layers
- Output Layer

Each connection between neurons has a learnable weight.

---

# Feedforward vs Recurrent Networks

## Feedforward Network

- Information flows only in one direction.
- No feedback loops.
- Most commonly used in function approximation.

```
Input → Hidden Layer(s) → Output
```

## Recurrent Neural Network (RNN)

- Contains feedback connections.
- Can remember previous information.
- Suitable for sequential data.

---

# Neuron Computation

Each neuron computes a weighted sum of its inputs:

$$
z=\sum_i w_i x_i+b
$$

Then applies a nonlinear activation function:

$$
a=f(z)
$$

where

- $x_i$ : input
- $w_i$ : weight
- $b$ : bias
- $a$ : neuron output

---

# Activation Functions

## Sigmoid

$$
f(x)=\frac{1}{1+e^{-x}}
$$

Properties:

- Output in $(0,1)$
- Smooth and differentiable
- Common in early neural networks

---

## ReLU

$$
f(x)=\max(0,x)
$$

Properties:

- Very simple
- Computationally efficient
- Most widely used in deep learning

---

## Step Function

$$
f(x)=
\begin{cases}
1,&x\ge\theta\\
0,&x<\theta
\end{cases}
$$

Historically important but rarely used for training modern networks.

---

# Universal Approximation Theorem

A feedforward neural network with:

- one hidden layer
- a sufficient number of neurons

can approximate any continuous function with arbitrary accuracy.

This is known as the **Universal Approximation Theorem**.

---

# Why Deep Networks?

Although one hidden layer is theoretically sufficient, deep networks are much more efficient for learning complex functions.

Instead of learning everything in one layer, features are learned hierarchically.

Example:

```
Pixels
    ↓
Edges
    ↓
Corners
    ↓
Shapes
    ↓
Objects
```

This hierarchical representation is called **Feature Learning**.

---

# Learning the Weights

The goal of training is to find the optimal weights.

Weights are updated using **Stochastic Gradient Descent (SGD)**.

General update rule:

$$
w \leftarrow w-\alpha\nabla J(w)
$$

where

- $\alpha$ : learning rate
- $J(w)$ : objective (loss) function

---

# Gradient

The gradient is the vector of partial derivatives of the objective function with respect to all network parameters.

$$
\nabla J(w)
=
\left(
\frac{\partial J}{\partial w_1},
\frac{\partial J}{\partial w_2},
\ldots
\right)
$$

The gradient indicates the direction of the steepest increase of the objective.

Gradient Descent moves in the opposite direction.

---

# Backpropagation

Backpropagation efficiently computes gradients in multilayer neural networks.

It consists of two phases.

## 1. Forward Pass

- Compute neuron activations.
- Produce the network output.

## 2. Backward Pass

- Propagate the prediction error backward.
- Compute gradients using the chain rule.
- Update every weight.

---

## Overfitting

Overfitting occurs when a neural network memorizes the training data instead of learning the underlying patterns.

As a result:

- Low training error
- Poor generalization on unseen data

---

# Regularization

Regularization prevents overfitting by penalizing large weights.

A common objective function is

$$
Loss = Error + \lambda ||w||^2
$$

where

- $\lambda$ controls the regularization strength.

---

# Dropout

Dropout randomly disables neurons during training.

Benefits:

- Prevents co-adaptation
- Improves generalization
- Reduces overfitting

---

# Batch Normalization

Batch Normalization normalizes layer outputs during training.

Advantages:

- Faster convergence
- More stable optimization
- Less sensitivity to initialization

---
