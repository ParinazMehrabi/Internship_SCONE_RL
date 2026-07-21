# Chapter 2:  Multi-armed Bandits
## Introduction: Evaluative vs. Instructive Feedback

The core difference between **Reinforcement Learning (RL)** and other learning paradigms (like Supervised Learning) lies in the nature of the feedback:

*   **Instructive Feedback:** Tells the learner the "correct" action to take. It is independent of the action actually taken. (Basis of Supervised Learning).
*   **Evaluative Feedback:** Only tells the learner "how good" the taken action was, but not if it was the best possible action. (Basis of Reinforcement Learning).

### Key Concept: The Need for Exploration
Because RL relies on evaluative feedback, the agent must actively **explore** different behaviors to find the one that yields the highest reward.
### Scope of Chapter 2: Nonassociative Setting
In this chapter, we study RL in a simplified **nonassociative** setting (the k-armed bandit problem). In this setting:
*   There is only one situation/state.
*   The goal is to learn the best action through repeated trials without worrying about changing environments.
## 2.1 The k-armed Bandit Problem

### Formal Definition
*   **Goal:** Maximize the expected total reward over a period of time (e.g., 1000 steps).
*   **Action ($A_t$):** The choice made at time step $t$.
*   **Reward ($R_t$):** The numerical feedback received after an action.
*   **Action Value ($q_*(a)$):** The true expected reward of action $a$:
    $$q_*(a) = E[R_t | A_t = a]$$
*   **Estimated Value ($Q_t(a)$):** Our current estimate of the value of action $a$ at time $t$.

### Exploration vs. Exploitation
*   **Exploitation (Greedy Actions):** Selecting the action with the highest estimated value ($Q_t(a)$) to maximize immediate reward.
*   **Exploration (Non-greedy Actions):** Selecting other actions to improve the estimates of their true values.
*   **The Conflict:** One cannot both explore and exploit with a single action selection. 
    *   *Short-term:* Exploitation is better.
    *   *Long-term:* Exploration might lead to discovering a better total strategy by finding superior actions.
### Balancing Exploration and Exploitation
The decision to explore or exploit depends on:
1.  **Precision of Estimates:** How much we trust our current $Q_t(a)$.
2.  **Uncertainty:** How little we know about other actions.
3.  **Remaining Steps:** How much time is left to benefit from new knowledge.

*   **Complex Methods vs. Reality:** While sophisticated mathematical methods exist for balancing these two, they often rely on "stationary" assumptions (environments that don't change).
*   **Approach of this Book:** We focus on simple, robust balancing methods that work even in complex, non-stationary reinforcement learning problems.
## 2.2 Action-value Methods

### 1. Estimating Values: Sample-Average Method
To estimate the value of an action ($Q_t(a)$), we calculate the average of all rewards received for that action prior to time $t$:
$$Q_t(a) = \frac{\sum \text{Rewards}}{\text{Number of times action } a \text{ was taken}}$$
*   As the number of trials increases, $Q_t(a)$ converges to the true value $q_*(a)$.

### 2. Action Selection Rules
*   **Greedy Selection:** Always chooses the action with the highest estimate: $A_t = \text{argmax}_a Q_t(a)$. It focuses entirely on **Exploitation**.
*   **$\epsilon$-greedy Selection:** 
    *   With probability $1-\epsilon$, select the greedy action.
    *   With probability $\epsilon$, select an action at random (Exploration).
    *   **Advantage:** Ensures that all actions are sampled, eventually leading to the discovery of the optimal action.

### Exercise 2.1 (Solution)
For 2 actions and $\epsilon = 0.5$:
*   Probability of picking greedy directly: $1 - \epsilon = 0.5$
*   Probability of picking greedy during random exploration: $\epsilon \times 0.5 = 0.25$
*   **Total probability of selecting the greedy action:** $0.5 + 0.25 = 0.75$.

## 2.3 The 10-armed Testbed

### Purpose of the Experiment
To compare the performance of different action-selection strategies (Greedy and $\epsilon$-greedy), Sutton & Barto designed a standard benchmark called the **10-armed Testbed**.

### Experimental Setup
- The environment contains **10 possible actions** ($k = 10$).
- Each action has an unknown true action value, denoted by $q_*(a)$.
- The true action values are sampled from a normal distribution:

$$
q_*(a) \sim \mathcal{N}(0,1)
$$

- Whenever an action is selected, the reward is also sampled from a normal distribution:

$$
R_t \sim \mathcal{N}(q_*(A_t),1)
$$

- Therefore, rewards are **stochastic (noisy)**, meaning the same action can produce different rewards at different times.

---

### Performance Evaluation
To fairly compare different algorithms:

- **2000 independent bandit problems** were generated.
- Each algorithm interacted with each problem for **1000 steps**.
- The reported results are the **average over all runs**.

Two evaluation metrics were used:

1. **Average Reward**
   - Measures the average reward obtained over time.
   - Higher values indicate better performance.

2. **Percentage of Optimal Action**
   - Measures how often the algorithm selects the true optimal action.
   - Higher percentages indicate more successful learning.

---

### Comparison of Action-Selection Methods

Three methods were evaluated:

- **Greedy ($\epsilon = 0$)**
- **$\epsilon$-greedy ($\epsilon = 0.01$)**
- **$\epsilon$-greedy ($\epsilon = 0.1$)**

Main observations:

- **Greedy ($\epsilon=0$)** quickly exploits the action with the highest current estimate but often becomes trapped in a suboptimal action because it stops exploring.

- **$\epsilon$-greedy methods** continue exploring other actions, allowing them to discover the true optimal action even if early estimates are inaccurate.

- **$\epsilon=0.1$** explores more frequently, learns faster during the early stages, and reaches higher average rewards sooner.

- **$\epsilon=0.01$** explores less, learns more slowly, but eventually achieves better long-term performance because it spends more time exploiting the optimal action after discovering it.

---

### Effect of the Environment

The benefit of exploration depends on the nature of the environment.

- **Stationary Environment**
  - The true action values remain constant over time.
  - Once the optimal action is found, additional exploration is less important.

- **Nonstationary Environment**
  - The true action values change over time.
  - Continuous exploration is necessary because previously suboptimal actions may become optimal later.

Since most real Reinforcement Learning problems are **nonstationary**, maintaining a balance between exploration and exploitation is essential.

---

### Exercise 2.2 (Solution)

- $A_1=1, R_1=-1 \rightarrow Q_2(1)=-1, Q_2(2,3,4)=0$
- $A_2=2, R_2=1 \rightarrow Q_3(2)=1, Q_3(1)=-1, Q_3(3,4)=0$
- $A_3=2, R_3=-2 \rightarrow Q_4(2)=-0.5, Q_4(1)=-1, Q_4(3,4)=0$
- $A_4=2, R_4=2 \rightarrow Q_5(2)=0.33, Q_5(1)=-1, Q_5(3,4)=0$
- $A_5=3, R_5=0$

1. On which steps did the $\epsilon$ case (random selection) **definitely** occur?
2. On which steps could it have **possibly** occurred?

- **Definitely occurred:**
    - **Step 4:** The greedy actions were 3 and 4 ($Q=0$), but action 2 ($Q=-0.5$) was selected.
    - **Step 5:** The greedy action was 2 ($Q=0.33$), but action 3 ($Q=0$) was selected.
- **Possibly occurred:**
    - **Steps 1, 2, 3, 4, 5.** In an $\epsilon$-greedy algorithm, even if the selected action is greedy, it might have been chosen by the random $\epsilon$ process.

### Exercise 2.3: Quantitative Comparison
In the long run, the **ε = 0.01** method performs best in terms of both cumulative reward and probability of selecting the optimal action.

Although the **ε = 0.1** method learns faster because it explores more, it continues exploring 10% of the time even after finding the optimal action.

The **ε = 0.01** method explores less frequently, but once it identifies the optimal action, it exploits it almost all the time. Therefore, over a sufficiently long period, it achieves a higher cumulative reward.

Quantitatively:

- Greedy selects the optimal action only about **35%** of the time.
- ε = 0.1 selects the optimal action about **90–91%** of the time.
- ε = 0.01 would approach about **99%** optimal-action selection in a very long run.
## 2.4 Incremental Implementation

In the previous section, action values were estimated as sample averages of observed rewards.  
A direct implementation would require storing all past rewards and recomputing the average every time a new reward is observed. This is inefficient in both memory and computation.

For a single action, if it has been selected $n-1$ times and the observed rewards are $R_1, R_2, \dots, R_{n-1}$, then the estimated action value is:

$$
Q_n = \frac{R_1 + R_2 + \cdots + R_{n-1}}{n-1}
$$

Instead of recomputing this average from scratch, Sutton & Barto derive an incremental update rule:

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

This formula has a very important interpretation:

- $Q_n$ is the old estimate
- $R_n$ is the new reward
- $R_n - Q_n$ is the estimation error
- $\frac{1}{n}$ is the step size

So the new estimate is obtained by moving the old estimate a small step toward the new reward.

This method is efficient because it requires only constant memory: we only need to store the current estimate $Q_n$ and the count $n$. The computation per step is also constant.

The general update form used throughout reinforcement learning is:

$$
\text{NewEstimate} \leftarrow \text{OldEstimate} + \text{StepSize}[\text{Target} - \text{OldEstimate}]
$$

In this section, the step size is $\frac{1}{n}$, which makes the update equivalent to a sample average.

### Simple Bandit Algorithm

The book also presents a simple bandit algorithm using incremental updates and $\epsilon$-greedy action selection:

- Initialize for each action $a$:
  - $Q(a) \leftarrow 0$
  - $N(a) \leftarrow 0$

- At each step:
  - Select an action using $\epsilon$-greedy policy
  - Observe reward $R$
  - Increase the action count:
    $$
    N(A) \leftarrow N(A) + 1
    $$
  - Update the action-value estimate incrementally:
    $$
    Q(A) \leftarrow Q(A) + \frac{1}{N(A)}[R - Q(A)]
    $$

This is the standard efficient implementation of sample-average action-value methods.

## 2.5 Tracking a Nonstationary Problem

The averaging methods discussed so far are appropriate for stationary problems. In many reinforcement learning tasks, the environment is **nonstationary**, meaning reward probabilities change over time.

### Constant Step-Size Parameter ($\alpha$)
To handle nonstationarity, we replace $\alpha_n = 1/n$ with a **constant step-size parameter** $\alpha \in (0, 1]$. The update rule becomes:
$$Q_{n+1} = Q_n + \alpha[R_n - Q_n]$$

This creates an **exponentially recency-weighted average**. By expanding the update rule, we can see how past rewards are weighted:
$$
\begin{aligned}
Q_{n+1} &= Q_n + \alpha[R_n - Q_n] \\
&= (1 - \alpha)^n Q_1 + \sum_{i=1}^{n} \alpha(1 - \alpha)^{n-i} R_i
\end{aligned}
$$

- The weight given to the reward $R_i$ is $\alpha(1 - \alpha)^{n-i}$.
- As the number of intervening rewards ($n-i$) increases, the weight decays exponentially.
- This ensures that more recent rewards have a greater influence on the current estimate than older ones.

### Theoretical Convergence vs. Practical Adaptability
Stochastic approximation theory requires two conditions to guarantee convergence to true action values:
1. $\sum_{n=1}^{\infty} \alpha_n(a) = \infty$ (Steps are large enough to overcome initial fluctuations).
2. $\sum_{n=1}^{\infty} \alpha_n^2(a) < \infty$ (Steps eventually become small enough to ensure convergence).

- **Sample-Average ($\alpha_n = 1/n$):** Meets both conditions.
- **Constant Step-Size ($\alpha_n = \alpha$):** Violates the second condition. While this means the estimate does not converge to a single value, it is **desirable** in nonstationary environments because it allows the agent to continuously track and adapt to changes.

In practice, constant step-size parameters are preferred because they adapt to environmental changes much faster than the shrinking step-size method.

## 2.6 Optimistic Initial Values

We can encourage exploration by initializing action-value estimates $Q_1(a)$ to values significantly higher than the expected rewards (e.g., $Q_1(a) = +5$ in the 10-armed testbed).

### Mechanism of Encouraged Exploration
- **The "Disappointment" Effect:** Since the initial estimates are "optimistic," the first observed rewards will almost certainly be lower than the estimates.
- **Forced Exploration:** The agent will try different actions, become "disappointed" by the actual rewards, and keep switching to other actions that still have "optimistic" high estimates.
- **Performance:** As seen in Figure 2.3, this method forces significant exploration early on. While it performs worse initially than $\epsilon$-greedy, it eventually outperforms it as its exploration naturally decreases once true values are learned.

### Limitations
- **Not for Nonstationary Problems:** The exploration drive is inherently **temporary**. If the environment changes later, this method provides no mechanism to restart exploration.
- **Simple Trick:** It is a useful technique for stationary tasks but is not a general-purpose solution for exploration.

## 2.7 Upper-Confidence-Bound Action Selection

Exploration is necessary because action-value estimates are uncertain. Unlike $\epsilon$-greedy, which explores randomly, UCB chooses actions based on both their estimated value and how uncertain those estimates are.

### UCB Action Selection Rule
The action selected at time $t$ is:
$$
A_t \doteq \arg\max_a \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

where:
- $Q_t(a)$ is the current estimated value of action $a$,
- $N_t(a)$ is the number of times action $a$ has been selected,
- $c > 0$ controls the amount of exploration,
- $\ln t$ ensures that exploration decreases slowly over time.

### Intuition
- Actions with high estimated value are preferred.
- Actions that have been tried less often get a larger exploration bonus.
- This creates a principled balance between exploitation and exploration.

### Advantages over $\epsilon$-greedy
- $\epsilon$-greedy explores uniformly at random, without considering uncertainty.
- UCB explores more intelligently by focusing on actions that might be optimal but are still uncertain.

### Empirical Behavior
In the 10-armed testbed, UCB often performs better than $\epsilon$-greedy because it explores systematically at the beginning and then exploits the best actions more effectively.

### Limitations
- UCB is mainly effective in stationary problems.
- It is harder to apply in nonstationary settings and in large or complex reinforcement learning problems.

## 2.8 Gradient Bandit Algorithms

Instead of estimating action values directly, gradient bandit methods learn a preference for each action, denoted by $H_t(a)$. Action selection is done using a softmax distribution:

$$
\Pr\{A_t=a\}=\pi_t(a)=\frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b)}}
$$

Preferences are updated after receiving reward $R_t$:

$$
H_{t+1}(A_t)=H_t(A_t)+\alpha (R_t-\bar{R}_t)(1-\pi_t(A_t))
$$

and for $a\neq A_t$:

$$
H_{t+1}(a)=H_t(a)-\alpha (R_t-\bar{R}_t)\pi_t(a)
$$

Here, $\bar{R}_t$ is a reward baseline. The baseline reduces variance and makes learning more stable. Figure 2.5 shows that using a baseline improves performance, especially when the true action values are shifted to around +4 rather than near zero.

## The Bandit Gradient Algorithm as Stochastic Gradient Ascent

This section explains the gradient bandit algorithm as a stochastic approximation to gradient ascent.

### Main Idea
In exact gradient ascent, action preferences are updated to maximize expected reward:
$$
H_{t+1}(a)\doteq H_t(a)+\alpha \frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)}
$$

### Expected Reward
The performance measure is the expected reward:
$$
\mathbb{E}[R_t]=\sum_x \pi_t(x) q_*(x)
$$

### Why Stochastic?
Exact gradient ascent is not directly possible because the true action values \(q_*(x)\) are unknown.  
The gradient bandit algorithm uses sample rewards to approximate the gradient, so its updates are equal to the true gradient **in expectation**.

### Key Conclusion
The gradient bandit algorithm is an instance of **stochastic gradient ascent**.
## Deriving the Gradient Bandit Update

The goal is to transform the exact performance gradient into a sample-based update rule.

### 1. The Baseline Trick
A baseline $B_t$ (independent of $x$) can be subtracted from $q_*(x)$ without changing the expected gradient:
$$\sum_x \frac{\partial \pi_t(x)}{\partial H_t(a)} = 0$$
- **Purpose:** Reduces variance and stabilizes learning.
- **Common Choice:** $B_t = \bar{R}_t$ (average of all rewards received so far).

### 2. From Summation to Expectation
By multiplying and dividing by $\pi_t(x)$, we convert the gradient sum into an expectation:
$$\frac{\partial \mathbb{E}[R_t]}{\partial H_t(a)} = \mathbb{E} \left[ (q_*(A_t) - B_t) \frac{\partial \pi_t(A_t)}{\partial H_t(a)} \frac{1}{\pi_t(A_t)} \right]$$
This allows the agent to update preferences using **samples** $(A_t, R_t)$ instead of knowing all action values.

### 3. Final Stochastic Update Rule
Replacing $q_*(A_t)$ with the sampled reward $R_t$ and using the softmax derivative:
$$H_{t+1}(a) = H_t(a) + \alpha(R_t - \bar{R}_t)(\mathbb{1}_{a=A_t} - \pi_t(a))$$

- **If $R_t > \bar{R}_t$:** Preference for the chosen action increases.
- **If $R_t < \bar{R}_t$:** Preference for the chosen action decreases.

## Gradient Bandit: Softmax Derivative and Baseline

### Softmax derivative
For
$$
\pi_t(x)=\frac{e^{H_t(x)}}{\sum_{y=1}^{k} e^{H_t(y)}},
$$
the derivative with respect to $H_t(a)$ is:
$$
\frac{\partial \pi_t(x)}{\partial H_t(a)}
=
\pi_t(x)(\mathbb{1}_{a=x}-\pi_t(a)).
$$

### Intuition
- If $a=x$, increasing $H_t(a)$ directly increases $\pi_t(x)$.
- Because of normalization, increasing one preference decreases others.
- The indicator function handles both cases.

### Conclusion
The expected update of the gradient bandit algorithm equals the gradient of expected reward, so the method is an instance of **stochastic gradient ascent**.

### Baseline
- The baseline must not depend on the selected action.
- It does **not** change the expected update.
- It **does** affect the variance of updates and thus convergence speed.
- Choosing the average reward as baseline is simple and practical.
## 2.9 Associative Search (Contextual Bandits)

- Learn a **policy**: mapping from context/state to best action.
- Unlike nonassociative tasks, different situations require different actions.
- Similar to:
  - **k-armed bandits**: action affects only immediate reward.
  - **full RL**: goal is to learn a policy.
- Different from full RL because the action does **not** affect the next state/context.

### Key idea
If the agent knows the context, it can choose a better action than when all situations are mixed together.

## 2.10 Summary

- ε-greedy: random exploration with small probability ε.
- UCB: deterministic action choice with optimism for less-tried actions.
- Gradient bandit: learns action preferences and uses softmax.
- Optimistic initialization: starts with large Q0 values to encourage exploration.

### Figure 2.6
- Shows average reward over the first 1000 steps versus parameter value.
- Parameters tested on a log scale:
  - ε for ε-greedy
  - α for gradient bandit
  - c for UCB
  - Q0 for optimistic initialization
- Best performance usually occurs at intermediate parameter values.
- The curves are typically inverted-U shaped.
- UCB performs best in this testbed.


