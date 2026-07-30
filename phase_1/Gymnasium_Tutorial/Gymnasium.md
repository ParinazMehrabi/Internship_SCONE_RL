# Gymnasium Tutorial

# 1. What is Gymnasium?

Gymnasium is a Python library used to create and interact with Reinforcement Learning (RL) environments.

It provides a standard API so that every environment behaves in exactly the same way.

Instead of writing a different interface for every game, robot, or simulator, Gymnasium defines one common interface.

For example:

- CartPole
- MountainCar
- LunarLander
- Blackjack
- FrozenLake

All of them use exactly the same API.

---

# Why Gymnasium?

Suppose you train an RL algorithm on CartPole.

Later you want to train it on Poker.

If both environments follow the Gymnasium API, you don't need to change your RL algorithm.

Only this line changes:

```python
env = gym.make("CartPole-v1")
```

becomes

```python
env = gym.make("PokerEnv")
```

Everything else stays identical.

---

# Reinforcement Learning Components

Every RL problem consists of five components.

| Component | Description |
|------------|-------------|
| Environment | The world |
| Agent | The learner |
| Observation | What the agent sees |
| Action | What the agent does |
| Reward | Feedback from the environment |

Example (Poker)

Environment:
Poker Table

Agent:
Player

Observation:
Your cards

Action:
Check, Call, Raise, Fold

Reward:
Win or lose chips

---

# Gymnasium API

Every environment contains the same functions.

```python
env.reset()

env.step(action)

env.render()

env.close()
```

These four functions are enough to interact with every environment.

---

# reset()

Starts a new episode.

```python
observation, info = env.reset()
```

Returns

- initial observation
- extra information

---

# step()

Moves one step in the environment.

```python
observation, reward, terminated, truncated, info = env.step(action)
```

Returns

Observation

New state.

Reward

How good the action was.

Terminated

Game finished normally.

Truncated

Episode stopped because of a time limit.

Info

Extra debugging information.

---

# render()

Displays the environment.

```python
env.render()
```

---

# close()

Closes the graphical window.

```python
env.close()
```

---

# Action Space

The action space defines what actions are allowed.

Example:

```python
env.action_space
```

CartPole returns

```
Discrete(2)
```

Meaning

```
0 = Left
1 = Right
```

Random action

```python
action = env.action_space.sample()
```

---

# Observation Space

Observation Space defines what the agent can observe.

```python
print(env.observation_space)
```

Poker could return

```
Your Card = King

Pot = 3

Opponent Bet = True
```

---

# Episode

An episode is one complete game.

```
reset()

↓

step()

↓

step()

↓

step()

↓

terminated=True

↓

reset()
```

---

# Typical Gymnasium Loop

```python
import gymnasium as gym

env = gym.make("CartPole-v1", render_mode="human")

observation, info = env.reset()

done = False

while not done:

    action = env.action_space.sample()

    observation, reward, terminated, truncated, info = env.step(action)

    done = terminated or truncated

env.close()
```

This is the basic interaction loop used by almost every RL algorithm.

---
