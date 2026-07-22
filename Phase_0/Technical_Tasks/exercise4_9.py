import numpy as np
import matplotlib.pyplot as plt

def value_iteration(ph, theta=1e-9):
    GOAL = 100
    V = np.zeros(GOAL + 1)
    V[GOAL] = 1          
    while True:
        delta = 0
        for state in range(1, GOAL):
            old_value = V[state]
            actions = range(1, min(state, GOAL - state) + 1)
            action_values = []
            for action in actions:
                win_state = state + action
                lose_state = state - action
                value = (
                    ph * V[win_state] +
                    (1 - ph) * V[lose_state]
                )
                action_values.append(value)
            V[state] = max(action_values)
            delta = max(delta, abs(old_value - V[state]))
        if delta < theta:
            break

    policy = np.zeros(GOAL + 1)
    for state in range(1, GOAL):
        actions = range(1, min(state, GOAL - state) + 1)
        action_values = []
        for action in actions:
            win_state = state + action
            lose_state = state - action
            value = (
                ph * V[win_state] +
                (1 - ph) * V[lose_state]
            )
            action_values.append(value)
        action_values = np.round(action_values, 8)
        best_action = actions[np.argmax(action_values)]
        policy[state] = best_action
    return V, policy

V025, P025 = value_iteration(0.25)
V055, P055 = value_iteration(0.55)

fig, ax = plt.subplots(2, 2, figsize=(12, 8))

ax[0, 0].plot(V025)
ax[0, 0].set_title("Value Function (ph=0.25)")
ax[0, 0].set_xlabel("Capital")
ax[0, 0].set_ylabel("Probability of Winning")

ax[1, 0].step(range(101), P025, where='mid')
ax[1, 0].set_title("Optimal Policy (ph=0.25)")
ax[1, 0].set_xlabel("Capital")
ax[1, 0].set_ylabel("Stake")

ax[0, 1].plot(V055)
ax[0, 1].set_title("Value Function (ph=0.55)")
ax[0, 1].set_xlabel("Capital")
ax[0, 1].set_ylabel("Probability of Winning")

ax[1, 1].step(range(101), P055, where='mid')
ax[1, 1].set_title("Optimal Policy (ph=0.55)")
ax[1, 1].set_xlabel("Capital")
ax[1, 1].set_ylabel("Stake")
plt.tight_layout()
plt.savefig("gambler_fixed_result.png", dpi=300)
plt.show()