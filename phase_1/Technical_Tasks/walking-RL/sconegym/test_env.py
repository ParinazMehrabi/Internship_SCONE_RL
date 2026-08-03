import gym
import sconegym

env = gym.make("sconewalk_h0918_osim-v1")

obs = env.reset()
total_reward = 0

for step in range(300): 
    action = env.action_space.sample()  # اکشن تصادفی
    obs, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(f"Episode تمام شد در step {step}")
        break

print(f"Total reward: {total_reward:.3f}")
env.close()