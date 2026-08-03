import gym
import sconegym
from stable_baselines3 import PPO
import os

MODEL_DIR = "./models_ppo"
checkpoints = [f for f in os.listdir(MODEL_DIR) if f.endswith(".zip")]
if not checkpoints:
    print("No model found!")
    exit()

checkpoints.sort()
MODEL_PATH = os.path.join(MODEL_DIR, checkpoints[-1])
print(f"Loading model: {MODEL_PATH}")

env = gym.make("sconewalk_h0918_osim-v1")
model = PPO.load(MODEL_PATH)

for ep in range(3):
    print(f"\n=== Episode {ep + 1} ===")
    obs = env.reset()
    env.store_next_episode()

    total_reward = 0
    done = False
    step = 0

    while not done and step < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        step += 1

    print(f"Reward: {total_reward:.1f} | Steps: {step}")
    env.write_now()

env.close()
print("\nThe .sto files have been created.")
print("Now open SCONE Studio and locate the .sto files.")
