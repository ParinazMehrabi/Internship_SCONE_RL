import gym
import sconegym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
import os


class SB3CompatibleWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_mode = None

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)

        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple) and len(obs) == 2:
            return obs
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
            return obs, reward, terminated, truncated, info
        return result


ENV_ID = "sconewalk_h0918_osim-v1"
TOTAL_TIMESTEPS = 200_000
SAVE_FREQ = 20_000
LOG_DIR = "./logs_ppo"
MODEL_DIR = "./models_ppo"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

env = gym.make(ENV_ID)
env = SB3CompatibleWrapper(env)

if not hasattr(env, "render_mode"):
    env.render_mode = None

model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,
    verbose=1,
    tensorboard_log=LOG_DIR,
)

checkpoint_callback = CheckpointCallback(
    save_freq=SAVE_FREQ,
    save_path=MODEL_DIR,
    name_prefix="ppo_scone_walk"
)

print("Starting training...")

model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=checkpoint_callback,
    progress_bar=True,
)

final_path = os.path.join(MODEL_DIR, "ppo_scone_walk_final")
model.save(final_path)

print(f"Final model saved to: {final_path}")

env.close()
