import os
import time
import numpy as np
import gym
import sconegym
from gym.envs.registration import register
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.utils import get_linear_fn

import sconegym.init_v0 as init_v0
curr_dir = os.path.dirname(os.path.abspath(init_v0.__file__))

CURRICULUM_TARGET_VEL = 0.5 

register(
    id="sconewalk_h0918_osim_curriculum-v1",
    entry_point="sconegym.gaitgym:GaitGym",
    kwargs={
        'model_file': curr_dir + '/data-v1/H0918_osim.scone',
        'obs_type': '2D',
        'left_leg_idxs': [3, 4, 5],
        'right_leg_idxs': [6, 7, 8],
        'clip_actions': True,
        'run': False,
        'target_vel': CURRICULUM_TARGET_VEL,
        'leg_switch': True,
        'rew_keys': {
            "vel_coeff": 10,
            "grf_coeff": -0.07281,
            "joint_limit_coeff": -0.1307,
            "smooth_coeff": -0.097,
            "nmuscle_coeff": -1.57929,
            "self_contact_coeff": 0.0,
        }
    }
)
MODEL_DIR = "./models_ppo_v4_curriculum"
os.makedirs(MODEL_DIR, exist_ok=True)

CHECKPOINT_NAME = "ppo_scone_walk_curriculum_final.zip"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, CHECKPOINT_NAME)
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize_curriculum_final.pkl")
ENV_ID = "sconewalk_h0918_osim_curriculum-v1"
NUM_ENVS = 4
ADDITIONAL_TIMESTEPS = 4_000_000
SAVE_FREQ = 100_000
LOG_EVERY_STEPS = 20_000

class SB3CompatibleWrapper(gym.Wrapper):
    def __init__(self, env, alive_bonus=0.02):
        super().__init__(env)
        self.alive_bonus = alive_bonus

    def reset(self, **kwargs):
        kwargs.pop("seed", None)
        kwargs.pop("options", None)
        obs = self.env.reset(**kwargs)
        if isinstance(obs, tuple):
            return obs
        return obs, {}

    def step(self, action):
        result = self.env.step(action)
        if len(result) == 4:
            obs, reward, done, info = result
            reward += self.alive_bonus
            return obs, reward, done, False, info
        return result

    def set_target_vel(self, vel):
        self.env.unwrapped.target_vel = vel

    @property
    def render_mode(self):
        return None

class VelocityCurriculumCallback(BaseCallback):
    def __init__(self, start_vel=0.4, end_vel=1.2, total_steps=8_000_000, verbose=0):
        super().__init__(verbose)
        self.start_vel = start_vel
        self.end_vel = end_vel
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_steps, 1.0)
        current_vel = self.start_vel + progress * (self.end_vel - self.start_vel)
        self.training_env.env_method("set_target_vel", current_vel)

        if self.num_timesteps % 50_000 < self.training_env.num_envs:
            print(f">>> [Curriculum] Timesteps: {self.num_timesteps:,} | Target Velocity: {current_vel:.3f}")
        return True

class DetailedTableCallback(BaseCallback):
    def __init__(self, log_every_steps=20_000, verbose=0):
        super().__init__(verbose)
        self.log_every_steps = log_every_steps
        self.next_log = log_every_steps
        self.t0 = None
        self.header_printed = False

    def _on_training_start(self) -> None:
        self.t0 = time.time()
        print("\n" + "=" * 110)
        print(f"RESUMING TRAINING FROM: {CHECKPOINT_NAME}")
        print("=" * 110)

    def _print_header(self):
        cols = ["timesteps", "elapsed_s", "fps", "ep_rew_mean", "lr", "entropy", "value_loss", "approx_kl"]
        header = " | ".join(f"{c:>12s}" for c in cols)
        print(header)
        print("-" * len(header))
        self.header_printed = True

    def _on_step(self) -> bool:
        if self.num_timesteps >= self.next_log:
            if not self.header_printed:
                self._print_header()
            elapsed = time.time() - self.t0
            fps = int(self.num_timesteps / elapsed) if elapsed > 0 else 0

            logger_vals = self.model.logger.name_to_value
            ep_rew = logger_vals.get("rollout/ep_rew_mean", float("nan"))
            entropy = logger_vals.get("train/entropy_loss", float("nan"))
            v_loss = logger_vals.get("train/value_loss", float("nan"))
            kl = logger_vals.get("train/approx_kl", float("nan"))

            progress = self.model._current_progress_remaining
            lr = self.model.lr_schedule(progress)

            row = [f"{self.num_timesteps:,}", f"{elapsed:.1f}", f"{fps}", f"{ep_rew:.2f}",
                   f"{lr:.2e}", f"{entropy:.3f}", f"{v_loss:.3f}", f"{kl:.4f}"]
            print(" | ".join(f"{v:>12s}" for v in row))
            self.next_log += self.log_every_steps
        return True


def make_env():
    def _init():
        env = gym.make(ENV_ID)
        env = SB3CompatibleWrapper(env, alive_bonus=0.02)
        return env
    return _init


if __name__ == "__main__":
    env = SubprocVecEnv([make_env() for _ in range(NUM_ENVS)])

    if os.path.exists(VEC_PATH):
        print(f"Loading normalization from: {VEC_PATH}")
        env = VecNormalize.load(VEC_PATH, env)
        env.training = True
        env.norm_reward = True
    else:
        print(f"[warning] {VEC_PATH} not found - creating NEW normalization stats "
              f"(this means you will NOT actually be continuing from the 10s-walk run).")
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

    if os.path.exists(CHECKPOINT_PATH):
        print(f"Loading Model: {CHECKPOINT_PATH}")
        model = PPO.load(
            CHECKPOINT_PATH,
            env=env,
            learning_rate=get_linear_fn(start=1e-4, end=1e-5, end_fraction=1.0),
            target_kl=0.03,  # passed directly, same as the run that worked
        )
    else:
        print(f"[warning] {CHECKPOINT_PATH} not found! Starting from scratch "
              f"(this means you will NOT actually be continuing from the 10s-walk run).")
        model = PPO("MlpPolicy", env, verbose=1, use_sde=True, sde_sample_freq=4)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(SAVE_FREQ // NUM_ENVS, 1),
        save_path=MODEL_DIR,
        name_prefix="ppo_scone_resume",
        save_vecnormalize=True
    )

    velocity_curriculum = VelocityCurriculumCallback(
        start_vel=0.4,
        end_vel=1.2,
        total_steps=8_000_000
    )

    table_callback = DetailedTableCallback(log_every_steps=LOG_EVERY_STEPS)

    model.learn(
        total_timesteps=ADDITIONAL_TIMESTEPS,
        callback=[checkpoint_callback, velocity_curriculum, table_callback],
        progress_bar=True,
        reset_num_timesteps=False
    )

    model.save(os.path.join(MODEL_DIR, "ppo_scone_walk_final_v2"))
    env.save(os.path.join(MODEL_DIR, "vecnormalize_final_v2.pkl"))
    env.close()
    print("Training Complete.")