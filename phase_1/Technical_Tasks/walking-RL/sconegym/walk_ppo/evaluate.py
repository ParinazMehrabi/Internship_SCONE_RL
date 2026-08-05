import os
import gym
import sconegym
from gym.envs.registration import register

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import sconegym.init_v0 as init_v0
curr_dir = os.path.dirname(os.path.abspath(init_v0.__file__))

# Set this to whatever speed you actually want to evaluate walking at.
EVAL_TARGET_VEL = 0.8  # e.g. matches where the velocity curriculum ended up

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
        'target_vel': EVAL_TARGET_VEL,
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

ENV_ID = "sconewalk_h0918_osim_curriculum-v1"


class SB3CompatibleWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

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
            return obs, reward, done, False, info

        return result

    @property
    def render_mode(self):
        return None


MODEL_DIR = "./models_ppo_v4_curriculum"

MODEL_PATH = os.path.join(MODEL_DIR, "ppo_scone_walk_curriculum_final.zip")
VEC_PATH = os.path.join(MODEL_DIR, "vecnormalize_curriculum_final.pkl")

OUTPUT_SUBDIR = "my_walk_results"

NUM_EPISODES = 10
MAX_STEPS_PER_EPISODE = 100000


def make_env():
    def _init():
        env = gym.make(ENV_ID)
        env = SB3CompatibleWrapper(env)
        return env

    return _init


def unwrap_to_core_env(env):
    visited = set()

    while True:
        env_id = id(env)

        if env_id in visited:
            break

        visited.add(env_id)
        if hasattr(env, "gym_env"):
            next_env = env.gym_env

            if next_env is not env:
                env = next_env
                continue
        if hasattr(env, "env"):
            next_env = env.env

            if next_env is not env:
                env = next_env
                continue

        break

    return env


assert os.path.exists(MODEL_PATH), f"Model checkpoint not found: {MODEL_PATH}"
assert os.path.exists(VEC_PATH), f"VecNormalize file not found: {VEC_PATH}"

base_vec_env = DummyVecEnv([make_env()])

vec_env = VecNormalize.load(
    VEC_PATH,
    base_vec_env
)
vec_env.training = False
vec_env.norm_reward = False

core = unwrap_to_core_env(vec_env.envs[0])

print("=" * 60)
print("Core environment type:")
print(type(core))
print("=" * 60)

core.set_output_dir(OUTPUT_SUBDIR)

print("Results directory :", core.results_dir)
print("Output directory  :", core.output_dir)


model = PPO.load(
    MODEL_PATH,
    env=vec_env
)


for ep in range(NUM_EPISODES):

    print("\n" + "=" * 60)
    print(f"Episode {ep + 1} / {NUM_EPISODES}")
    print("=" * 60)
    core.store_next = True

    print("store_next before reset:", core.store_next)

    obs = vec_env.reset()

    print("store_next after reset :", core.store_next)

    terminated = False
    reward_sum = 0.0
    steps = 0

    while not terminated:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, rewards, dones, infos = vec_env.step(action)
        reward_sum += float(rewards[0])
        terminated = bool(dones[0])

        steps += 1

        if steps >= MAX_STEPS_PER_EPISODE:
            print(
                f"Episode stopped because it reached "
                f"MAX_STEPS_PER_EPISODE = {MAX_STEPS_PER_EPISODE}"
            )
            break
    core.write_now()

    print("-" * 60)
    print(f"Episode {ep + 1} finished")
    print(f"Reward = {reward_sum:.6f}")
    print(f"Steps  = {steps}")
    print(f"Saved in: {core.output_dir}")
    print("-" * 60)

vec_env.close()

print("\nDone.")