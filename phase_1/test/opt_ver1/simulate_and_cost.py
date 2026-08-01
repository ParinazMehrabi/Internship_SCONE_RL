import numpy as np
import pandas as pd
import sconepy
from knee_cost import KneeSimilarityCost

MODEL_FILE = "data/H0918_osim4.scone"
MAX_TIME = 3.0
FALL_HEIGHT = 0.35
FALL_PENALTY = 200.0         
COT_FALL_PENALTY = 30.0
GRAVITY = 9.80665

cost_fn = KneeSimilarityCost("knee_reference_gait_cycle.csv")

sconepy.set_log_level(3)
model = sconepy.load_model(MODEL_FILE)

knee_dofs = {d.name(): d for d in model.dofs() if "knee" in d.name().lower()}
knee_r_name = next((n for n in knee_dofs if n.endswith("_r") or "right" in n.lower()), None)
knee_l_name = next((n for n in knee_dofs if n.endswith("_l") or "left" in n.lower()), None)

print("Right knee:", knee_r_name)
print("Left  knee:", knee_l_name)


def body_weight(m):
    return sum(b.mass() for b in m.bodies()) * GRAVITY


def get_activation_cost(m):
    try:
        acts = m.muscle_activation_array()
        return float(np.sum(np.abs(acts)))
    except Exception:
        return 0.0


def simulate(params, store_data=False, tag=None, random_seed=42):
    k_force, len_offset, k_vel = params

    model.reset()
    model.set_store_data(store_data)

    rng = np.random.default_rng(random_seed)
    n_mus = len(model.muscles())

    # نویز کمتر
    model.init_muscle_activations(0.20 + 0.15 * rng.random(n_mus))

    dof_pos = model.dof_position_array()
    dof_pos += 0.02 * rng.random(len(dof_pos)) - 0.01
    model.set_dof_positions(dof_pos)

    for d in model.dofs():
        if d.name() == "pelvis_ty":
            d.set_pos(d.pos() + 0.10)  
            break

    model.init_state_from_dofs()

    com_x0 = model.com_pos().x
    bw = body_weight(model)

    time_log = []
    knee_r_log = []
    knee_l_log = []
    act_cost_log = []

    for t in np.arange(0, MAX_TIME + 1e-9, 0.01):
        mus_in = (k_force * model.muscle_force_array()
                  + (model.muscle_fiber_length_array() - len_offset)
                  + k_vel * model.muscle_fiber_velocity_array())

        model.set_actuator_inputs(mus_in)
        model.advance_simulation_to(t)

        time_log.append(model.time())
        dof_pos = {d.name(): d.pos() for d in model.dofs()}
        knee_r_log.append(np.degrees(dof_pos.get(knee_r_name, np.nan)))
        knee_l_log.append(np.degrees(dof_pos.get(knee_l_name, np.nan)))
        act_cost_log.append(get_activation_cost(model))

        if model.com_pos().y < FALL_HEIGHT:
            break

    fell = model.com_pos().y < FALL_HEIGHT

    if store_data:
        model.write_results("sconepy_opt_results", f"{tag}_{model.time():0.3f}")

    t_arr = np.array(time_log)
    kr = np.array(knee_r_log)
    kl = np.array(knee_l_log)

    knee_result = cost_fn.evaluate(t_arr, kr, kl)

    distance = abs(model.com_pos().x - com_x0)
    total_act = float(np.trapz(act_cost_log, t_arr)) if len(t_arr) > 1 else 0.0
    cost_of_walking = total_act / (bw * max(distance, 1e-3))

    if fell:
        cost_of_walking += COT_FALL_PENALTY
        knee_result["rmse_r"] += FALL_PENALTY
        knee_result["rmse_l"] += FALL_PENALTY
        knee_result["total"]  += FALL_PENALTY

    return knee_result, cost_of_walking, fell, (t_arr, kr, kl)