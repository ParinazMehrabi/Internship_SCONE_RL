import numpy as np
import pandas as pd
import sconepy
from knee_cost import KneeSimilarityCost

MODEL_FILE = "data/H0918_osim4.scone"
MAX_TIME = 4.0
FALL_HEIGHT = 0.35
FALL_PENALTY = 150.0
GRAVITY = 9.80665

CONTROL_PARAMS = [
    "S00011.vasti.KL",
    "S00011.vasti.KF",
    "S11111.soleus.KL",
    "S11111.soleus.KF",
    "S11100.iliopsoas.KL",
    "S00011.hamstrings.KL",
    "S00011.hamstrings-pelvis_tilt.KP",
    "S11100.iliopsoas-pelvis_tilt.KP",
]

BOUNDS = [
    (0.5, 4.0),    # vasti.KL
    (0.0, 3.0),    # vasti.KF
    (0.2, 3.0),    # soleus.KL
    (-2.0, 3.0),   # soleus.KF
    (0.3, 4.0),    # iliopsoas.KL
    (0.2, 3.0),    # hamstrings.KL
    (0.0, 4.0),    # hamstrings-pelvis KP
    (0.0, 4.0),    # iliopsoas-pelvis KP
]

cost_fn = KneeSimilarityCost("knee_reference_gait_cycle.csv")

sconepy.set_log_level(3)
model = sconepy.load_model(MODEL_FILE)

knee_dofs = {d.name(): d for d in model.dofs() if "knee" in d.name().lower()}
knee_r_name = next((n for n in knee_dofs if n.endswith("_r") or "right" in n.lower()), None)
knee_l_name = next((n for n in knee_dofs if n.endswith("_l") or "left" in n.lower()), None)

print("Right knee:", knee_r_name)
print("Left  knee:", knee_l_name)
print("Optimizing", len(CONTROL_PARAMS), "reflex parameters")


def body_weight(m):
    return sum(b.mass() for b in m.bodies()) * GRAVITY


def get_activation_cost(m):
    try:
        return float(np.sum(np.abs(m.muscle_activation_array())))
    except Exception:
        return 0.0


def simulate(params, store_data=False, tag=None):
    model.reset()
    model.set_store_data(store_data)

    # تنظیم پارامترهای کنترلر
    for name, value in zip(CONTROL_PARAMS, params):
        model.set_control_parameter(name, float(value))

    model.init_state_from_dofs()

    com_x0 = model.com_pos().x
    bw = body_weight(model)

    time_log = []
    knee_r_log = []
    knee_l_log = []
    act_log = []

    for t in np.arange(0, MAX_TIME + 1e-9, 0.01):
        model.advance_simulation_to(t)

        time_log.append(model.time())
        dof_pos = {d.name(): d.pos() for d in model.dofs()}
        knee_r_log.append(np.degrees(dof_pos.get(knee_r_name, np.nan)))
        knee_l_log.append(np.degrees(dof_pos.get(knee_l_name, np.nan)))
        act_log.append(get_activation_cost(model))

        if model.com_pos().y < FALL_HEIGHT:
            break

    fell = model.com_pos().y < FALL_HEIGHT

    if store_data:
        model.write_results("reflex_opt_results", f"{tag}_{model.time():0.3f}")

    t_arr = np.array(time_log)
    kr = np.array(knee_r_log)
    kl = np.array(knee_l_log)

    knee_result = cost_fn.evaluate(t_arr, kr, kl)

    distance = abs(model.com_pos().x - com_x0)
    total_act = float(np.trapz(act_log, t_arr)) if len(t_arr) > 1 else 0.0
    cost_of_walking = total_act / (bw * max(distance, 1e-3))

    if fell:
        cost_of_walking += 20.0
        knee_result["rmse_r"] += FALL_PENALTY
        knee_result["rmse_l"] += FALL_PENALTY
        knee_result["total"] += FALL_PENALTY

    return knee_result, cost_of_walking, fell, (t_arr, kr, kl)