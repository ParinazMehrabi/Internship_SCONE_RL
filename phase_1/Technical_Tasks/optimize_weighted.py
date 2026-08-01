import numpy as np
import pandas as pd
from sconetools import sconepy
from knee_cost import KneeSimilarityCost

MODEL_FILE = "data/H0918_hfd.scone"
MAX_TIME = 3.0
W_KNEE = 1.0
W_COT = 0.5

GRAVITY = 9.80665

cost_fn = KneeSimilarityCost("knee_reference_gait_cycle.csv")
model = sconepy.load_model(MODEL_FILE)

knee_dofs = {d.name(): d for d in model.dofs() if "knee" in d.name().lower()}
knee_r_name = next((n for n in knee_dofs if n.endswith("_r") or "right" in n), None)
knee_l_name = next((n for n in knee_dofs if n.endswith("_l") or "left" in n), None)


def body_weight_newtons(m):
    return sum(b.mass() for b in m.bodies()) * GRAVITY


def get_metabolic_rate(m):
    for attr in ("metabolic_energy_rate", "energy_expenditure_rate", "metabolic_power"):
        if hasattr(m, attr):
            try:
                return float(getattr(m, attr)())
            except Exception:
                pass
    total = 0.0
    for mus, act in zip(m.muscles(), m.muscle_activation_array()):
        try:
            total += float(act) * mus.max_isometric_force()
        except Exception:
            pass
    return total


def simulate(params, random_seed=1, store_data=False, tag=None):
    k_force, len_offset, k_vel = params

    model.reset()
    model.set_store_data(store_data)

    rng = np.random.default_rng(random_seed)
    model.init_muscle_activations(0.1 + 0.4 * rng.random(len(model.muscles())))

    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)

    for d in model.dofs():
        if d.name() == "pelvis_ty":
            d.set_pos(0.1 + d.pos())

    model.init_state_from_dofs()

    com_x0 = model.com_pos().x
    bw = body_weight_newtons(model)

    time_log = []
    knee_r_log = []
    knee_l_log = []
    metab_rate_log = []

    for t in np.arange(0, MAX_TIME, 0.01):
        mus_in = k_force * model.muscle_force_array()
        mus_in += model.muscle_fiber_length_array() - len_offset
        mus_in += k_vel * model.muscle_fiber_velocity_array()

        model.set_actuator_inputs(mus_in)
        model.advance_simulation_to(t)

        time_log.append(model.time())

        dof_pos = {d.name(): d.pos() for d in model.dofs()}
        knee_r_log.append(np.degrees(dof_pos.get(knee_r_name, np.nan)))
        knee_l_log.append(np.degrees(dof_pos.get(knee_l_name, np.nan)))

        metab_rate_log.append(get_metabolic_rate(model))

        if model.com_pos().y < 0.3:
            break

    if store_data:
        dirname = "sconepy_weighted_opt"
        filename = f"{model.name()}_{tag or 'result'}_{model.time():0.3f}"
        model.write_results(dirname, filename)

    t_arr = np.array(time_log)
    kr = np.array(knee_r_log)
    kl = np.array(knee_l_log)

    knee_result = cost_fn.evaluate(t_arr, kr, kl)

    distance = abs(model.com_pos().x - com_x0)
    total_metabolic_energy = float(np.trapz(metab_rate_log, t_arr)) if len(t_arr) > 1 else 0.0
    cost_of_walking = total_metabolic_energy / (bw * max(distance, 1e-3))

    fell = model.com_pos().y < 0.3
    if fell:
        cost_of_walking += 10.0

    return knee_result, cost_of_walking, fell, (t_arr, kr, kl)


def weighted_objective(params):
    knee_result, cost_of_walking, fell, _ = simulate(params)
    fall_penalty = 50.0 if fell else 0.0

    return (
        W_KNEE * knee_result["total"]
        + W_COT * cost_of_walking
        + fall_penalty
    )


def main():
    x0 = np.array([1.0, 1.2, 0.1])
    bounds = [(0.0, 3.0), (0.8, 1.6), (-0.5, 0.5)]

    try:
        import cma

        es = cma.CMAEvolutionStrategy(
            x0,
            0.2,
            {
                "bounds": [
                    [b[0] for b in bounds],
                    [b[1] for b in bounds],
                ],
                "maxiter": 60,
                "popsize": 12,
            },
        )

        history = []

        while not es.stop():
            solutions = es.ask()
            fitnesses = [weighted_objective(s) for s in solutions]
            es.tell(solutions, fitnesses)

            history.append(
                {
                    "iter": len(history),
                    "best": min(fitnesses),
                }
            )

            print(history[-1])

        best_x = es.result.xbest

    except ImportError:
        print("Package 'cma' not found. Falling back to Nelder-Mead optimization.")

        from scipy.optimize import minimize

        res = minimize(
            weighted_objective,
            x0,
            method="Nelder-Mead",
            options={"maxiter": 200},
        )

        best_x = res.x
        history = [{"iter": 0, "best": res.fun}]

    print("Best parameters:", best_x)

    pd.DataFrame(history).to_csv(
        "weighted_optimization_history.csv",
        index=False,
    )

    knee_result, cost_of_walking, fell, (t, kr, kl) = simulate(
        best_x,
        store_data=True,
        tag="best",
    )

    print("Final result:")
    print("Knee similarity:", knee_result)
    print("Cost of walking:", cost_of_walking)
    print("Fall detected:", fell)

    pd.DataFrame(
        {
            "time": t,
            "knee_r_deg": kr,
            "knee_l_deg": kl,
        }
    ).to_csv(
        "weighted_best_knee_trajectory.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "k_force": best_x[0],
                "len_offset": best_x[1],
                "k_vel": best_x[2],
                **knee_result,
                "cost_of_walking": cost_of_walking,
            }
        ]
    ).to_csv(
        "weighted_best_params.csv",
        index=False,
    )


if __name__ == "__main__":
    sconepy.set_log_level(3)
    main()
