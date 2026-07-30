import numpy as np
import pandas as pd
try:
    from sconetools import sconepy 
except ImportError:
    import sconepy  

MODEL_FILE = "data/H0918_hfd.scone"  
MAX_TIME = 3.0
DT = 0.01
MIN_COM_HEIGHT = 0.3


def run_and_record(model, random_seed=1, max_time=MAX_TIME, dt=DT, store_data=True):
    model.reset()
    model.set_store_data(store_data)

    rng = np.random.default_rng(random_seed)
    muscle_activations = 0.1 + 0.4 * rng.random(len(model.muscles()))
    model.init_muscle_activations(muscle_activations)

    #  DOFs
    dof_positions = model.dof_position_array()
    dof_positions += 0.1 * rng.random(len(dof_positions)) - 0.05
    model.set_dof_positions(dof_positions)

    for d in model.dofs():
        if d.name() == "pelvis_ty":
            d.set_pos(0.1 + d.pos())

    model.init_state_from_dofs()
    knee_dofs = [d.name() for d in model.dofs() if "knee" in d.name().lower()]
    print("[INFO] Knee-like DOFs found:", knee_dofs)

    knee_r_name = next(
        (n for n in knee_dofs if n.lower().endswith("_r") or "right" in n.lower()),
        None,
    )
    knee_l_name = next(
        (n for n in knee_dofs if n.lower().endswith("_l") or "left" in n.lower()),
        None,
    )

    print(f"[INFO] Selected right-knee DOF: {knee_r_name}")
    print(f"[INFO] Selected left-knee DOF:  {knee_l_name}")

    if knee_r_name is None or knee_l_name is None:
        print(
            "[WARNING] Could not confidently detect both knee DOFs.\n"
            "          Please print all DOFs and set knee_r_name/knee_l_name manually."
        )
        print("[INFO] All DOFs in this model:")
        for d in model.dofs():
            print(" -", d.name())

    time_log, knee_r_log, knee_l_log = [], [], []

    for t in np.arange(0, max_time, dt):
        mus_in = model.muscle_force_array()
        mus_in += model.muscle_fiber_length_array() - 1
        mus_in += 0.2 * model.muscle_fiber_velocity_array()
        model.set_actuator_inputs(mus_in)

        model.advance_simulation_to(t)

        time_log.append(model.time())

        dof_pos = {d.name(): d.pos() for d in model.dofs()}
        knee_r_rad = dof_pos.get(knee_r_name, np.nan)
        knee_l_rad = dof_pos.get(knee_l_name, np.nan)

        knee_r_log.append(np.degrees(knee_r_rad))
        knee_l_log.append(np.degrees(knee_l_rad))

        com_y = model.com_pos().y
        if com_y < MIN_COM_HEIGHT:
            print(
                f"[ABORT] Stopping early at t={model.time():.3f}s because COM height dropped "
                f"below threshold: com_y={com_y:.4f} < {MIN_COM_HEIGHT}"
            )
            break
 
    if store_data:
        dirname = "sconepy_baseline_" + model.name()
        filename = model.name() + f"_{random_seed}_{model.time():0.3f}"
        model.write_results(dirname, filename)
        print(f"[INFO] SCONE results written to: {dirname}/{filename}")

    return np.asarray(time_log), np.asarray(knee_r_log), np.asarray(knee_l_log)


if __name__ == "__main__":
    sconepy.set_log_level(3)

    print("[INFO] Loading model:", MODEL_FILE)
    model = sconepy.load_model(MODEL_FILE)

    print("[INFO] Running baseline simulation...")
    t, kr, kl = run_and_record(model, random_seed=1)

    out_csv = "baseline_knee_trajectory.csv"
    pd.DataFrame({"time": t, "knee_r_deg": kr, "knee_l_deg": kl}).to_csv(out_csv, index=False)
    print(f"[INFO] Knee trajectories saved to: {out_csv}")
    try:
        from knee_cost import KneeSimilarityCost

        print("[INFO] Computing KneeSimilarityCost against knee_reference_gait_cycle.csv ...")
        cost = KneeSimilarityCost("knee_reference_gait_cycle.csv")
        result = cost.evaluate(t, kr, kl)
        print("[RESULT] Baseline knee-similarity cost:", result)

    except Exception as e:
        print("[ERROR] Failed to compute KneeSimilarityCost.")
        print("        Make sure knee_cost.py is in the same folder and reference CSV exists.")
        print("        Exception:", repr(e))
