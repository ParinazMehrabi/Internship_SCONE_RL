import numpy as np
import pandas as pd
import sconepy
from simulate_reflex import simulate, CONTROL_PARAMS, BOUNDS

W_KNEE = 1.0
W_COT  = 0.3


def weighted_objective(params):
    knee_result, cot, fell, _ = simulate(params)
    return W_KNEE * knee_result["total"] + W_COT * cot


def main():
    sconepy.set_log_level(3)
    x0 = np.array([
        2.0,   # vasti.KL
        0.8,   # vasti.KF
        0.66,  # soleus.KL
        1.0,   # soleus.KF
        1.0,   # iliopsoas.KL
        0.8,   # hamstrings.KL
        1.3,   # hamstrings-pelvis KP
        1.7,   # iliopsoas-pelvis KP
    ])

    try:
        import cma
        print("Using CMA-ES on reflex parameters ...")
        es = cma.CMAEvolutionStrategy(
            x0, 0.25,
            {
                "bounds": [[b[0] for b in BOUNDS], [b[1] for b in BOUNDS]],
                "maxiter": 12,
                "popsize": 8,
            }
        )

        history = []
        while not es.stop():
            sols = es.ask()
            fits = [weighted_objective(s) for s in sols]
            es.tell(sols, fits)
            history.append({"iter": len(history), "best": min(fits)})
            print(history[-1])

        best_x = es.result.xbest

    except ImportError:
        from scipy.optimize import minimize
        print("Using Nelder-Mead")
        res = minimize(weighted_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 200})
        best_x = res.x
        history = [{"iter": 0, "best": res.fun}]

    print("\n===== Best Reflex Parameters =====")
    for name, val in zip(CONTROL_PARAMS, best_x):
        print(f"{name:<45s} = {val:.4f}")

    knee_result, cot, fell, (t, kr, kl) = simulate(
        best_x, store_data=True, tag="reflex_best"
    )

    print("\nFinal:")
    print("Knee:", knee_result)
    print("Cost of Walking:", cot)
    print("Fell:", fell)

    pd.DataFrame(history).to_csv("reflex_weighted_history.csv", index=False)
    pd.DataFrame({"time": t, "knee_r": kr, "knee_l": kl}).to_csv(
        "reflex_best_trajectory.csv", index=False
    )
    pd.DataFrame({
        "parameter": CONTROL_PARAMS,
        "value": best_x
    }).to_csv("reflex_best_params.csv", index=False)


if __name__ == "__main__":
    main()