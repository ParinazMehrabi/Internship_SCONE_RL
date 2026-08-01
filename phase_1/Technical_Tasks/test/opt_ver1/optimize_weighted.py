import numpy as np
import pandas as pd
import sconepy
from simulate_and_cost import simulate

W_KNEE = 1.0
W_COT  = 0.4


def weighted_objective(params):
    knee_result, cot, fell, _ = simulate(params)
    return W_KNEE * knee_result["total"] + W_COT * cot


def main():
    sconepy.set_log_level(3)

    x0 = np.array([1.0, 1.05, 0.0])         
    bounds = [(0.2, 3.0), (0.75, 1.5), (-0.5, 0.5)]

    try:
        import cma
        print("Using CMA-ES ...")
        es = cma.CMAEvolutionStrategy(
            x0, 0.2,
            {
                "bounds": [[b[0] for b in bounds], [b[1] for b in bounds]],
                "maxiter": 45,
                "popsize": 16,
                "verb_disp": 1,
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
        print("cma not found → using Nelder-Mead")
        res = minimize(weighted_objective, x0, method="Nelder-Mead",
                       options={"maxiter": 250})
        best_x = res.x
        history = [{"iter": 0, "best": res.fun}]

    print("\n===== Best parameters =====")
    print(best_x)

    knee_result, cot, fell, (t, kr, kl) = simulate(
        best_x, store_data=True, tag="weighted_best"
    )

    print("\nFinal evaluation:")
    print("Knee:", knee_result)
    print("Cost of Walking:", cot)
    print("Fell:", fell)

    pd.DataFrame(history).to_csv("weighted_history.csv", index=False)
    pd.DataFrame({"time": t, "knee_r": kr, "knee_l": kl}).to_csv(
        "weighted_best_trajectory.csv", index=False
    )
    pd.DataFrame([{
        "k_force": best_x[0],
        "len_offset": best_x[1],
        "k_vel": best_x[2],
        **knee_result,
        "cost_of_walking": cot,
        "fell": fell
    }]).to_csv("weighted_best_params.csv", index=False)

    print("\nFiles saved.")


if __name__ == "__main__":
    main()