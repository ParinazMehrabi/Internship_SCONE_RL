import numpy as np
import pandas as pd
import sconepy
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.core.callback import Callback

from simulate_reflex import simulate, CONTROL_PARAMS, BOUNDS

XL = np.array([b[0] for b in BOUNDS])
XU = np.array([b[1] for b in BOUNDS])


class PrintCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["best_knee"] = []
        self.data["best_cot"] = []

    def notify(self, algorithm):
        F = algorithm.pop.get("F")
        best_knee = F[:, 0].min()
        best_cot = F[:, 1].min()
        self.data["best_knee"].append(best_knee)
        self.data["best_cot"].append(best_cot)
        print(f"Gen {algorithm.n_gen:3d} | Best Knee RMSE: {best_knee:8.3f} | Best Cost of Walking: {best_cot:8.5f}")


class ReflexProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(
            n_var=len(CONTROL_PARAMS),
            n_obj=2,
            n_constr=0,
            xl=XL,
            xu=XU
        )

    def _evaluate(self, x, out, *args, **kwargs):
        knee_result, cot, fell, _ = simulate(x)
        out["F"] = [knee_result["total"], cot]


def main():
    sconepy.set_log_level(3)

    problem = ReflexProblem()
    callback = PrintCallback()

    algorithm = NSGA2(
        pop_size=24,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    print("=" * 70)
    print("Starting NSGA-II on Reflex Parameters")
    print("=" * 70)
    print(f"Number of parameters : {len(CONTROL_PARAMS)}")
    print(f"Population size      : 24")
    print(f"Generations          : 35")
    print("=" * 70)

    res = minimize(
        problem,
        algorithm,
        ("n_gen", 35),
        seed=1,
        verbose=False,
        callback=callback
    )

    df = pd.DataFrame(res.X, columns=CONTROL_PARAMS)
    df["knee_rmse"] = res.F[:, 0]
    df["cost_of_walking"] = res.F[:, 1]
    df.to_csv("reflex_nsga2_pareto.csv", index=False)

    print("\n" + "=" * 70)
    print("Pareto Front (first 10 solutions)")
    print("=" * 70)
    print(df.head(10).to_string())
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 6))
    plt.scatter(res.F[:, 0], res.F[:, 1], c="steelblue", s=50, alpha=0.8)
    plt.xlabel("Knee RMSE (deg)")
    plt.ylabel("Cost of Walking")
    plt.title("NSGA-II Pareto Front – Reflex Parameters")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("reflex_nsga2_pareto.png", dpi=150)
    print("\nPlot saved → reflex_nsga2_pareto.png")
    best_knee_idx = np.argmin(res.F[:, 0])
    best_cot_idx  = np.argmin(res.F[:, 1])

    print("\n" + "=" * 70)
    print("BEST SOLUTION FOR KNEE RMSE")
    print("=" * 70)
    print(f"Knee RMSE        : {res.F[best_knee_idx, 0]:.4f}")
    print(f"Cost of Walking  : {res.F[best_knee_idx, 1]:.6f}")
    print("\nParameters:")
    for name, val in zip(CONTROL_PARAMS, res.X[best_knee_idx]):
        print(f"  {name:<45s} = {val:.4f}")

    print("\n" + "=" * 70)
    print("BEST SOLUTION FOR COST OF WALKING")
    print("=" * 70)
    print(f"Knee RMSE        : {res.F[best_cot_idx, 0]:.4f}")
    print(f"Cost of Walking  : {res.F[best_cot_idx, 1]:.6f}")
    print("\nParameters:")
    for name, val in zip(CONTROL_PARAMS, res.X[best_cot_idx]):
        print(f"  {name:<45s} = {val:.4f}")

    for label, idx in [("best_knee", best_knee_idx), ("best_cot", best_cot_idx)]:
        knee_result, cot, fell, (t, kr, kl) = simulate(
            res.X[idx], store_data=True, tag=f"nsga2_{label}"
        )
        pd.DataFrame({
            "time": t,
            "knee_r": kr,
            "knee_l": kl
        }).to_csv(f"reflex_nsga2_{label}_trajectory.csv", index=False)

        print(f"\n[{label}] Fell: {fell} | Final Knee RMSE: {knee_result['total']:.3f}")

    print("\nAll files saved.")


if __name__ == "__main__":
    main()