import numpy as np
import pandas as pd
from sconetools import sconepy
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM

from optimize_weighted import simulate

MODEL_FILE = "data/H0918_hfd.scone"

XL = np.array([0.0, 0.8, -0.5])
XU = np.array([3.0, 1.6, 0.5])


class KneeCostOfWalkingProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=XL, xu=XU)

    def _evaluate(self, x, out, *args, **kwargs):
        knee_result, cost_of_walking, fell, _ = simulate(x)
        fall_penalty = 50.0 if fell else 0.0
        f1 = knee_result["total"] + fall_penalty
        f2 = cost_of_walking + fall_penalty
        out["F"] = [f1, f2]


def main():
    problem = KneeCostOfWalkingProblem()
    algorithm = NSGA2(
        pop_size=20,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", 30), seed=1, verbose=True)

    pareto_df = pd.DataFrame(res.X, columns=["k_force", "len_offset", "k_vel"])
    pareto_df["knee_rmse"] = res.F[:, 0]
    pareto_df["cost_of_walking"] = res.F[:, 1]
    pareto_df.to_csv("nsga2_pareto_front.csv", index=False)
    print("Saved: nsga2_pareto_front.csv")
    print(pareto_df)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Knee RMSE (deg)")
    plt.ylabel("Cost of Walking (metabolic energy / (body weight x distance))")
    plt.title("NSGA-II Pareto Front")
    plt.savefig("nsga2_pareto_front.png", dpi=150)
    print("Saved: nsga2_pareto_front.png")

    best_knee_idx = np.argmin(res.F[:, 0])
    lowest_cot_idx = np.argmin(res.F[:, 1])

    for label, idx in [("best_knee_match", best_knee_idx), ("lowest_cost_of_walking", lowest_cot_idx)]:
        knee_result, cost_of_walking, fell, (t, kr, kl) = simulate(
            res.X[idx], store_data=True, tag=label
        )
        pd.DataFrame({"time": t, "knee_r_deg": kr, "knee_l_deg": kl}).to_csv(
            f"nsga2_{label}_knee_trajectory.csv", index=False
        )
        print(f"Label: {label}, Result: {knee_result}, Cost of walking: {cost_of_walking}")


if __name__ == "__main__":
    sconepy.set_log_level(3)
    model = sconepy.load_model(MODEL_FILE)
    main()
