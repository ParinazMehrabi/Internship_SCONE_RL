import numpy as np
import pandas as pd
import sconepy
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from simulate_and_cost import simulate

XL = np.array([0.0, 0.7, -0.6])
XU = np.array([3.5, 1.7,  0.6])


class KneeCotProblem(ElementwiseProblem):
    def __init__(self):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=XL, xu=XU)

    def _evaluate(self, x, out, *args, **kwargs):
        knee_result, cot, fell, _ = simulate(x)
        out["F"] = [knee_result["total"], cot]


def main():
    sconepy.set_log_level(3)

    problem = KneeCotProblem()
    algorithm = NSGA2(
        pop_size=24,
        sampling=FloatRandomSampling(),
        crossover=SBX(prob=0.9, eta=15),
        mutation=PM(eta=20),
        eliminate_duplicates=True,
    )

    res = minimize(problem, algorithm, ("n_gen", 30), seed=1, verbose=True)

    df = pd.DataFrame(res.X, columns=["k_force", "len_offset", "k_vel"])
    df["knee_rmse"] = res.F[:, 0]
    df["cost_of_walking"] = res.F[:, 1]
    df.to_csv("nsga2_pareto.csv", index=False)
    print(df)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.scatter(res.F[:, 0], res.F[:, 1])
    plt.xlabel("Knee RMSE")
    plt.ylabel("Cost of Walking")
    plt.title("NSGA-II Pareto Front")
    plt.grid(True, alpha=0.3)
    plt.savefig("nsga2_pareto.png", dpi=150)
    print("Saved nsga2_pareto.png")


if __name__ == "__main__":
    main()