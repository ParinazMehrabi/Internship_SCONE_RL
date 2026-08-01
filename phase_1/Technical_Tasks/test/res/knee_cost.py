
import numpy as np
import pandas as pd


class KneeSimilarityCost:
    def __init__(self, reference_csv="knee_reference_gait_cycle.csv"):
        ref = pd.read_csv(reference_csv)
        self.pct = ref["percent_gait_cycle"].to_numpy()
        self.ref_r = ref["knee_r_mean_deg"].to_numpy()
        self.ref_l = ref["knee_l_mean_deg"].to_numpy()
    @staticmethod
    def detect_heel_strikes_from_grf(time, grf_vertical, threshold=20.0):
        above = grf_vertical > threshold
        idx = np.where((~above[:-1]) & (above[1:]))[0] + 1
        return time[idx]

    @staticmethod
    def resample_to_gait_cycle(time, angle, strike_times, n_percent=101):
        cycles = []
        for i in range(len(strike_times) - 1):
            mask = (time >= strike_times[i]) & (time < strike_times[i + 1])
            if mask.sum() < 5:
                continue
            seg = angle[mask]
            x_old = np.linspace(0, 100, len(seg))
            x_new = np.linspace(0, 100, n_percent)
            cycles.append(np.interp(x_new, x_old, seg))
        if not cycles:
            return None
        return np.mean(cycles, axis=0)
    def evaluate(self, time, knee_r_sim_deg, knee_l_sim_deg,
                 grf_r=None, grf_l=None):
        if grf_r is not None:
            strikes_r = self.detect_heel_strikes_from_grf(time, grf_r)
            sim_r = self.resample_to_gait_cycle(time, knee_r_sim_deg, strikes_r)
        else:
            sim_r = np.interp(np.linspace(0, len(knee_r_sim_deg) - 1, 101),
                               np.arange(len(knee_r_sim_deg)), knee_r_sim_deg)

        if grf_l is not None:
            strikes_l = self.detect_heel_strikes_from_grf(time, grf_l)
            sim_l = self.resample_to_gait_cycle(time, knee_l_sim_deg, strikes_l)
        else:
            sim_l = np.interp(np.linspace(0, len(knee_l_sim_deg) - 1, 101),
                               np.arange(len(knee_l_sim_deg)), knee_l_sim_deg)

        BIG_PENALTY = 1000.0
        rmse_r = BIG_PENALTY if sim_r is None else float(np.sqrt(np.mean((sim_r - self.ref_r) ** 2)))
        rmse_l = BIG_PENALTY if sim_l is None else float(np.sqrt(np.mean((sim_l - self.ref_l) ** 2)))

        return {
            "rmse_r": rmse_r,
            "rmse_l": rmse_l,
            "total": 0.5 * (rmse_r + rmse_l),
        }
if __name__ == "__main__":
    try:
        cost_calculator = KneeSimilarityCost("knee_reference_gait_cycle.csv")
        df = pd.read_csv("pooria_walk_medium.csv")
        
        result = cost_calculator.evaluate(
            time=df["listen_time"].to_numpy(),
            knee_r_sim_deg=df["mr_pos"].to_numpy(),
            knee_l_sim_deg=df["ml_pos"].to_numpy()
        )
        print("--- Cost Function Result ---")
        print(result)
        
    except Exception as e:
        print(f"error {e}")
