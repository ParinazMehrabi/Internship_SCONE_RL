import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


CSV_PATH = "/Users/parinaz/Internship_SCONE_RL/phase_1/Technical_Tasks/pooria_walk_medium.csv"
N_PERCENT = 101  
FSR_THRESHOLD_FRAC = 0.15


def detect_heel_strikes(fsr_sum, threshold):
    above = fsr_sum > threshold
    strikes = np.where((~above[:-1]) & (above[1:]))[0] + 1
    return strikes


def segment_and_average(signal, strike_idx, n_percent=N_PERCENT):
    cycles = []

    for i in range(len(strike_idx) - 1):
        a, b = strike_idx[i], strike_idx[i + 1]

        if b - a < 5:
            continue

        seg = signal[a:b]

        x_old = np.linspace(0, 100, len(seg))
        x_new = np.linspace(0, 100, n_percent)

        cycles.append(np.interp(x_new, x_old, seg))

    cycles = np.array(cycles)

    return (
        cycles.mean(axis=0),
        cycles.std(axis=0),
        cycles.shape[0],
    )


def main():
    df = pd.read_csv(CSV_PATH)
    fsrR_cols = [
        c for c in df.columns
        if c.startswith("fsr") and c.endswith("R")
    ]

    fsrL_cols = [
        c for c in df.columns
        if c.startswith("fsr") and c.endswith("L")
    ]

    fsrR = df[fsrR_cols].sum(axis=1).to_numpy()
    fsrL = df[fsrL_cols].sum(axis=1).to_numpy()

    strikesR = detect_heel_strikes(
        fsrR,
        FSR_THRESHOLD_FRAC * fsrR.max()
    )

    strikesL = detect_heel_strikes(
        fsrL,
        FSR_THRESHOLD_FRAC * fsrL.max()
    )

    knee_r = df["mr_pos"].to_numpy()
    knee_l = df["ml_pos"].to_numpy()

    mean_r, std_r, n_r = segment_and_average(knee_r, strikesR)
    mean_l, std_l, n_l = segment_and_average(knee_l, strikesL)

    pct = np.linspace(0, 100, N_PERCENT)

    out = pd.DataFrame({
        "percent_gait_cycle": pct,
        "knee_r_mean_deg": mean_r,
        "knee_r_std_deg": std_r,
        "knee_l_mean_deg": mean_l,
        "knee_l_std_deg": std_l,
    })

    out.to_csv("knee_reference_gait_cycle.csv", index=False)

    print(f"Number of valid gait cycles: right={n_r}, left={n_l}")
    print("Saved reference data: knee_reference_gait_cycle.csv")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))

    # Right knee
    ax[0].plot(
        pct,
        mean_r,
        color="#c0392b",
        label="Mean right knee angle"
    )

    ax[0].fill_between(
        pct,
        mean_r - std_r,
        mean_r + std_r,
        color="#c0392b",
        alpha=0.2,
        label="±1 standard deviation"
    )

    ax[0].set_title(f"Right knee (mr_pos) - n={n_r} cycles")
    ax[0].set_xlabel("% Gait Cycle")
    ax[0].set_ylabel("Knee angle (deg, raw motor units)")
    ax[0].legend()
    ax[0].grid(True, alpha=0.3)

    # Left knee
    ax[1].plot(
        pct,
        mean_l,
        color="#2980b9",
        label="Mean left knee angle"
    )

    ax[1].fill_between(
        pct,
        mean_l - std_l,
        mean_l + std_l,
        color="#2980b9",
        alpha=0.2,
        label="±1 standard deviation"
    )

    ax[1].set_title(f"Left knee (ml_pos) - n={n_l} cycles")
    ax[1].set_xlabel("% Gait Cycle")
    ax[1].set_ylabel("Knee angle (deg, raw motor units)")
    ax[1].legend()
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        "knee_reference_gait_cycle.png",
        dpi=150
    )

    print("Saved gait-cycle plot: knee_reference_gait_cycle.png")


if __name__ == "__main__":
    main()
