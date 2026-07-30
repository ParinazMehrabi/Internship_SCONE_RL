import time
import sconepy

MODEL = "data/H0918_osim3.scone"


def run_once():
    model = sconepy.load_model(MODEL)
    model.reset()

    for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
        model.advance_simulation_to(t)

    return model.time()


if __name__ == "__main__":
    start = time.perf_counter()

    sim_time = run_once()

    elapsed = time.perf_counter() - start

    print("=" * 40)
    print("Single Process Test")
    print("=" * 40)
    print(f"Simulation Time : {sim_time:.2f} s")
    print(f"Execution Time  : {elapsed:.2f} s")