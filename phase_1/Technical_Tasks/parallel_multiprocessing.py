import time
import multiprocessing as mp
import sconepy

MODEL = "data/H0918_osim3.scone"


def run_simulation(worker_id):

    start = time.perf_counter()

    model = sconepy.load_model(MODEL)
    model.reset()

    for t in [0.2, 0.4, 0.6, 0.8, 1.0]:
        model.advance_simulation_to(t)

    elapsed = time.perf_counter() - start

    print(
        f"Worker {worker_id} | "
        f"Simulation={model.time():.2f}s | "
        f"Execution={elapsed:.2f}s"
    )


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    start = time.perf_counter()

    processes = []

    NUM_PROCESSES = 4

    for i in range(NUM_PROCESSES):
        p = mp.Process(target=run_simulation, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total = time.perf_counter() - start

    print("=" * 40)
    print("Parallel Test")
    print("=" * 40)
    print(f"Processes : {NUM_PROCESSES}")
    print(f"Total Time: {total:.2f} s")