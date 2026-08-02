import subprocess
import time
import os

env = os.environ.copy()

scripts = [
    "single_process.py",
    "parallel_multiprocessing.py"
]

for script in scripts:
    print("="*40)
    print("Running:", script)

    start = time.perf_counter()

    subprocess.run(
        ["python3.9", script],
        env=env
    )

    end = time.perf_counter()

    print(f"Finished in {end-start:.2f}s\n")