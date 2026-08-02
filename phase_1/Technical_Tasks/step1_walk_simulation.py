import sconepy
import numpy as np

sconepy.set_log_level(3)

print("SCONE Version:", sconepy.version())

if sconepy.is_supported("ModelOpenSim4"):
    MODEL_FILE = "data/H0918_osim4.scone"
elif sconepy.is_supported("ModelOpenSim3"):
    MODEL_FILE = "data/H0918_osim3.scone"
else:
    raise RuntimeError("Neither ModelOpenSim4 nor ModelOpenSim3 is supported in this SCONE install.")

print("Using model file:", MODEL_FILE)

model = sconepy.load_model(MODEL_FILE)
model.set_store_data(True)
model.reset()

MAX_TIME = 5.0
MIN_COM_HEIGHT = 0.3

print(f"{'time':>6} {'com_x':>10} {'com_y':>10}")
fell = False

for t in np.arange(0, MAX_TIME, 0.01):
    model.advance_simulation_to(t)
    print(f"{t:6.2f} {model.com_pos().x:10.4f} {model.com_pos().y:10.4f}")

    if model.com_pos().y < MIN_COM_HEIGHT:
        print(f"FALL at t={model.time():.3f}")
        fell = True
        break

dirname = "sconepy_step1_walk"
filename = f"walk_result_{model.time():0.3f}_{'fell' if fell else 'ok'}"
model.write_results(dirname, filename)
print(f"Results written to {dirname}/{filename}.sto")

if fell:
    print("DONE (fell) - open the .sto file above in SCONE Studio to see what happened.")
else:
    print("DONE - walked the full 5 seconds without falling. Step 1 complete.")