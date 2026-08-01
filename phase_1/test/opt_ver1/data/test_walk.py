import sconepy
import numpy as np


sconepy.set_log_level(3)


model = sconepy.load_model(
    "data/H0918_walk_test.scone"
)


model.set_store_data(True)

model.reset()


for t in np.arange(0,5,0.01):

    model.advance_simulation_to(t)

    print(
        t,
        model.com_pos().x,
        model.com_pos().y
    )


    if model.com_pos().y < 0.3:
        print("FALL")
        break


model.write_results(
    "walk_test",
    "trial"
)


print("DONE")