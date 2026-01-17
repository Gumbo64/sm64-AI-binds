from population import Population, write_gamepads_to_file
from sm64_binds import GamePad, buttons
import numpy as np
from collections import defaultdict
import itertools


if __name__ == "__main__":
    from LivePointCloud import LivePointCloud
    from pop_curiosity import CuriosityScoring

    import pickle
    import os

    viewer = LivePointCloud()  # Opens browser tab

    pop_size = 100
    genome_length = 500
    step_amount = 10

    rebirth_time = 30 * 60

    # CuriosityScoring
    filename = 'curiosity_scoring.pkl'
    if os.path.exists(filename):
        with open(filename, 'rb') as file:
            scoring = pickle.load(file)
    else:
        scoring = CuriosityScoring()

    pop = Population(pop_size=pop_size, genome_length=genome_length, scoring=scoring)

    while True:
        for _ in range(rebirth_time // step_amount):
            # print(pop.iterate())
            pop.iterate()

            pop.base_step(n=step_amount, viewer=viewer)
            write_gamepads_to_file(pop.base_path, filename=f"runs/curiosity-pop{pop_size}-genlen{genome_length}-step{step_amount}.m64")

        pop.reset_base()
        with open(filename, 'wb') as file:
            pickle.dump(scoring, file)
