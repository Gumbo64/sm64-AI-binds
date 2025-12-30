from population import Population, write_gamepads_to_file
from sm64_binds import GamePad, buttons
import numpy as np
from collections import defaultdict
import itertools

class CuriosityScoring:
    # tmp_states are states between pop.base_game and self.temp_game that we do not want to finalise yet
    def __init__(self):
        self.circle_radius = 1

        # Initialize bucket counts using a dictionary for sparse representation
        self.bucket_counts = defaultdict(lambda: 0)

    def get_fitness(self, tmp_states):
        bucket_keys = [self.get_bucket_key(state) for state in tmp_states]
        
        tmp_bucket_counts = defaultdict(lambda: 0)

        scores = []
        for i, key in enumerate(bucket_keys):
            visits = self.bucket_counts[key] + tmp_bucket_counts[key]
            
            s = -visits - i
            scores.append(s)

            self.add_circle(key, tmp_bucket_counts)

        scores = np.array(scores)

        # curiosity_score = np.median(scores)
        curiosity_score = np.percentile(scores, 90)
        # curiosity_score = np.max(scores)
        return curiosity_score

    def get_bucket_key(self, state):
        x_bucket = self.discretize_position(state.pos[0])
        y_bucket = self.discretize_position(state.pos[1])
        z_bucket = self.discretize_position(state.pos[2])

        # Create a unique key for the bucket
        return (state.courseNum, state.actNum, state.areaIndex, x_bucket, y_bucket, z_bucket)

    def add_state(self, state):
        key = self.get_bucket_key(state)
        self.add_circle(key, self.bucket_counts)

    def add_circle(self, key, bucket_dict):
        n, an, ai, x, y, z = key
        for i, j, k in itertools.product(list(range(-self.circle_radius, self.circle_radius + 1)), repeat=3):
            key = (n, an, ai, x+i, y+j, z+k)
            bucket_dict[key] += 1

    def discretize_position(self, value):
        # Define edges for digitization
        edges = np.arange(-8002, 8002, 1000 / (2*self.circle_radius + 1))

        # Use np.digitize to find the appropriate bucket index
        return np.digitize(value, edges)


if __name__ == "__main__":
    from LivePointCloud import LivePointCloud
    viewer = LivePointCloud()  # Opens browser tab

    pop_size = 100
    genome_length = 500
    step_amount = 1

    # Use the new CuriosityPopulation class
    scoring = CuriosityScoring()
    pop = Population(pop_size=pop_size, genome_length=genome_length, scoring=scoring)

    for i in range(1700):
        button = 0
        stickx = 0
        sticky = 0
        if (150 < i < 160) or (200 < i < 300):
            button = buttons.START_BUTTON
        if (i % 2 == 0):
            button = buttons.A_BUTTON

        pad = GamePad(button, stickx, sticky)
        pop.base_step(pad)

        base_state = pop.base_game.get_game_state()
        viewer.add_points([base_state.pos])

    while True:
        print(pop.iterate())

        pop.base_step(n=step_amount, viewer=viewer)
        write_gamepads_to_file(pop.base_path, filename=f"runs/curiosity-pop{pop_size}-genlen{genome_length}-step{step_amount}.m64")
