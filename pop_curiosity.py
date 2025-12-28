from population import Population, write_gamepads_to_file
from sm64_binds import GamePad, buttons
import numpy as np
from collections import defaultdict

class CuriosityScoring:
    # tmp_states are states between pop.base_game and self.temp_game that we do not want to finalise yet
    def __init__(self):
        # Initialize bucket counts using a dictionary for sparse representation
        self.bucket_counts = defaultdict(lambda: 0)

    def get_fitness(self, tmp_states):
        bucket_keys = [self.get_bucket_key(state) for state in tmp_states]
        
        tmp_bucket_counts = defaultdict(lambda: 0)

        scores = []
        for key in bucket_keys:
            visits = self.bucket_counts[key] + tmp_bucket_counts[key]
            scores.append(-visits)

            tmp_bucket_counts[key] += 1

        curiosity_score = np.median(np.array(scores))
        # curiosity_score = np.max(scores)
        return curiosity_score

    def get_bucket_key(self, state):
        x_bucket = self.discretize_position(state.pos[0])
        y_bucket = self.discretize_position(state.pos[1])
        z_bucket = self.discretize_position(state.pos[2])

        # Create a unique key for the bucket
        return (state.courseNum, state.actNum, state.areaIndex, x_bucket, y_bucket, z_bucket)

    def add_state(self, state):
        bucket_key = self.get_bucket_key(state)

        # Increment the count for the corresponding bucket
        if bucket_key not in self.bucket_counts:
            self.bucket_counts[bucket_key] = 0
        self.bucket_counts[bucket_key] += 1

    def discretize_position(self, value):
        # Define edges for digitization
        edges = np.arange(-8002, 8002, 1000)

        # Use np.digitize to find the appropriate bucket index
        return np.digitize(value, edges) - 1  # -1 to convert to zero-based index


if __name__ == "__main__":
    from LivePointCloud import LivePointCloud
    viewer = LivePointCloud()  # Opens browser tab

    pop_size = 1000
    genome_length = 50
    step_amount = 10

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
