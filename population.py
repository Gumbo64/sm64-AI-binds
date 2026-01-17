from sm64_binds import SM64Game, SM64GameGenerator, GamePad, buttons
import numpy as np

from LivePointCloud import LivePointCloud
from tqdm import tqdm
import time

def generate_gamepad_array(N):
    ints = np.random.randint(-128, 128, size=(N, 2))
    bools = np.random.choice([True, False], size=(N, 3), p=[0.04, 1-0.04])

    gamepad = np.empty((N, 5), dtype=object)
    gamepad[:, 0:2] = ints
    gamepad[:, 2:5] = bools

    return gamepad

def generate_population(pop_size, length):
    """
    Create an array of gamepad arrays using reshape.

    Returns
    -------
    np.ndarray
        Shape (M, N, 5)
    """
    flat = generate_gamepad_array(pop_size * length)
    return flat.reshape(pop_size, length, 5)

def pad_from_np(pad_np):
    stick_x = pad_np[0]
    stick_y = pad_np[1]

    button = 0
    if pad_np[2]:
        button |= buttons.A_BUTTON
    if pad_np[3]:
        button |= buttons.B_BUTTON
    if pad_np[4]:
        button |= buttons.Z_TRIG

    return GamePad(button, stick_x, stick_y)

import struct
import os
def write_gamepads_to_file(gamepads, filename="output.m64"):
    # Create the directory if it doesn't exist
    directory = os.path.dirname(filename)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "wb") as f:
        for gp in gamepads:
            data = struct.pack(
                ">Hb b",  # > = big-endian, H = uint16, b = int8
                gp.button,
                gp.stick_x,
                gp.stick_y
            )
            f.write(data)

import copy
class Scoring:
    # tmp_states are states between pop.base_game and self.temp_game that we do not want to finalise yet
    def get_fitness(self, tmp_states):
        scores = np.array([-state.pos[2] for state in tmp_states])

        i = np.argmax(scores)

        return scores[i] - i
    def add_state(self, state):
        pass


class Population:
    def __init__(self, pop_size=100, genome_length=50, mutation_rate=0.05, pop_elite=None, scoring=None, skip_menu=True):
        if scoring == None:
            scoring = Scoring()
        self.scoring = scoring
        self.pop_size = pop_size
        self.skip_menu = skip_menu

        if pop_elite is None:
            self.pop_elite = (pop_size // 5) + 1
        else:
            self.pop_elite = pop_elite

        self.genome_length = genome_length
        self.mutation_rate = mutation_rate


        self.pop = generate_population(self.pop_size, self.genome_length)

        rom_bytes = open("./baserom.us.z64", "rb").read()
        self.game_gen = SM64GameGenerator.new(rom_bytes)

        self.base_game = self.game_gen.create_game()
        self.base_path = []
        self.reset_base()


        self.temp_game = self.game_gen.create_game()

    def reset_base(self):
        self.base_game = self.game_gen.create_game()
        self.base_path = []
        if self.skip_menu:
            for i in range(1680):
                button = 0
                stickx = 0
                sticky = 0
                if (150 < i < 160) or (200 < i < 300):
                    button = buttons.START_BUTTON
                if (i % 2 == 0):
                    button = buttons.A_BUTTON
                pad = GamePad(button, stickx, sticky)
                self.base_step(pad)

    def base_step(self, pad=None, n=5, viewer=None):
        """
        Step the base game.

        If `pad` is provided, use it for a single step.
        Otherwise, use the first `n` pads of the fittest genome and then
        roll all genomes left by `n` and fill the end with new random pads.
        """
        if pad is not None:
            # Single step with provided pad
            self.base_game.step_game(pad)
            self.base_path.append(pad)
            return

        # --- Step with first n pads of fittest genome ---
        fittest_genome = self.pop[0]

        positions = []
        for i in range(n):
            step_pad_np = fittest_genome[i]
            step_pad = pad_from_np(step_pad_np)
            self.base_game.step_game(step_pad)
            self.base_path.append(step_pad)

            base_state = self.base_game.get_game_state()
            self.scoring.add_state(base_state)

            positions.append(base_state.pos)

        if viewer is not None:
            viewer.add_points(positions)


        # --- Roll all genomes left by n and fill new pads ---
        new_pads = generate_population(self.pop_size, n)  # shape: (pop_size, n, 5)
        self.pop = np.hstack([self.pop[:, n:, :], new_pads])

    def fitness(self, gamepad_array):
        self.game_gen.copy_to(self.base_game, self.temp_game)
        tmp_states = []

        for i, pad_np in enumerate(gamepad_array):
            pad = pad_from_np(pad_np)
            self.temp_game.step_game(pad)
            state = self.temp_game.get_game_state()
            tmp_states.append(state)

        return self.scoring.get_fitness(tmp_states)

    def iterate(self):
        fitnesses = []
        for pad_array in self.pop:
            fitnesses.append(self.fitness(pad_array))
        idx = np.argsort(fitnesses)[::-1]  # highest fitness first

        new_pop = np.empty_like(self.pop)

        # Elitism: copy top elite
        new_pop[:self.pop_elite] = self.pop[idx[:self.pop_elite]]

        # Generate offspring
        for i in range(self.pop_elite, self.pop_size):
            # Pick two parent IDs
            p1_id, p2_id = np.random.choice(self.pop_size, 2, replace=False)
            parent1 = self.pop[p1_id]
            parent2 = self.pop[p2_id]

            child = np.empty_like(parent1)

            # Probabilistic crossover per gene
            for j in range(self.genome_length):
                r = np.random.rand()
                if r < 0.4:
                    child[j] = parent1[j]  # 40% parent1
                elif r < 0.8:
                    child[j] = parent2[j]  # 40% parent2
                elif r < 0.9:
                    # 10% parent1 sticks + parent2 buttons
                    child[j, 0:2] = parent1[j, 0:2]
                    child[j, 2:5] = parent2[j, 2:5]
                else:
                    # 10% parent2 sticks + parent1 buttons
                    child[j, 0:2] = parent2[j, 0:2]
                    child[j, 2:5] = parent1[j, 2:5]

                # Mutation, completely random
                if np.random.rand() < self.mutation_rate:
                    child[j] = generate_gamepad_array(1)[0]

            new_pop[i] = child

        self.pop = new_pop
        return fitnesses[idx[0]]  # return best fitness


if __name__ == "__main__":
    from LivePointCloud import LivePointCloud
    viewer = LivePointCloud()  # opens browser tab

    pop_size = 1000
    genome_length = 50
    step_amount = 10

    pop = Population(pop_size=pop_size, genome_length=genome_length)

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
        write_gamepads_to_file(pop.base_path, filename=f"runs/pop{pop_size}-genlen{genome_length}-step{step_amount}.m64")


        
