import random

import struct

import numpy as np

import os

from gameplay import GAMEPLAY
from sm64_binds import SM64Game, SM64GameGenerator, GamePad, buttons


import plotly.graph_objects as go

from LivePointCloud import LivePointCloud

rom_bytes = open("./baserom.us.z64", "rb").read()

generator = SM64GameGenerator.new(rom_bytes)
game = generator.create_game()

viewer = LivePointCloud()  # opens browser tab

for i in range(5000):
    pad = GamePad(0, 0, 0)

    if 150 < i < 160 or 200 < i < 300:
        pad.button = buttons.START_BUTTON

    if i > 300:
        pad.stick_y = 80

    if i % 2 == 0:
        pad.button = buttons.A_BUTTON

    # Step
    game.step_game(pad)

    # Read state
    state = game.get_game_state()

    color = min(np.sqrt((state.vel[0]*state.vel[0] + state.vel[1] * state.vel[1] + state.vel[2] * state.vel[2])) / 100, 1)
    print(color)
    viewer.add_points([state.pos], colors=[color])
    print(state.pos)

