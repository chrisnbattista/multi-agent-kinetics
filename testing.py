import numpy as np
from . import potentials, worlds, indicators, forces

# Base setup
# '2d': ('t', 'id', 'm', 'b_1', 'b_2', 'v_1', 'v_2'),
gravity_test_world = worlds.World(
    initial_state=np.array([
        [0, 0, 10, 0, 0, 0, 0],
        [0, 1, 10, 5, 0, 0, 0]
    ])
)
G = 6.674*(10**(-11))

# Test 1. Gravity function
print("Test 1")
print(gravity_test_world.get_state())
print(forces.newtons_law_of_gravitation(gravity_test_world, G=G))

# Test 2. GPE function
print("Test 2")
print(potentials.gravitational_potential_energy(gravity_test_world, G=G))