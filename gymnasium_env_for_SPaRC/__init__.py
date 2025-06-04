"""
This module initializes the Witness environment package.

It imports the `WitnessEnv` class and registers the environment for use with Gymnasium.

Modules:
    - gym_Witness: Contains the implementation of the Witness environment.
    - register_env: Handles the registration of the Witness environment with Gymnasium.
"""

from .gym_env_for_SPaRC import GymEnvSPaRC
from .register_env import *