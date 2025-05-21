from gymnasium.envs.registration import register
'''
This script registers the custom environment "Witness-v0" with Gymnasium.
'''
register(
    id="Witness-v0",  # Unique identifier for your environment
    entry_point="gymnasium_env.gym_Witness:WitnessEnv",  # Path to your environment class
)