from gymnasium.envs.registration import register

register(
    id="Witness-v0",  # Unique identifier for your environment
    entry_point="gymnasium_env.gym_Witness:WitnessEnv",  # Path to your environment class
)