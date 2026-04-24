"""
Central configuration constants for the Sociogenesis multi-agent system.
"""

# Simulation settings
N_AGENTS = 10
MAX_TICKS = 1000
TICK_SLEEP = 0  # Seconds to sleep between ticks

# Economy settings
START_TOKENS = 100
TICK_COST = 1
EARN_REWARD_LOW = 3
EARN_REWARD_HIGH = 5

# Communication settings
BUS_TIMEOUT = 0.001

# Registry settings
REPUTATION_START = 1.0
REPUTATION_MIN = 0.1
REPUTATION_MAX = 10.0
FINGERPRINT_DIM = 128
