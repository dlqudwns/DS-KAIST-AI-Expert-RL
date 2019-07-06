from gym.envs.registration import register

register(
    id='maze-v0',
    entry_point='envs.maze_env:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-sample-5x5-v0',
    entry_point='envs.maze_env:MazeEnvSample5x5',
    timestep_limit=2000,
)

register(
    id='maze-random-5x5-v0',
    entry_point='envs.maze_env:MazeEnvRandom5x5',
    timestep_limit=2000,
    nondeterministic=True,
)

register(
    id='maze-sample-10x10-v0',
    entry_point='envs.maze_env:MazeEnvSample10x10',
    timestep_limit=10000,
)

register(
    id='maze-random-10x10-v0',
    entry_point='envs.maze_env:MazeEnvRandom10x10',
    timestep_limit=10000,
    nondeterministic=True,
)

register(
    id='maze-sample-3x3-v0',
    entry_point='envs.maze_env:MazeEnvSample3x3',
    timestep_limit=1000,
)

register(
    id='maze-random-3x3-v0',
    entry_point='envs.maze_env:MazeEnvRandom3x3',
    timestep_limit=1000,
    nondeterministic=True,
)


register(
    id='maze-sample-100x100-v0',
    entry_point='envs.maze_env:MazeEnvSample100x100',
    timestep_limit=1000000,
)

register(
    id='maze-random-100x100-v0',
    entry_point='envs.maze_env:MazeEnvRandom100x100',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-10x10-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom10x10Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-20x20-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom20x20Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)

register(
    id='maze-random-30x30-plus-v0',
    entry_point='envs.maze_env:MazeEnvRandom30x30Plus',
    timestep_limit=1000000,
    nondeterministic=True,
)
