games = {
    'TopGear2-Genesis': {
        'action_space': [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        ],
        'frame_shape': [73, 172],
        'training': {
            'dqn': {
                'total_episodes': 100,
                'max_steps': 50,
                'batch_size': 64
            },
            'pgn': {
                'total_episodes': 1000,
                'max_steps': 2000,
                'batch_size': 2000
            }
        },
        'render_format': [[10, 208], [0, 256]],
        'process_image_format': [[135, 208], [42, 214]]
    }
}

networks = {
    'dqn': {
        'stack_size': 4,
        'learning_rate': 0.0002,
        'explore_start': 1.0,
        'explore_stop': 0.01,
        'decay_rate': 0.00001,
        'gamma': 0.99,
        'memory_size': 16000
    },
    'pgn': {
        'stack_size': 4,
        'alpha_start': 0.001,
        'alpha_stop': 0.001,
        'decay_rate': 0.00001,
        'gamma': 0.95
    }
}