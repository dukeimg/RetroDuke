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
                'total_episodes': 101,
                'max_steps': 50,
                'batch_size': 64
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
    }
}