splitting_config = {
    'test_ratio': 0.2
}

clustering_config = {
    'elbow_method': {
        'first': 500,
        'last': 2000,
        'step_size': 500
    },
    'percentage': {
        'first': 0,
        'last': 10,
        'step_size': 2
    },
    'threshold': {
        'first': 1,
        'step_size': 1
    }
}

predictor_config = {
    'n_neighbors': [20, 15, 10, 5, 3, 1]
}