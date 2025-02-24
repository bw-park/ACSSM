import ml_collections


def get_physionet_extrapolation_configs():
    config = ml_collections.ConfigDict()
    
    config.random_seed = 42
    config.data_random_seed = 0
    config.T = 100

    config.dataset = 'physionet'
    config.task = 'extrapolation'
    config.info_type = 'history'
    config.num_workers = 8
    config.pin_memory = True
    config.ts = 0.2
    config.lamda_1 = 1e-6
    config.lamda_2 = 1e-8

    config.epochs = 200
    config.lr = 1e-3
    config.wd = 0
    config.batch_size = 100
    config.num_basis = 20
    config.state_dim = 24
    config.n_layer = 4
    config.drop_out = 0.
    config.init_sigma = 10.

    config.out_dim = 37
    config.cut_time = 24
    
    return config