import ml_collections


def get_ushcn_interpolation_configs():
    config = ml_collections.ConfigDict()
    
    config.random_seed = 42
    config.data_random_seed = 0
    config.T = 100

    config.dataset = 'ushcn'
    config.task = 'interpolation'
    config.info_type = 'full'
    config.num_workers = 8
    config.pin_memory = True
    config.ts = 0.3
    config.lamda_1 = 1e-6
    config.lamda_2 = 1e-8
    
    config.sample_rate = 0.5
    config.unobserved_rate = 0.2
    config.epochs = 500 #100
    config.lr = 1e-3
    config.wd = 0
    config.batch_size = 50
    config.num_basis = 20
    config.state_dim = 20
    config.n_layer = 4
    config.drop_out = 0.
    config.init_sigma = 10.

    config.out_dim = 5

    return config