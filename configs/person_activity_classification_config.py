import ml_collections


def get_person_activity_classification_configs():
    config = ml_collections.ConfigDict()
    
    config.random_seed = 42
    config.data_random_seed = 0
    config.T = 100

    config.dataset = 'person_activity'
    config.task = 'classification'
    config.info_type = 'full'
    config.num_workers = 8
    config.pin_memory = True
    config.ts = 1/221
    config.lamda_1 = 1e-6
    config.lamda_2 = 1e-8

    config.epochs = 400
    config.lr = 1e-3
    config.wd = 1e-2     # 0
    config.batch_size = 256
    config.num_basis = 256
    config.state_dim = 288
    config.n_layer = 1
    config.drop_out = 0.2
    config.init_sigma = 1.

    config.out_dim = 7

    
    return config