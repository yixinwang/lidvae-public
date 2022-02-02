
params={
    'enc_type': 'lstm',
    'dec_type': 'lstm',
    'nz': 5,
    'ni': 50,
    'enc_nh': 50,
    'dec_nh': 50,
    'dec_dropout_in': 0.5,
    'dec_dropout_out': 0.5,
    'epochs': 100,
    'batch_size': 16,
    'test_nepoch': 1,
    'train_data': 'datasets/synthetic_data/synthetic_train.txt',
    'val_data': 'datasets/synthetic_data/synthetic_test.txt',
    'test_data': 'datasets/synthetic_data/synthetic_test.txt',
    'icnn_num_layers': 2,
    'icnn_nh': 128
}
