import numpy as np
import pickle

models = ['linear', 'twolayer_small', 'twolayer_large', 'cnn_small', 'cnn_large']
bag_sizes = [10, 100, 1000]
DATASET = 'MNIST'
param = {
    'DATASET': 'MNIST',
    'MODEL': None,
    'BAG_SIZE': 10
}

for bag in bag_sizes:
    for model in models:
        param['MODEL'] = model
        with open(f'./results/{DATASET}/{model}{bag}.pickle', 'rb') as handle:
            ret = pickle.load(handle)

        print()
        print(param)
        print('easyllplog: ', 100 * ret['test_loss_EZLOG'][1], '\t learning rate: ', ret['test_loss_EZLOG'][0])
        print('easyllpsq: ', 100 * ret['test_loss_EZSQ'][1], '\t learning rate: ', ret['test_loss_EZSQ'][0])
        print('log: ', 100 * ret['test_loss_LOG'][1], '\t learning rate: ', ret['test_loss_LOG'][0])
        print('sq: ', 100 * ret['test_loss_SQ'][1], '\t learning rate: ', ret['test_loss_SQ'][0])
        print('dsq_stream: ', 100 * ret['test_loss_DSQ_STREAM_V2'][1], '\t learning rate: ', ret['test_loss_DSQ_STREAM_V2'][0])
