import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import argparse
import time
from models import Linear, TwoLayer, SmallCNN, LargeCNN
from losses import EasyLLPLog, EasyLLPSquare, Square, DSQ_STREAM, Log, ClassificationLoss
import pickle


param = {
    'DATASET' : None,
    'SEED' : 10000,
    'BAG_SIZE' : None,
    'OPT': None,
    'LR' : None,
    'EPOCHS': None,
    'NUM_TRIALS': None,
    'BATCH_SIZE': None,
    'BETA': None
}


def get_model():
    if param['MODEL'] == 'linear': return Linear()
    elif param['MODEL'] == 'twolayer_small': return TwoLayer(d_in=1024, d_hidden=100, d_out=1)
    elif param['MODEL'] == 'twolayer_large': return TwoLayer(d_in=1024, d_hidden=1000, d_out=1)
    elif param['MODEL'] == 'cnn_small': return SmallCNN()
    elif param['MODEL'] == 'cnn_large': return LargeCNN()


def get_optimizer():
    if param['OPT'] == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate = param['LR'])
    else:
        return tf.keras.optimizers.Adam(learning_rate = param['LR'])


def train_step(model, x_data, y_data, loss_object, optimizer, train_metric):
    with tf.GradientTape() as tape:
        predictions = model(x_data, training=True)
        loss = loss_object(y_data, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_metric.update_state(loss)


def test_step(model, x_data, y_data, loss_object, test_metric):
    predictions = model(x_data, training=False)
    test_loss = loss_object(y_data, predictions)
    test_metric(test_loss)


def sq_estimate_error(model, x_data, y_data, loss_object):
    predictions = tf.squeeze(model(x_data, training=False))
    sq_losses = tf.math.square(y_data - predictions)
    bag_sq_loss_true = tf.reduce_mean(tf.reshape(sq_losses, shape=(-1, param['BAG_SIZE'])), axis=1) # gets the average square loss for a bag.
    bag_loss_estimates = loss_object.get_losses(y_data, predictions)
    return bag_sq_loss_true, bag_loss_estimates
                                            

def single_run(train_ds, test_ds, loss_object, loss_class, train_loss, test_loss):
    start = time.time()

    tf.keras.utils.set_random_seed(param['SEED'])

    cls = []
    bag_true_data = []
    bag_est_data = []
    
    for trial in range(param['NUM_TRIALS']):
        model = get_model()
        optimizer = get_optimizer()

        outcome = []
        bag_losses_true = []
        bag_losses_est = []
        for e in range(param['EPOCHS']):
            train_loss.reset_state()
            test_loss.reset_state()
            for x_train, y_train in train_ds:
                train_step(model, x_train, y_train, loss_object, optimizer, train_loss)
            for x_test, y_test in test_ds:
                test_step(model, x_test, y_test, loss_class, test_loss)
                bt, be = sq_estimate_error(model, x_test, y_test, loss_object)
                bag_losses_true.append(bt)
                bag_losses_est.append(be)
            print(f'Epoch {e+1}', f'Train Loss: {train_loss.result()}', f'Test Loss: {test_loss.result()}')
            print(f'Bag Est Difference: ', tf.reduce_mean(bt-be))
            outcome.append(test_loss.result())
        
        cls.append(outcome)
        bag_true_data.append(bag_losses_true)
        bag_est_data.append(bag_losses_est)

    # Saving data for Sq Loss Tracking Experiment
    # with open(f"./results/{param['DATASET']}/bag_est_{param['MODEL']}_batchsize{param['BATCH_SIZE']}_{param['METHOD']}.pickle", 'wb') as handle:
    #     pickle.dump({'true': bag_true_data, 'est': bag_est_data}, handle)

    end = time.time()
    print("Elapsed Time in Sec: ", end-start)
    return cls

def run_method(train_ds, test_ds, p_hat, train_loss, test_loss):
    if param['METHOD'] == 'ezlog':
        print("EasyLLP with Log Loss")    
        cls_error = single_run(train_ds, test_ds,\
                            EasyLLPLog(param['BAG_SIZE'], p=p_hat),\
                            ClassificationLoss(),\
                            train_loss, test_loss)
    elif param['METHOD'] == 'ezsq':
        print("EasyLLP with Sq Loss")
        cls_error = single_run(train_ds, test_ds,\
                            EasyLLPSquare(param['BAG_SIZE'], p=p_hat),\
                            ClassificationLoss(),\
                            train_loss, test_loss)
    elif param['METHOD'] == 'sq':
        print("Sq Loss, No Bias Correction")
        cls_error = single_run(train_ds, test_ds,\
                            Square(param['BAG_SIZE'], p=p_hat, bias=True),\
                            ClassificationLoss(),\
                            train_loss, test_loss)
    elif param['METHOD'] == 'dsq_stream':
        print("Sq Loss, Debiased with Streaming")
        cls_error = single_run(train_ds, test_ds,\
                        DSQ_STREAM(param['BAG_SIZE'], p=p_hat, beta=param['BETA']),\
                        ClassificationLoss(),\
                        train_loss, test_loss)
    elif param['METHOD'] == 'log':
        print("Log Loss")
        cls_error = single_run(train_ds, test_ds,\
                            Log(param['BAG_SIZE']),\
                            ClassificationLoss(),\
                            train_loss, test_loss)

    return cls_error

def mnist_odd_even():
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train = np.expand_dims(X_train, axis=-1)
    X_train = tf.image.resize(X_train, [32,32]).numpy()
    X_test = np.expand_dims(X_test, axis=-1)
    X_test = tf.image.resize(X_test, [32,32]).numpy()
    if param['MODEL'] in ['linear', 'twolayer_small', 'twolayer_large']:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    X_train = X_train / 255.
    X_test = X_test / 255.

    Y_train_oe = (Y_train % 2 == 0).astype(np.int32)
    tf.keras.utils.set_random_seed(param['SEED'])
    Y_test_oe = (Y_test % 2 == 0).astype(np.float32)

    p_hat = np.mean(Y_train_oe)
    p_hat_test = np.mean(Y_test_oe)
    print("P HAT:", p_hat, p_hat_test)

    tf.keras.utils.set_random_seed(param['SEED'])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train_oe))\
                    .shuffle(param['SEED'], reshuffle_each_iteration=False).batch(param['BATCH_SIZE'])
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test_oe))\
                    .batch(10000)
    
    return run_method(train_ds, test_ds, p_hat, train_loss, test_loss)


def cifar_animal_machine():
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.cifar10.load_data()

    if param['MODEL'] in ['linear', 'twolayer_small', 'twolayer_large']:
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    

    X_train = X_train / 255.
    X_test = X_test / 255.
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']
    animals = [2, 3, 4, 5, 6, 7]
    # we assign label of 1 to animals.
    Y_train_am = np.logical_and(Y_train >= 2, Y_train <= 7).astype(np.int32)
    tf.keras.utils.set_random_seed(param['SEED'])

    Y_test_am = np.logical_and(Y_test >= 2, Y_test <= 7).astype(np.float32)

    p_hat = np.mean(Y_train_am)


    tf.keras.utils.set_random_seed(param['SEED'])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, Y_train_am))\
                    .shuffle(param['SEED'], reshuffle_each_iteration=False).batch(param['BATCH_SIZE'])
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, Y_test_am))\
                    .batch(10000)
    
    return run_method(train_ds, test_ds, p_hat, train_loss, test_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('DATASET')
    parser.add_argument('METHOD')
    parser.add_argument('MODEL')
    parser.add_argument('NUM_TRIALS', type=int)
    parser.add_argument('EPOCHS', type=int)
    parser.add_argument('BAG_SIZE', type=int)
    parser.add_argument('OPT')
    parser.add_argument('LR', type=float)
    # optional parameters: for beta variation experiment.
    parser.add_argument('BATCH_SIZE', nargs="?", default=1000, type=int)
    parser.add_argument('BETA', nargs="?", default=0.99, type=float)
    

    args = parser.parse_args()
    param['DATASET'] = args.DATASET
    param['METHOD'] = args.METHOD
    param['MODEL'] = args.MODEL
    param['NUM_TRIALS'] = args.NUM_TRIALS
    param['EPOCHS'] = args.EPOCHS
    param['BAG_SIZE'] = args.BAG_SIZE
    param['OPT'] = args.OPT
    param['LR'] = args.LR
    param['BATCH_SIZE'] = args.BATCH_SIZE
    param['BETA'] = args.BETA

    print(param, '\n')

    if param['DATASET'] == 'MNIST': cls_error = mnist_odd_even()
    if param['DATASET'] == 'CIFAR': cls_error = cifar_animal_machine()
    
    # Save classification error data.
    with open(f"./results/{param['DATASET']}/{param['METHOD']}_{param['MODEL']}_bagsize{param['BAG_SIZE']}.pickle", 'wb') as handle:
        pickle.dump(cls_error, handle)
