import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import tensorflow as tf
import argparse
import time

param = {
    'SEED' : 2,
    'BAG_SIZE' : None,
    'OPT': None,
    'LR' : None,
    'EPOCHS': 100,
    'LABEL_NOISE': 0.0,
    'NUM_TRIALS': None,
}

class TwoLayer(tf.keras.Model):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.d2 = tf.keras.layers.Dense(d_out)

    # @tf.function
    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return tf.math.sigmoid(x)
    
    # @tf.function
    def train_step(self, x_data, y_data, loss_object, optimizer, train_metric):
        with tf.GradientTape() as tape:
            predictions = self(x_data, training=True)
            loss = loss_object(y_data, predictions)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        train_metric.update_state(loss)

    # @tf.function
    def test_step(self, x_data, y_data, loss_object, test_metric):
        predictions = self(x_data, training=False)
        test_loss = loss_object(y_data, predictions)
        test_metric(test_loss)

class EasyLLPLog(tf.keras.losses.Loss):
    def __init__(self, K, p):
        super(EasyLLPLog, self).__init__()
        self.K = K
        self.p = p
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pos_losses = tf.reduce_mean(tf.reshape(-tf.math.log(y_pred+1e-8), shape=(-1, self.K)), axis=1)
        neg_losses = tf.reduce_mean(tf.reshape(-tf.math.log(1 - y_pred+1e-8), shape=(-1, self.K)), axis=1)
        losses = (self.K*(alphas - self.p) + self.p) * pos_losses + (self.K*(self.p - alphas) + 1-self.p) * neg_losses
        return tf.reduce_mean(losses)

class EasyLLPSquare(tf.keras.losses.Loss):
    def __init__(self, K, p):
        super(EasyLLPSquare, self).__init__()
        self.K = K
        self.p = p
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pos_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred-1), shape=(-1, self.K)), axis=1)
        neg_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred), shape=(-1, self.K)), axis=1)
        losses = (self.K*(alphas - self.p) + self.p) * pos_losses + (self.K*(self.p - alphas) + 1-self.p) * neg_losses
        return tf.reduce_mean(losses)
    
class Square(tf.keras.losses.Loss):
    def __init__(self, K, p, bias=False):
        super(Square, self).__init__()
        self.K = K
        self.p = p
        self.bias = bias
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        T1 = self.K * tf.math.square(alphas - pred_props)
        T2 = self.K * tf.math.square(pred_props - self.p)
        T3 = tf.reduce_mean(tf.reshape(tf.math.square(y_pred - self.p), shape=(-1, self.K)), axis=1)
        if self.bias: losses = T1
        else: losses = T1 - T2 + T3
        return tf.reduce_mean(losses)
    
class ExpectedSquare(tf.keras.losses.Loss):
    def __init__(self, K, p):
        super(ExpectedSquare, self).__init__()
        self.K = K
        self.p = p
        self.error_est = 0
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        T1 = tf.reduce_mean(self.K * tf.math.square(alphas - pred_props))
        T2 = (self.K-1) * tf.math.square( tf.reduce_mean(pred_props - self.p) )

        T1R = tf.reduce_mean(self.K * tf.math.square(1- alphas - pred_props))
        T2R = (self.K-1) * tf.math.square( tf.reduce_mean(pred_props + self.p  - 1))
        loss = (1-self.error_est) * (T1 - T2) + (self.error_est) * (1 - T1R + T2R) 
        return loss

class Log(tf.keras.losses.Loss):
    def __init__(self, K):
        super(Log, self).__init__()
        self.K = K
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        correction = tf.reduce_mean(-alphas * tf.math.log(alphas+1e-8) - (1-alphas)*tf.math.log(1-alphas+1e-8))
        return tf.reduce_mean(-alphas * tf.math.log(pred_props+1e-8) - (1-alphas) * tf.math.log(1-pred_props+1e-8)) - correction


class ClassificationLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        y_true_cls = tf.reshape(tf.cast(y_true > 0.5, tf.int64), [-1])
        y_pred_cls = tf.reshape(tf.cast(y_pred > 0.5, tf.int64), [-1])
        equality = tf.math.equal(y_true_cls, y_pred_cls)
        return tf.reduce_mean(tf.cast(~equality, tf.float32))


def single_run(train_ds, test_ds, loss_object, loss_class, train_loss, test_loss):
    start = time.time()

    tf.keras.utils.set_random_seed(param['SEED'])
    model = TwoLayer(d_in=784, d_hidden=100, d_out=1)
    if param['OPT'] == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = param['LR'])
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate = param['LR'])

    cls = []

    for trial in range(param['NUM_TRIALS']):
        outcome = []
        for e in range(param['EPOCHS']):
            train_loss.reset_states()
            test_loss.reset_states()
            for x_train, y_train in train_ds:
                model.train_step(x_train, y_train, loss_object, optimizer, train_loss)
            for x_test, y_test in test_ds:
                model.test_step(x_test, y_test, loss_class, test_loss)
            # if hasattr(loss_object, 'error_est'):
            #     loss_object.error_est = test_loss.result()
            print(f'Epoch {e+1}', f'Train Loss: {train_loss.result()}', f'Test Loss: {test_loss.result()}')
            outcome.append(test_loss.result())
        cls.append(outcome)
    
    end = time.time()
    print("Elapsed Time in Sec: ", end-start)
    return cls

def mnist_odd_even():
    train_loss = tf.keras.metrics.Mean()
    test_loss = tf.keras.metrics.Mean()

    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()
    X_train_pre = X_train.reshape(X_train.shape[0], -1)
    X_test_pre = X_test.reshape(X_test.shape[0], -1)
    X_train_pre = X_train_pre / 255.
    X_test_pre = X_test_pre / 255.

    Y_train_oe = (Y_train % 2 == 0).astype(np.int32)
    tf.keras.utils.set_random_seed(param['SEED'])
    noise = np.random.binomial(1, param['LABEL_NOISE'], Y_train_oe.shape)
    Y_train_oe_noisy = ((Y_train_oe + noise) % 2).astype(np.float32)

    Y_test_oe = (Y_test % 2 == 0).astype(np.float32)

    p_hat = np.mean(Y_train_oe_noisy)

    tf.keras.utils.set_random_seed(param['SEED'])
    train_ds = tf.data.Dataset.from_tensor_slices((X_train_pre, Y_train_oe_noisy))\
                    .shuffle(param['SEED'], reshuffle_each_iteration=False).batch(60000)
                    # seems like this makes it train faster than doing single batch.
                    # .shuffle(param['SEED'], reshuffle_each_iteration=False).batch(param['BAG_SIZE'])
    test_ds = tf.data.Dataset.from_tensor_slices((X_test_pre, Y_test_oe))\
                    .batch(10000)
    
    print("EasyLLP with Log Loss")
    cls_easyllplog = None    
    cls_easyllplog = single_run(train_ds, test_ds,\
                        EasyLLPLog(param['BAG_SIZE'], p=p_hat),\
                        ClassificationLoss(),\
                        train_loss, test_loss)

    print("EasyLLP with Sq Loss")
    cls_easyllpsq = None
    # cls_easyllpsq = single_run(train_ds, test_ds,\
    #                     EasyLLPSquare(param['BAG_SIZE'], p=p_hat),\
    #                     ClassificationLoss(),\
    #                     train_loss, test_loss)

    print("Sq Loss with Bias Correction")
    cls_sq = single_run(train_ds, test_ds,\
                        Square(param['BAG_SIZE'], p=p_hat),\
                        ClassificationLoss(),\
                        train_loss, test_loss)
    
    print("Sq Loss, No Bias Correction")
    cls_sq_biased = single_run(train_ds, test_ds,\
                        Square(param['BAG_SIZE'], p=p_hat, bias=True),\
                        ClassificationLoss(),\
                        train_loss, test_loss)
    
    print("Sq Loss, Expected Bias Correction")
    cls_sq_expected = single_run(train_ds, test_ds,\
                    ExpectedSquare(param['BAG_SIZE'], p=p_hat),\
                    ClassificationLoss(),\
                    train_loss, test_loss)

    print("Log Loss")
    cls_log = None
    cls_log = single_run(train_ds, test_ds,\
                        Log(param['BAG_SIZE']),\
                        ClassificationLoss(),\
                        train_loss, test_loss)
    
    return cls_easyllplog, cls_easyllpsq, cls_sq, cls_sq_biased, cls_sq_expected, cls_log

def plot_mnist_odd_even(cls_easyllplog, cls_easyllpsq, cls_sq, cls_sq_biased, cls_sq_expected, cls_log):
    plt.figure(figsize=(10,10))
    xaxis = [e+1 for e in range(param['EPOCHS'])]

    def plt_bands(data, label):
        if not data: return
        data = np.array(data)
        lower = np.percentile(data, 5, axis=0)
        upper = np.percentile(data, 95, axis=0)
        avg = np.mean(data, axis=0)
        plt.plot(xaxis, avg, label=label)
        plt.fill_between(xaxis, lower, upper, alpha=.1)

    plt_bands(cls_easyllplog, 'EasyLLP w/ Log Loss')
    plt_bands(cls_easyllpsq, 'EasyLLP w/ Sq Loss')
    plt_bands(cls_sq, 'Sq Loss')
    plt_bands(cls_sq_biased, 'Biased Sq Loss')
    plt_bands(cls_sq_expected, 'Expected Sq Loss')
    plt_bands(cls_log, 'Aggregate Log Loss')

    plt.ylim(0, 0.51)
    plt.legend()
    plt.title(f"MNIST Odd vs Even \n Bag Size = {param['BAG_SIZE']} \n {param['OPT']} with LR = {param['LR']} \n Avg over {param['NUM_TRIALS']} Runs")
    plt.savefig(f"/home-nfs/gene/aggregate_learning/figures/v5/bag{param['BAG_SIZE']}-{param['OPT']}-lr{param['LR']}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('BAG_SIZE', type=int)
    parser.add_argument('OPT')
    parser.add_argument('LR', type=float)
    parser.add_argument('NUM_TRIALS', type=int)
    parser.add_argument('LABEL_NOISE', type=float)

    args = parser.parse_args()
    param['BAG_SIZE'] = args.BAG_SIZE
    param['OPT'] = args.OPT
    param['LR'] = args.LR
    param['NUM_TRIALS'] = args.NUM_TRIALS
    param['LABEL_NOISE'] = args.LABEL_NOISE

    cls_easyllplog, cls_easyllpsq, cls_sq, cls_sq_biased, cls_sq_expected, cls_log = mnist_odd_even()
    plot_mnist_odd_even(cls_easyllplog, cls_easyllpsq, cls_sq, cls_sq_biased, cls_sq_expected, cls_log)
