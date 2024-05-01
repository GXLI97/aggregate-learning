import tensorflow as tf


class EasyLLPLog(tf.keras.losses.Loss):
    def __init__(self, K, p):
        super(EasyLLPLog, self).__init__()
        self.K = K
        self.p = p
        self.name = "EZLOG"
    
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
        self.name = "EZSQ"
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pos_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred-1), shape=(-1, self.K)), axis=1)
        neg_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred), shape=(-1, self.K)), axis=1)
        losses = (self.K*(alphas - self.p) + self.p) * pos_losses + (self.K*(self.p - alphas) + 1-self.p) * neg_losses
        return tf.reduce_mean(losses)
    
    def get_losses(self, y_true, y_pred):
        # auxiliary function for sq loss estimation experiment.
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pos_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred-1), shape=(-1, self.K)), axis=1)
        neg_losses = tf.reduce_mean(tf.reshape(tf.math.square(y_pred), shape=(-1, self.K)), axis=1)
        losses = (self.K*(alphas - self.p) + self.p) * pos_losses + (self.K*(self.p - alphas) + 1-self.p) * neg_losses
        return losses
    
class Square(tf.keras.losses.Loss):
    def __init__(self, K, p, bias=False):
        super(Square, self).__init__()
        self.K = K
        self.p = p
        self.bias = bias
        self.name = "SQ"
    
    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        T1 = self.K * tf.math.square(alphas - pred_props)
        T2 = self.K * tf.math.square(pred_props - self.p)
        T3 = tf.reduce_mean(tf.reshape(tf.math.square(y_pred - self.p), shape=(-1, self.K)), axis=1)
        if self.bias: losses = T1
        else: losses = T1 - T2 + T3
        return tf.reduce_mean(losses)
    
    
class DSQ_STREAM(tf.keras.losses.Loss):
    def __init__(self, K, p, beta=0.99):
        super(DSQ_STREAM, self).__init__()
        self.K = K
        self.p = p
        self.fhat = None
        self.beta = beta
        self.name = "DSQ_STREAM"

    def call(self, y_true, y_pred):
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        T1 = tf.reduce_mean(self.K * tf.math.square(alphas - pred_props))
        if not self.fhat: z = tf.reduce_mean(pred_props)
        else:
            z = self.beta * self.fhat + (1-self.beta) * tf.reduce_mean(pred_props)
        T2 = (self.K-1) * tf.math.square( z - self.p )
        loss = (T1 - T2)
        self.fhat = tf.constant(z)
        return loss
    
    def get_losses(self, y_true, y_pred):
        # auxiliary function for sq loss estimation experiment.
        alphas = tf.reduce_mean(tf.reshape(y_true, shape=(-1, self.K)), axis=1)
        pred_props = tf.reduce_mean(tf.reshape(y_pred, shape=(-1, self.K)), axis=1)
        T1 = tf.reduce_mean(self.K * tf.math.square(alphas - pred_props))
        T2 = (self.K-1) * tf.math.square( self.fhat - self.p )
        losses = self.K * tf.math.square(alphas - pred_props) - T2
        return losses


class Log(tf.keras.losses.Loss):
    def __init__(self, K):
        super(Log, self).__init__()
        self.name = "LOG"
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