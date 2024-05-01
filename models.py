import tensorflow as tf

class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.d1(x)
        # return x
        return tf.math.sigmoid(x)

class TwoLayer(tf.keras.Model):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(d_hidden, activation='relu')
        self.d2 = tf.keras.layers.Dense(d_out)

    def call(self, x):
        x = self.d1(x)
        x = self.d2(x)
        return tf.math.sigmoid(x)


class SmallCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.c2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.mp2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.c1(x)
        x = self.mp1(x)
        x = self.c2(x)
        x = self.mp2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.dense(x)
        return tf.math.sigmoid(x)
    
class LargeCNN(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # self.c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))
        self.c1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.c2 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.mp1 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.25)
        self.c3 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.c4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.mp2 = tf.keras.layers.MaxPooling2D((2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.25)
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dropout3 = tf.keras.layers.Dropout(0.5)
        self.dense2 = tf.keras.layers.Dense(1)

    def call(self, x):
        x = self.c1(x)
        x = self.c2(x)
        x = self.mp1(x)
        x = self.dropout1(x)
        x = self.c3(x)
        x = self.c4(x)
        x = self.mp2(x)
        x = self.dropout2(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return tf.math.sigmoid(x)