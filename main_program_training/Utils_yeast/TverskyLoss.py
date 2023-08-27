import tensorflow as tf

class TverskyLoss(tf.keras.losses.Loss):

    def __init__(self, beta, smooth=1):
        super().__init__()
        self.beta = beta
        self.smooth=smooth

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        numerator = tf.reduce_sum(y_true * y_pred) + self.smooth
        denominator = tf.reduce_sum(y_true * y_pred) + (1-self.beta) * tf.reduce_sum((1 - y_true) * y_pred) + self.beta * tf.reduce_sum(y_true * (1 - y_pred)) + self.smooth
        tversky = 1 - numerator/denominator
        return tversky
