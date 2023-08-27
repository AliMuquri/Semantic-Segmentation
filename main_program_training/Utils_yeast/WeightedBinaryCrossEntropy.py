import tensorflow as tf

class WeightedBinaryCrossEntropy(tf.keras.losses.Loss):

        def __init__(self, weight_back, weight_front):
            super().__init__()
            self.weight0 = weight_back
            self.weight1 = weight_front

        def call(self, y_true, y_pred):
            bin_cross = tf.keras.losses.binary_crossentropy(y_true, y_pred)
            weights = y_true * self.weight1  + (1-y_true) * self.weight0
            wbce = weights * bin_cross
            return wbce
