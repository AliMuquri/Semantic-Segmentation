from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import utils
import os

WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                       'releases/download/v0.1/'
                       'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

weights_path = utils.get_file(
            'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5',
            WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='253f8cb515780f3b799900260a226db6')

class VGG19(Model):
    """ This VGG19 model follows the architecture given by
    https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py
    The call model has been altered to accomedate output for the skip connection in the VGG19-U-Net architecture.

    """
    def __init__(self):
        super(VGG19, self).__init__()

        self.block1_conv1 = layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv1')
        self.block1_conv2= layers.Conv2D(64, (3, 3),
                            activation='relu',
                            padding='same',
                            name='block1_conv2')

        self.block1_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')


        self.block2_conv1 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv1')

        self.block2_conv2 = layers.Conv2D(128, (3, 3),
                          activation='relu',
                          padding='same',
                          name='block2_conv2')

        self.block2_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')

    # Block 3
        self.block3_conv1 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')
        self.block3_conv2 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')
        self.block3_conv3 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')
        self.block3_conv4 = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')
        self.block3_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')

    # Block 4
        self.block4_conv1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')
        self.block4_conv2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')
        self.block4_conv3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')
        self.block4_conv4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')
        self.block4_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')

    # Block 5
        self.block5_conv1 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')
        self.block5_conv2 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')
        self.block5_conv3 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')
        self.block5_conv4 = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')
        self.block5_pool = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

    def call(self, input, training=None):
        x=self.block1_conv1(input)
        s1=self.block1_conv2(x)
        x=self.block1_pool(s1)
        x=self.block2_conv1(x)
        s2=self.block2_conv2(x)
        x=self.block2_pool(s2)
        x=self.block3_conv1(x)
        x=self.block3_conv2(x)
        x=self.block3_conv3(x)
        s3=self.block3_conv4(x)
        x=self.block3_pool(s3)
        x=self.block4_conv1(x)
        x=self.block4_conv2(x)
        x=self.block4_conv3(x)
        s4=self.block4_conv4(x)
        x=self.block4_pool(s4)
        x=self.block5_conv1(x)
        x=self.block5_conv2(x)
        x=self.block5_conv3(x)
        b1=self.block5_conv4(x)
        x=self.block5_pool(b1)

        return s1,s2,s3,s4,b1

    def load(self):
        """ This method loads the pre trained weights from Tensorflow."""
        weights_filename = 'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5'
        cache_subdir = 'models'
        file_hash = '253f8cb515780f3b799900260a226db6'

        # Check if the weights file already exists
        if not os.path.exists(os.path.join(cache_subdir, weights_filename)):
            WEIGHTS_PATH_NO_TOP = ('https://github.com/fchollet/deep-learning-models/'
                                'releases/download/v0.1/'
                                'vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')

            # Download the weights if they don't exist
            weights_path = utils.get_file(
                weights_filename,
                WEIGHTS_PATH_NO_TOP,
                cache_subdir=cache_subdir,
                file_hash=file_hash
            )

            print("Weights downloaded and saved at:", weights_path)
        else:
            print("Weights file already exists. Skipping download.")
        
        self.load_weights(weights_path)
