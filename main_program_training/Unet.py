from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from VGG19 import *
from tensorflow.keras import initializers
from tensorflow.keras import regularizers


class ConvBlock(Model):

    def __init__(self, number_filters):
        super(ConvBlock, self).__init__()
        self.conv_1 = Conv2D(number_filters, kernel_size=(3,3), kernel_initializer=initializers.HeNormal(), padding="same")
        self.batch_1 = BatchNormalization(name='batch1')
        #self.activ_1 = Activation("relu")
        self.conv_2 = Conv2D(number_filters, kernel_size=(3,3), kernel_initializer=initializers.HeNormal(), padding="same")
        self.batch_2 = BatchNormalization(name='batch2')
        #self.activ_2 = Activation("relu")

    def call(self, input):
        x = self.conv_1(input)
        x = Activation("relu")(x)
        x = self.batch_1(x)
        x = self.conv_2(x)
        x = Activation("relu")(x)
        x = self.batch_2(x)
        return x

        # x = self.conv_1(input)
        # x = self.batch_1(x)
        # x = Activation("relu")(x)
        # x = self.conv_2(x)
        # x = self.batch_2(x)
        # x = Activation("relu")(x)
        # return x

class Encoder(Model):
    def __init__(self, number_filters):
        super(Encoder, self).__init__()
        self.convolution_block = ConvBlock(number_filters)
        self.maxpool = MaxPool2D(strides=(2,2))

    def call(self, input):
        skip_connect = self.convolution_block(input)
        reduced_fmap = self.maxpool(skip_connect)
        return skip_connect, reduced_fmap


class Decoder(Model):

    def __init__(self, number_filters):
        super(Decoder, self).__init__()

        self.transpose =  Conv2DTranspose(number_filters, kernel_size=(2, 2), kernel_initializer=initializers.HeNormal(),  strides=(2,2), padding="same")
        self.concat = Concatenate()
        self.convolution_block = ConvBlock(number_filters)

    def call(self, input, skip_connect):

        x = self.transpose(input)
        x = self.concat([x, skip_connect])
        x = self.convolution_block(x)
        return x


class Unet(Model):
    """In the VGG19-U-Net architecture the encoder blocks of the U-Net model is replaced
    with VGG19 model. """

    def __init__(self, pre_train_model):
        super(Unet, self).__init__()

        self.encoder = pre_train_model
        self.decoder_1 = Decoder(512)
        self.decoder_2 = Decoder(256)
        self.decoder_3 = Decoder(128)
        self.decoder_4 = Decoder(64)

        self.out = Conv2D(1,1, padding = "same", activation = 'sigmoid')

    def call(self, input, training=None, mask=None):

        s1, s2, s3, s4, b1 = self.encoder(input)
   
        d1 = self.decoder_1(b1, s4)
        d2 = self.decoder_2(d1, s3)
        d3 = self.decoder_3(d2, s2)
        d4 = self.decoder_4(d3, s1)

        outputs = self.out(d4)

        return outputs

