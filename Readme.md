#semantic segmentation

In this repo I showcase how to utilize labelme to turn jsons annotations created by labelme into a annotation mask. This customized approach differs from the one provide by labelme since it can take the an input of images and jsons in a VOC structure and output images and image annotation the same structure. To speed up the process subprocessing is used.

In this repo I also showcase how to build a VGG19-U-Net with a object-oriented model (thus avoiding the functional API), moreover, I also show how migrate a generator based on TF Sequence to TF Data. I also show how to create callbacks with live visual graphs of the loss, accuracy and predictions masks.
