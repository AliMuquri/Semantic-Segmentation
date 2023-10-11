# Semantic segmentation

## Project Description
In this repo I showcase how to utilize labelme to turn jsons annotations created by labelme into a annotation mask. This customized approach differs from the one provide by labelme since it can take the an input of images and jsons in a VOC structure and output images and image annotation the same structure. To speed up the process subprocessing is used.

In this repo I also showcase how to build a VGG19-U-Net with a object-oriented model (thus avoiding the functional API), moreover.Furthermore this project also utlize transfer learning and dataprocessing and data augmentation by a datapipline.
In a different branch I also show how to migrate a generator based on TF Sequence to TF Data. I also show how to create callbacks with live visual graphs of the loss, accuracy and predictions masks.

## Getting started

First create a virtual environment.
Then install the packages in requirments.txt:
pip install -r requirments.txt

## Usage
** main_program_voc **:

Place you annonated dataset here (jpg and json files in the same folder). Only single folder or VOC strucutre (train, test, val) is allowed.

To run the to conversion program use this command:
py main_program.py -V input_path output_path


Rembember to activate the virtual environment first since it runs on labelme.


** main_program_training **
Make appropriate changes to the datapipeline and hyperparameters.Otherwise after created the dataset voc directory. 
Run the main_program.py
 

