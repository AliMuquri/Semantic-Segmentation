from tensorflow.keras.callbacks import Callback
import matplotlib.pyplot as plt
import subprocess
import os
import numpy as np
import time
import shutil
import random

class CallbackVisual(Callback):

    def __init__(self, data, path, load_weights, old_previous_run):
        
        self.data = data
        self.main_path = path
        self.load_weights = load_weights
        self.old_previous_run = old_previous_run

        #If an old checkpoint is used old paths are set to continue with correct graphs and animation.
        if load_weights:
            self.old_folder_path_img = os.path.join(self.old_previous_run, "ImgArrays")
            self.old_folder_path_loss = os.path.join(self.old_previous_run, "LossArrays")
            self.old_folder_path_saved = os.path.join(self.old_previous_run, "SAVED")
            self.old_folder_path_acc = os.path.join(self.old_previous_run, "AccArrays")
            self.old_text_path = os.path.join(self.old_folder_path_img, "title.txt")
            self.old_val_loss_path = os.path.join(self.old_folder_path_loss, "val_loss.txt")
            self.old_train_loss_path = os.path.join(self.old_folder_path_loss, "train_loss.txt")
            self.old_val_acc_path = os.path.join(self.old_folder_path_acc, "val_acc.txt")
            self.old_train_acc_path = os.path.join(self.old_folder_path_acc, "train_acc.txt")

        self.folder_path_img = os.path.join(self.main_path, "ImgArrays")
        self.folder_path_loss = os.path.join(self.main_path, "LossArrays")
        self.folder_path_saved = os.path.join(self.main_path, "SAVED")
        self.folder_path_acc = os.path.join(self.main_path, "AccArrays")

        #If folders to save training history do not exist in the main run folder they are created.
        if not os.path.isdir(self.folder_path_img):
            os.mkdir(self.folder_path_img)

        if not os.path.isdir(self.folder_path_loss):
            os.mkdir(self.folder_path_loss)

        if not os.path.isdir(self.folder_path_saved):
            os.mkdir(self.folder_path_saved)

        if not os.path.isdir(self.folder_path_acc):
            os.mkdir(self.folder_path_acc)


        self.titles = ["Original", "Mask", "Prediction"]
        self.save_paths = [os.path.join(self.folder_path_img, name) for name in self.titles]
        self.text_path = os.path.join(self.folder_path_img, "title.txt")
        self.val_loss_path = os.path.join(self.folder_path_loss, "val_loss.txt")
        self.train_loss_path = os.path.join(self.folder_path_loss, "train_loss.txt")
        self.val_acc_path = os.path.join(self.folder_path_acc, "val_acc.txt")
        self.train_acc_path = os.path.join(self.folder_path_acc, "train_acc.txt")
        self.processes = []
        self.communicate = None

        if load_weights:
            #Old files are loaded and written to the new run (in which the old run is continued.)
            paths_from = [self.old_text_path, self.old_val_loss_path, self.old_train_loss_path, self.old_val_acc_path, self.old_train_acc_path]
            paths_to = [self.text_path, self.val_loss_path, self.train_loss_path, self.val_acc_path, self.train_acc_path]
            for path_from, path_to in zip(paths_from, paths_to):
                file = open(path_from, 'r')
                old_file = file.read()
                file.close()
                file = open(path_to, 'w')
                file.write(old_file)
                file.close()
        else:
            f = open(self.text_path, 'w')
            f.close()

            with open(self.val_loss_path, 'w') as f:
                f.write('VAL')
                f.write('\n')
                f.write("Epoch,Loss")
                f.write('\n')

            with open(self.train_loss_path, 'w') as f:
                f.write('TRAIN')
                f.write('\n')
                f.write("Epoch,Loss")
                f.write('\n')

            with open(self.val_acc_path, 'w') as f:
                f.write('VAL')
                f.write('\n')
                f.write("Epoch,Acc")
                f.write('\n')

            with open(self.train_acc_path, 'w') as f:
                f.write('TRAIN')
                f.write('\n')
                f.write("Epoch,Acc")
                f.write('\n')


    def save_results(self, img_arrays, epoch, logs):
        """Saves the results of the training epoch to appropriate file
            these files are accessed by the subprocesses and rendered visually.

            Arguments:
            img_array -- contains the arrays of the origina image, annotation mask and prediction mask
            epoch -- the current epoch number
            logs -- contains the performance for current epoch"""
        
        saved = list(map(np.save, self.save_paths, img_arrays))

        with open(self.text_path, 'a') as f:
            f.write(str(epoch+1))
            f.write('\n')

        with open(self.val_loss_path, 'a') as f:
            f.write(str(epoch+1))
            f.write(',')
            f.write(str(logs['val_loss']))
            f.write('\n')

        with open(self.train_loss_path, 'a') as f:
            f.write(str(epoch+1))
            f.write(',')
            f.write(str(logs['loss']))
            f.write('\n')

        with open(self.val_acc_path, 'a') as f:
            f.write(str(epoch+1))
            f.write(',')
            f.write(str(logs['val_binary_io_u']))
            f.write('\n')

        with open(self.train_acc_path, 'a') as f:
            f.write(str(epoch+1))
            f.write(',')
            f.write(str(logs['binary_io_u']))
            f.write('\n')



    def on_train_begin(self, logs=None):
        """Starts three subprocesses for callbacks, 1) loss grpah 2) performance graph 3)images of prediction"""
        self.processes.append(subprocess.Popen(["py",  "DisplayApp.py", "-I", "-P", str(self.main_path)], shell=True))
        self.processes.append(subprocess.Popen(["py", "DisplayApp.py", "-L", "-P", str(self.main_path)], shell=True))
        self.processes.append(subprocess.Popen(["py", "DisplayApp.py", "-A", "-P", str(self.main_path)], shell=True))

    def on_epoch_end(self, epoch, logs=None):
        data_samples = next(iter(self.data))
        choice = random.randint(0,len(data_samples[0])-1)
        img_array = data_samples[0][choice]

        mask_array = data_samples[1][choice]
        pred_array = self.model.predict(np.expand_dims(img_array.numpy(), axis=0))

        img_array = img_array.numpy().astype(np.float32)
        mean = np.array([103.939, 116.779, 123.68])
        for x, mean_i in enumerate(mean):
            img_array[..., x] = img_array[..., x] + mean_i
        img_array = img_array[...,::-1]
        img_array = np.clip(img_array, 0, 255).astype('uint8')
        try:

            self.save_results([img_array, mask_array.numpy(), pred_array.squeeze(axis=0)], epoch, logs=logs)
        except Exception as e:
            print(e)
            # self.save_results([img_array.numpy(), mask_array.numpy(), pred_array.squeeze(axis=0)], epoch, logs=logs)

    def on_train_end(self, logs=None):
        time.sleep(3)
        with open(self.text_path, 'w') as f:
            f.write('END')
