import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
import sys
import os
import numpy as np
import time
import argparse
from PIL import Image, ImageOps

class DisplayApp():

    def __init__(self, main_path):

        self.main_path = main_path
        self.folder_path_img = os.path.join(self.main_path, "ImgArrays")
        self.folder_path_loss = os.path.join(self.main_path, "LossArrays")
        self.folder_path_saved = os.path.join(self.main_path, "SAVED")
        self.folder_path_acc = os.path.join(self.main_path, "AccArrays")
        self.titles = ["Original", "Mask", "Prediction"]
        self.load_paths = [os.path.join(self.folder_path_img, name+".npy") for name in self.titles]
        self.text_path = os.path.join(self.folder_path_img, "title.txt")
        self.val_loss_path = os.path.join(self.folder_path_loss, "val_loss.txt")
        self.train_loss_path = os.path.join(self.folder_path_loss, "train_loss.txt")
        self.val_acc_path = os.path.join(self.folder_path_acc, "val_acc.txt")
        self.train_acc_path = os.path.join(self.folder_path_acc, "train_acc.txt")

        plt.figure(figsize=(20,20))
        self.epoch = -1

    def get_epoch(self):
        if os.path.getsize(self.text_path) == 0:
            return None
        try:
            return int(open(self.text_path, 'r').readlines()[-1])
        except:
            return open(self.text_path, 'r').readlines()[-1]

    def remove_last_line(self):
        with open(self.text_path, 'r+') as file:
            lines = file.readlines()
            file.seek(0)
            file.truncate()
            file.writelines(lines[:-1])

    def animate_accuracy(self):
        try:
            data = {}
            paths =  [self.val_acc_path, self.train_acc_path]
            plt.clf()
            for i, path in enumerate(paths):
                data[path] = {}
                graph_data = open(path, 'r')
                x_key = None
                y_key = None
                label = None
                for index, line in enumerate(graph_data):
                    if index == 0:
                        label = line
                        continue
                    if index == 1 :
                        x_key,y_key = line.split(',')
                        data[path][x_key] = []
                        data[path][y_key] = []
                        continue
                    x,y = line.split(',')
                    data[path][x_key].append(float(x))
                    data[path][y_key].append(float(y))

                plt.plot(x_key, y_key, data=data[path], label=label)

            plt.legend(loc='best')
            plt.ylabel('BinaryIoU')
            plt.xlabel('Epoch')
            plt.title('Accuracy')
            plt.grid()
            plt.show(block=False)
            self.epoch = self.get_epoch()
            if self.epoch % 5 == 0:
                plt.savefig(os.path.join(self.folder_path_saved,"GRAPH_ACC" + str(self.epoch) +".png"), bbox_inches='tight')
        except:
            time_sleep = 60
            print("FAILURE WITH GRAPH ACC, retry in {} sec".format(time_sleep))
            time.sleep(time_sleep)


    def animate_loss(self):
        try:
            data = {}
            paths =  [self.val_loss_path, self.train_loss_path]
            plt.clf()
            for i, path in enumerate(paths):
                data[path] = {}
                graph_data = open(path, 'r')
                x_key = None
                y_key = None
                label = None
                for index, line in enumerate(graph_data):
                    if index == 0:
                        label = line
                        continue
                    if index == 1 :
                        x_key,y_key = line.split(',')
                        data[path][x_key] = []
                        data[path][y_key] = []
                        continue
                    x,y = line.split(',')
                    data[path][x_key].append(float(x))
                    data[path][y_key].append(float(y))

                plt.plot(x_key, y_key, data=data[path], label=label)

            plt.legend(loc='best')
            plt.ylabel('Tversky Loss')
            plt.xlabel('Epoch')
            plt.title('Loss')
            plt.grid()
            plt.show(block=False)
            self.epoch = self.get_epoch()
            if self.epoch % 5 == 0:
                plt.savefig(os.path.join(self.folder_path_saved,"GRAPH_Loss" + str(self.epoch) +".png"), bbox_inches='tight')
        except:
            time_sleep = 60
            print("FAILURE WITH GRAPH LOSS, retry in {} sec".format(time_sleep))
            time.sleep(time_sleep)

    def animate_pictures(self):
        try:
            img_arrays = list(map(np.load, self.load_paths))
            size = len(img_arrays)
            #images = list(map(Image.fromarray, [img.squeeze() for img in img_arrays]))
            plt.clf()
            self.epoch = self.get_epoch()
            plt.suptitle("Epoch {}".format(self.epoch))

            for i, (title, img) in enumerate(zip(self.titles, img_arrays)):
                plt.subplot(1, size, i+1 )
                plt.title(title)
                if i < 1:
                    plt.imshow(array_to_img(img))
                else:
                    img = Image.fromarray(img.squeeze().astype(np.uint8))
                    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
                plt.axis('off')

            plt.show(block=False)

            if  self.epoch % 5 == 0:
                plt.savefig(os.path.join(self.folder_path_saved,"PIC_" + str(self.epoch) +".png"), bbox_inches='tight')
        except Exception as e:
            print(e)
            time_sleep = 60
            print("FAILURE WITH IMAGE, retry in {} sek".format(time_sleep))
            time.sleep(time_sleep)


def run():

    parser = argparse.ArgumentParser(prog='DisplayApp',
                                    description='Display app for visualizing training progress with images and graphs',
                                     )
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-I', '--I', action='store_true', help='''Use for starting a subprocess with Images
                                            ''')
    group.add_argument('-L', '--L', action='store_true', help='''Use for starting a subprocess with loss graphs
                                    ''')
    group.add_argument('-A', '--A', action='store_true', help='''Use for starting a subprocess with accuracy graphs
                                    ''')

    parser.add_argument('-P', '--P', help='Provide main path', required=True)

    args = parser.parse_args()
    display_app = DisplayApp(args.P)
    time.sleep(20)
    while(True):
        if display_app.get_epoch() == 'END':
            display_app.remove_last_line()
            break

        if display_app.get_epoch()!=display_app.epoch:
            if args.I:
                display_app.animate_pictures()

            if args.L:
                display_app.animate_loss()

            if args.A:
                display_app.animate_accuracy()
        plt.pause(1)
        time.sleep(1)

    plt.show()

run()
