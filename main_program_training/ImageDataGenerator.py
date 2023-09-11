import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
import math
import os
import cv2
import random
import matplotlib.pyplot as plt
import itertools


class ImageDataGeneratorSplit():
    def __init__(self, input_shape, batch_size, data_path, val_path, this_run_path=None,  old_run_path=None, split=0.1, multiprocessing=False):
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.multiprocessing = multiprocessing
        image_path = os.path.join(data_path, "JPEGImages")
        mask_path = os.path.join(data_path, "SegmentationClassPNG")
        self.images_path = [os.path.join(image_path,  name) for name in os.listdir(image_path)]
        self.masks_path = [os.path.join(mask_path,  name) for name in os.listdir(mask_path)]

        val_img_path = os.path.join(val_path, "JPEGImages")
        val_mask_path = os.path.join(val_path, "SegmentationClassPNG")
        self.val_img_path = [os.path.join(val_img_path,  name) for name in os.listdir(val_img_path)]
        self.val_mask_path = [os.path.join(val_mask_path,  name) for name in os.listdir(val_mask_path)]

        self.merg_img = self.images_path + self.val_img_path
        self.merg_mask = self.masks_path + self.val_mask_path


        merg_size = len(self.merg_img)
        val_size = math.ceil(merg_size * split)
        sample_indices = None

        if type(old_run_path)!= type(None):
            with open(os.path.join(old_run_path,"sample_indices.txt"), 'r') as f:
                indices = f.read()
                sample_indices = [int(ind) for ind in indices.split(',')[:-1]]
        else:
            sample_indices = random.sample(range(merg_size), val_size)

        with open(os.path.join(this_run_path, "sample_indices.txt"), 'w') as f:
            for ind in sample_indices:
                f.write(str(ind))
                f.write(',')

        self.images_path, self.masks_path, self.val_img_path, self.val_mask_path = [],[],[],[]
        for i in range(merg_size):
            if i in sample_indices:
                self.val_img_path.append(self.merg_img[i])
                self.val_mask_path.append(self.merg_mask[i])
            else:
                self.images_path.append(self.merg_img[i])
                self.masks_path.append(self.merg_mask[i])

    def get_generators(self):
        train_gen = ImageDataGenerator(self.input_shape, self.batch_size, [self.images_path, self.masks_path], split=True, multiprocessing=self.multiprocessing, train=True)
        val_gen = ImageDataGenerator(self.input_shape, self.batch_size, [self.val_img_path, self.val_mask_path], split=True,  multiprocessing=self.multiprocessing)
        return train_gen, val_gen

class ImageDataGenerator(tf.keras.utils.Sequence):
    #https://medium.com/analytics-vidhya/write-your-own-custom-data-generator-for-tensorflow-keras-1252b64e41c3
    def __init__(self, input_shape, batch_size, path, split=False, multiprocessing=False, train=False):
        self.input_shape = input_shape
        self.batch_size = batch_size # when using tf dataset
        self.data_path = path
        self.multiprocessing = multiprocessing
        self.train=train

        if split:
            self.images_path = path[0]
            self.masks_path = path[1]
        else:
            image_path = os.path.join(self.data_path, "JPEGImages")
            mask_path = os.path.join(self.data_path, "SegmentationClassPNG")
            self.images_path = [os.path.join(image_path,  name) for name in os.listdir(image_path)]
            self.masks_path = [os.path.join(mask_path,  name) for name in os.listdir(mask_path)]

        self.data_size = len(self.images_path)
        self.len = self.__len__()


        self.images_loaded = {}
        self.masks_loaded = {}
        # self.images = [np.array(Image.open(os.path.join(image_path,  name))) \
        #  for name in os.listdir(image_path)]
        # self.masks = [np.array(Image.open(os.path.join(mask_path,  name))) \
        #  for name in os.listdir(mask_path)]

        self.shuffled_indices = [x for x in range(self.data_size)]
        random.shuffle(self.shuffled_indices)


    def __len__(self):
        batch_size = 1 if self.multiprocessing else self.batch_size

        return math.ceil(self.data_size/batch_size)

    def getitem(self, index):
        return self.__getitem__(index)

    def __getitem__(self, index):
        original_index = index
        batch_size = 1 if self.multiprocessing else self.batch_size
        index = self.shuffled_indices[index]

        try:
            images_batch = [self.images_loaded[x] for x in range[index*batch_size:(index+1)*batch_size]]
            masks_batch = [self.masks_loaded[x] for x in range[index*batch_size:(index+1)*batch_size]]
        except:
            images_batch = [np.array(Image.open(x)) for x in self.images_path[index*batch_size:(index+1)*batch_size]]
            images_batch = list(map(ImageDataGenerator.clean_info_box,images_batch))
            masks_batch = [np.array(Image.open(x)) for x in self.masks_path[index*batch_size:(index+1)*batch_size]]
            masks_batch = list(map(ImageDataGenerator.neg_samples, masks_batch))
            masks_batch =  np.array(list(map(ImageDataGenerator.dim_exp, masks_batch)))
            if not self.multiprocessing:
                images_batch, masks_batch = ImageDataGenerator.preprocsseing_aug(images_batch, masks_batch, self.input_shape[0:2], train=self.train)

                if len(images_batch.shape)!=4:
                    images_batch = np.expand_dims(images_batch, axis=0)
                if len(masks_batch.shape)!=4:
                    masks_batch = np.expand_dims(masks_batch, axis=0)

            for img, mask, ind in zip(images_batch, masks_batch, range(index*batch_size,(index+1)*batch_size)):
                self.images_loaded[ind] = img
                self.masks_loaded[ind] = mask


        if original_index == (self.data_size-1):
            random.shuffle(self.shuffled_indices)


        return images_batch, masks_batch


    @staticmethod
    def clean_info_box(arr):
        new_arr = arr.copy()
        min_row, max_row, min_col, max_col = 1855,1903, 2345,2542
        for i in range(3):
            new_arr[min_row:max_row, min_col:max_col,i] = new_arr[...,i].mean()
        return arr


    @staticmethod
    def dim_exp(mask):
        return np.expand_dims(mask, 2)

    @staticmethod
    def zero_center_image(image):
        mean = np.array([103.939, 116.779, 123.68])
        image = image[..., ::-1]
        image = image.astype(np.float32)
        for i, mean_i in enumerate(mean):
            image[..., i] = image[..., i] - mean_i
        return image


    @staticmethod
    def data_augment_brightness_sharpness_color_constrast(img, mask):
        img = Image.fromarray(np.uint8(img))
        factor = random.uniform(0.75,1.25)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)
        factor = random.uniform(0.75,1.25)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)
        return np.array(img), mask

    @staticmethod
    def data_augment_rot(img, mask):
        num = random.uniform(0,1)
        img = tf.keras.preprocessing.image.apply_affine_transform(x=img, row_axis=0, col_axis=1, channel_axis=2, theta=360*num,  fill_mode='reflect')
        mask = tf.keras.preprocessing.image.apply_affine_transform(x=mask, row_axis=0, col_axis=1, channel_axis=2, theta=360*num,  fill_mode='reflect')

        return img, mask

    @staticmethod
    def data_augment_flip(img, mask):
        img = tf.image.flip_left_right(img)
        mask = tf.image.flip_left_right(mask)
        return np.array(img), np.array(mask)

    @staticmethod
    def data_augment_gaussian_noise(img, mask):
        img = tf.keras.layers.GaussianNoise(1e-1)(img)
        return img.numpy(), mask



    @staticmethod
    def data_augment_crop(img, mask_in, input_shape, val):

        height = input_shape[0]
        width = input_shape[1]
        max_heigth = img.shape[0]
        max_width = img.shape[1]
        row_min, row_max, col_min, col_max = None, None, None, None
        mask_in = mask_in.squeeze()
        rows_with_labels, cols_with_labels = mask_in.nonzero()

        start_row, start_col = None, None
        start_row = random.randint(0, height)
        start_col = random.randint(0, width)

        if len(rows_with_labels) != 0 and len(cols_with_labels) != 0:
            # return ImageDataGenerator.resize_images(img, input_shape), ImageDataGenerator.resize_masks(ImageDataGenerator.dim_exp(mask_in), input_shape)
            if random.randint(0,1) or val:
                indice = random.randint(0, rows_with_labels.shape[0])
                try:
                    start_row = rows_with_labels[indice]
                    start_col = cols_with_labels[indice]
                except:
                    pass


        try:
            row_min = random.randint(max(0, start_row), start_row)
            row_min = min(row_min, max_heigth-height)
            # print("Row min: {}".format(row_min))
            row_max = row_min + height
            # print("Row max: {}".format(row_max))
            col_min = random.randint(max(0, start_col), start_col)
            col_min = min(col_min, max_width-width)
            # print("col min: {}".format(col_min))
            col_max = col_min + width
            # print("col max: {}".format(col_max))
            image = img[row_min:row_max, col_min:col_max].copy()
            mask = mask_in[row_min:row_max, col_min:col_max].copy()


        except:

            return ImageDataGenerator.data_augment_crop(img, mask_in, input_shape)

        if image.shape[1]!=width or image.shape[0]!=height:
            print(image.shape)
            return ImageDataGenerator.data_augment_crop(img, mask_in, input_shape)

        return image.astype(np.uint8), ImageDataGenerator.dim_exp(mask).astype(np.uint8)

    @staticmethod
    def neg_samples(mask):
        rows_with_labels, cols_with_labels = mask.squeeze().nonzero()
        if len(rows_with_labels) < 100 and len(cols_with_labels) < 100:
            mask = np.zeros_like(mask)
        return mask

    @staticmethod
    def preprocsseing_aug(images_batch, masks_batch, input_shape, train=False):

        try:
            images_batch = images_batch.numpy().copy()
            masks_batch = masks_batch.numpy().copy()
            input_shape = tuple(input_shape.numpy().copy())
        except:
            pass



        nums = [random.randint(0,20) for _ in range(5)]

        if train:
            if  images_batch[0].shape[0:2] > input_shape:
                images_batch, masks_batch =  zip(*list(map(ImageDataGenerator.data_augment_crop, images_batch, masks_batch, itertools.repeat(input_shape), itertools.repeat(train))))

            images_batch, masks_batch = zip(*list(map(ImageDataGenerator.data_augment_brightness_sharpness_color_constrast, images_batch, masks_batch)))
            #
            if nums[3] > 10:
                images_batch, masks_batch = zip(*list(map(ImageDataGenerator.data_augment_flip, images_batch, masks_batch)))

            #
            # images_batch, masks_batch = zip(*list(map(ImageDataGenerator.data_augment_gaussian_noise, images_batch, masks_batch)))
            images_batch, masks_batch = zip(*list(map(ImageDataGenerator.data_augment_rot, images_batch, masks_batch)))
            #

        if not train:
            if images_batch[0].shape[0:2] > input_shape:
                images_batch, masks_batch =  zip(*list(map(ImageDataGenerator.data_augment_crop, images_batch, masks_batch, itertools.repeat(input_shape), itertools.repeat(train))))

        images_batch = list(map(ImageDataGenerator.zero_center_image, images_batch))

        im_ret = np.array(images_batch[0]) if len(images_batch)==1 else np.array(images_batch)
        mask_ret = np.array(masks_batch[0]) if len(masks_batch)==1 else np.array(masks_batch)

        return im_ret, mask_ret
