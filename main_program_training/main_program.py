import os
import sys
import tensorflow as tf
# tf.config.experimental_run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()
from Unet import *
from ImageDataGenerator import *
from VGG19 import *
from CallbackVisual import *
import Utils_yeast
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
import datetime
import shutil
import visualkeras


main_path = os.getcwd()

#The main folder must contain a dataset with semantically annotated files in VOC structure
dataset_path = os.path.join(main_path,'dataset_voc')
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
eval_path = os.path.join(dataset_path, "test")

#abs paths to all training data
train_jpg, train_masks = os.path.join(train_path, 'JPEGImages'), os.path.join(train_path, 'SegmentationClassPNG')
val_jpg, val_masks = os.path.join(val_path, 'JPEGImages'), os.path.join(val_path, 'SegmentationClassPNG')
train_img_paths = [os.path.join(train_jpg, filename) for filename in os.listdir(train_jpg)] 
train_mask_paths = [os.path.join(train_masks, filename) for filename in os.listdir(train_masks)]
val_img_paths = [os.path.join(val_jpg, filename) for filename in os.listdir(val_jpg)]
val_mask_paths = [os.path.join(val_masks, filename) for filename in os.listdir(val_masks)]

#batch_size and input_shape should be adjusted to match your GPU resources
#batch_size = 8

#crop size
crop_size = (224,224)

#The input size to the tensorflow model
input_shape = (None,224,224,3) #(None,224,224,3) #(None,1024,1024,3)
input_shape_mask = (None,224,224,1) #(None,224,224,1)


#The size of the images in the dataset
original_shape= (None,1920, 2560, 3)
original_mask = (None,1920, 2560,1)

#Move old runs to keep the main folder clean
for filepath in os.listdir(main_path):
    if os.path.isdir(filepath):
        if filepath.endswith('_RUN'):
            shutil.move(os.path.join(main_path, filepath), os.path.join(os.path.join(main_path, 'PreviousRUNS'), filepath))


#A new folder is created to contain the history and checkpoint of this run
save_folder_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

this_run = os.path.join(main_path, save_folder_date+'_RUN')
os.mkdir(this_run)

save_folder_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#This option loads the checkpoints weights
load_weights = False

#The path to the weights of the checkpoint is given here.
path = None
#The callbacks needs the path to where the training data history is saved.
old_checkpoint_path = path if path else None
path_to_old_run = os.path.dirname(old_checkpoint_path) if type(old_checkpoint_path)!=type(None) else None

#his option is choosen if the trained model should be fine_tuned
fine_tune_model = False

#This option is choosen if the fined tuned should be evalued.
eval_model = False

#The dataset is more efficiently loaded to memory by mapping where parallization can be utilized
@tf.function
def load(path_img, path_mask):
    img = tf.image.decode_image(tf.io.read_file(path_img), dtype=tf.float32)
    mask = tf.image.decode_image(tf.io.read_file(path_mask), dtype=tf.float32)
    mask = tf.where(mask >= 0.5, 1.0, 0.0)
    img = tf.expand_dims(img, axis=0) * 255
    mask = tf.expand_dims(mask, axis=0)
    return img, mask


#Used for cropping the images
@tf.function
def crop_images(img, mask):
    
    seed = tf.random.uniform((), maxval=2000, dtype=tf.int32)
    seed = tf.get_static_value(seed)
    if type(seed)==type(None):
        seed = 2000

    img_shape = tf.shape(img)
    batch_size, row_size, col_size, channel_size =\
        img_shape[0], img_shape[1], img_shape[2], img_shape[3]
    
    start_x = tf.random.uniform((), maxval=row_size - crop_size[0], dtype=tf.int32, seed=seed)
    start_y = tf.random.uniform((), maxval=col_size - crop_size[1], dtype=tf.int32, seed=seed)

    boxes = tf.convert_to_tensor([
        [start_y / col_size, start_x / row_size, (start_y + crop_size[1]) / col_size, (start_x + crop_size[0]) / row_size]
    ])

    boxes = tf.cast(boxes, tf.float32)
    cropped_mask = tf.image.crop_and_resize(mask, boxes, [0], crop_size, method="nearest") 
    cropped_img = tf.image.crop_and_resize(img, boxes, [0], crop_size, method="nearest")
    # print(tf.unique(tf.reshape(mask, [-1])))
    # print(tf.unique(tf.reshape(cropped_mask, [-1])))
    return cropped_img, cropped_mask

@tf.function
def random_image_manipulations(img, mask):
    factor = 0.5
    img = tf.image.random_brightness(img, max_delta = factor)

    return img, mask

@tf.function
def flip_left_right_images(img, mask):
    
    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)

    flipped_mask = tf.where(tf.less(rn, 0.5),
                              tf.image.flip_left_right(mask), mask)
    flipped_img = tf.where(tf.less(rn, 0.5), 
                              tf.image.flip_left_right(img), img)

    return flipped_img, flipped_mask

@tf.function
def flip_up_down_images(img, mask):
    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    flipped_mask = tf.where(tf.less(rn, 0.5),
                              tf.image.flip_up_down(mask), mask)
    flipped_img = tf.where(tf.less(rn, 0.5), 
                              tf.image.flip_up_down(img), img)
    
    return flipped_img, flipped_mask

@tf.function
def rotate_images(img, mask):

    rn = tf.random.uniform(shape=[], minval=0, maxval=1, dtype=tf.float32)
    k = tf.random.uniform(shape=[], minval=0, maxval=3, dtype=tf.int32)

    rotated_mask = tf.where(tf.less(rn, 0.5),
                              tf.image.rot90(mask, k), mask)
    rotated_img = tf.where(tf.less(rn, 0.5), 
                              tf.image.rot90(img, k), img)
    
    return rotated_img, rotated_mask

@tf.function
def zero_center_image(image, mask):
    mean = tf.constant([123.68, 116.779, 103.939])
    image = image - mean[tf.newaxis, tf.newaxis, tf.newaxis, :]
    
    return image, mask

@tf.function
def convert_rbg_brg(image, mask):
    image = tf.reverse(image, axis=[-1])
    return image, mask

@tf.function
def remove_info_box(image, mask):
    min_row, max_row, min_col, max_col = 1855,1903, 2345,2542
    img_shape  = tf.shape(image)
    batch_size, row_size, col_size, channel_size =\
          img_shape[0], img_shape[1], img_shape[2], img_shape[3]
    
    channel_means = tf.reduce_mean(image, axis=(1,2), keepdims=True)
    expected_shape = (None, 1, 1, 3)  # Assuming 3 channels for RGB
    channel_means = tf.ensure_shape(channel_means, expected_shape)

    channel_means = tf.tile(channel_means, [1, row_size, col_size, 1])

    # Create a range of indices for rows and columns
    row_indices = tf.range(row_size)
    col_indices = tf.range(col_size)

    # Create boolean masks for rows and columns
    row_mask = (row_indices >= min_row) & (row_indices < max_row)
    col_mask = (col_indices >= min_col) & (col_indices < max_col)
    
    row_mask = row_mask[:, tf.newaxis]
    col_mask = col_mask[tf.newaxis, :]

    bool_mask = tf.logical_and(row_mask, col_mask)
    
    bool_mask = tf.expand_dims(bool_mask, axis=0)  # Add batch dimension
    bool_mask = tf.expand_dims(bool_mask, axis=3)  # Add channel dimension  # Add batch and channel dimensions
    bool_mask = tf.tile(bool_mask, [batch_size, 1, 1, channel_size])  # Tile along batch and channel dimensions
    

    image = tf.where(bool_mask, channel_means, image)
    
    return image, mask

@tf.function
def remove_batch_dimension(img, mask):
    # Assuming image_batch has shape (batch_size, height, width, channels)
    # Remove the batch dimension
    img = tf.squeeze(img, axis=0)
    mask = tf.squeeze(mask, axis=0)
    return img, mask

@tf.function
def convert_mask_to_single_channel(img, mask):
    mask = tf.reduce_max(mask, axis=-1, keepdims=True)
    
    return img, mask

#a function that will merge the paths and return a split. It will also create a file that
#remembers the paths for the next training phase
def split_train_val(images_path, masks_path, val_img_path, val_mask_path, old_run_path, this_run_path, split=0.2):

    merg_img =  images_path + val_img_path
    merg_mask = masks_path  + val_mask_path

    merg_size = len(merg_img)
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

    images_path, masks_path, val_img_path, val_mask_path = [],[],[],[]
    for i in range(merg_size):
        if i in sample_indices:
            val_img_path.append(merg_img[i])
            val_mask_path.append(merg_mask[i])
        else:
            images_path.append(merg_img[i])
            masks_path.append(merg_mask[i])
    
    return images_path, masks_path, val_img_path, val_mask_path


def main():
    global batch_size
    global train_img_paths
    global train_mask_paths
    global val_img_paths
    global  val_mask_paths

    if eval_model:
        """Evalauation is perfomed with the original sized image. 
            The fine tuned models weights are loaded to new model which is complied
            with a new input shape"""
        batch_size = 1
        eval_data_generator = ImageDataGenerator(original_shape[1:3], batch_size, val_path)
        model = tf.keras.models.load_model('fine_tuned_model', compile=False)
        model_weights = model.get_weights()
        pre_train_model = VGG19()
        model = Unet(pre_train_model)
        model.build(original_shape)
        model.set_weights(model_weights)
        model.compile(optimizer=tf.keras.optimizers.Adamax(),
                    loss=Utils_yeast.TverskyLoss(beta=0.05),
                    metrics= [tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=1)])

        pred = model.evaluate(x=eval_data_generator)

        return

    #merges and splits the validation and training dataset
    train_img_paths, train_mask_paths, val_img_paths, val_mask_paths = \
      split_train_val(train_img_paths, train_mask_paths, val_img_paths, val_mask_paths,\
                       old_run_path=path_to_old_run, this_run_path=this_run)
    
    #Create a tuple of the training data
    train_dataset = tf.data.Dataset.from_tensor_slices((train_img_paths, train_mask_paths))
    val_dataset = tf.data.Dataset.from_tensor_slices((val_img_paths, val_mask_paths))

    #load the dataset
     #load the dataset from path
    train_dataset = train_dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    
    #remove remove info box
    train_dataset = train_dataset.map(remove_info_box, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(remove_info_box, num_parallel_calls=tf.data.AUTOTUNE)

    #preprocess the image
    train_dataset = train_dataset.map(zero_center_image, num_parallel_calls=tf.data.AUTOTUNE)\
            .map(convert_rbg_brg, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(zero_center_image, num_parallel_calls=tf.data.AUTOTUNE)\
            .map(convert_rbg_brg, num_parallel_calls=tf.data.AUTOTUNE)

    #Augmentations
    train_dataset = train_dataset.map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(random_image_manipulations, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_left_right_images, num_parallel_calls=tf.data.AUTOTUNE )\
        .map(rotate_images, num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(crop_images, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(random_image_manipulations, num_parallel_calls=tf.data.AUTOTUNE)\
        .map(flip_left_right_images, num_parallel_calls=tf.data.AUTOTUNE )\
        .map(rotate_images, num_parallel_calls=tf.data.AUTOTUNE)
    
    #remove batch dimension since it is added by tf.data.dataset batch
    train_dataset = train_dataset.map(remove_batch_dimension, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(remove_batch_dimension, num_parallel_calls=tf.data.AUTOTUNE)

    # #finally reduce to 1 channel for mask.
    train_dataset = train_dataset.map(convert_mask_to_single_channel, num_parallel_calls=tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(convert_mask_to_single_channel, num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(16).prefetch(buffer_size=tf.data.AUTOTUNE)

    
    # for i in train_dataset:
    #     img = i[0][0].numpy().astype(np.float32)
    #     plt.imshow(img.squeeze())
    #     plt.show()
    #     mean = np.array([103.939, 116.779, 123.68])
    #     for x, mean_i in enumerate(mean):
    #         img[..., x] = img[..., x] + mean_i
    #     img = img[...,::-1]
    #     img = np.clip(img, 0, 255).astype('uint8')
    #     plt.imshow(img.squeeze())
    #     plt.show()
    #     plt.imshow(i[1][0].numpy().squeeze())
    #     plt.show()


    model = None
    
    #hyperparamters
    learning_rate = None
    min_lr = None
    init_epoch = 0
    factor = None
    patience= None

    if not fine_tune_model:
        #The hyperparameters for first time runs or checkpoints
        learning_rate = 1e-3
        min_lr = 1e-7
        patience=2
        factor = 0.8
        pre_train_model = VGG19()
        pre_train_model.trainable=False
        epochs= 100
        model = Unet(pre_train_model)
        model.build(input_shape)
        model.layers[0].load()
        model.summary(expand_nested=True)

        if load_weights:
            #assert os.path.exists(old_checkpoint_path)
            model.load_weights(os.path.join(old_checkpoint_path, tf.train.latest_checkpoint(old_checkpoint_path))).expect_partial()
            with open(os.path.join(os.path.join(os.path.dirname(old_checkpoint_path), 'ImgArrays'), 'title.txt')) as file:
                init_epoch = int(file.readlines()[-1])
            learning_rate = 1e-6
            min_lr = 1e-8
            epochs= 200

    else:
        #The hyperparamters for fine tuning
        learning_rate = 1e-7
        min_lr = 1e-10
        patience=2
        factor = 0.8
        epochs=100
        model = tf.keras.models.load_model('transfer_model', compile=False)
        model.trainable = True
        model.summary(expand_nested=True)

    #callback for reducing learning rate when loss stops decreasing
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=factor,
                              patience=patience, verbose=1, min_lr=min_lr)

    #The model is compiled with the following optimizer, cost function, and performance metric. 
    model.compile(optimizer=tf.keras.optimizers.Adamax(),
                loss=Utils_yeast.TverskyLoss(beta=1),
                metrics= [tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)\
                          ,tf.keras.metrics.IoU(num_classes=2, target_class_ids=[0])\
                        , tf.keras.metrics.IoU(num_classes=2, target_class_ids=[1])])

    #CallbackVisual is a customized callback function for performance graphs and predictions.
    callback_visual = CallbackVisual(val_dataset, this_run, load_weights, path_to_old_run)

    #The path for where saving the model checkpoints during training
    checkpoint_path = os.path.join(this_run, save_folder_date + "_training/cp-{epoch:04d}.ckpt")

    checkpoint = ModelCheckpoint(
                                checkpoint_path,
                                save_weights_only= False if fine_tune_model else True,
                                monitor='val_loss',
                                #save_best_only=True,
                                mode='auto',
                                period= 1 if fine_tune_model else 10 )


    
    model.fit(train_dataset,
        validation_data=val_dataset,
        initial_epoch=init_epoch,
        epochs=epochs,
        callbacks=[callback_visual, reduce_lr, checkpoint])

    if not fine_tune_model:
        model.save('transfer_model')
    else:
        model.save('fine_tuned_model')
    print("Weights saved")
    print("Training Complete")



if __name__ == "__main__":
    main()
    sys.exit()
