import os
import sys
import tensorflow as tf
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

#batch_size and input_shape should be adjusted to match your GPU resources
batch_size = 8

#The input size to the tensorflow model
input_shape = (None,512,512,3) #(None,224,224,3) #(None,1024,1024,3)
input_shape_mask = (None,512,512,1) #(None,224,224,1)

#The expected shape is given to the tf.ensure_shape to ensure the size during runtime in the pipeline
expected_imageshape = (512,512,3) #(224,224,3)
expected_maskshape = (512,512,1)#(224,224,1)

#The size of the images in the dataset
original_shape= (None,1920, 2560, 3)
original_mask = (None,1920, 2560,1)



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

def main():
    global batch_size


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

        sys.exit()


    #Move old runs to keep the main folder clean
    for filepath in os.listdir(main_path):
        if os.path.isdir(filepath):
            if filepath.endswith('_RUN'):
                shutil.move(os.path.join(main_path, filepath), os.path.join(os.path.join(main_path, 'PreviousRUNS'), filepath))

    #A new folder is created to contain the history and checkpoint of this run
    this_run = os.path.join(main_path,save_folder_date+'_RUN')
    os.mkdir(this_run)


    # train_data_generator = ImageDataGenerator(input_shape[1:3], batch_size, train_path)
    # val_data_generator = ImageDataGenerator(input_shape[1:3], batch_size, val_path)

    #The val and train dataset is split and generators are created to accomedate each set
    gen_split = None
    if load_weights or fine_tune_model:
        gen_split = ImageDataGeneratorSplit(input_shape[1:3], batch_size, train_path, val_path, this_run_path=this_run, old_run_path=path_to_old_run, split=0.2, multiprocessing=True)
    else:
        gen_split = ImageDataGeneratorSplit(input_shape[1:3], batch_size, train_path, val_path, this_run_path=this_run, split=0.2, multiprocessing=True)

    train_data_generator, val_data_generator = gen_split.get_generators()


    #To migrate a tf sequence datagenerator to tf dataset a hack was utilized below because no formal procedure exists yet.

    def train_data_generator_public():
        for i in range(train_data_generator.data_size):
            yield train_data_generator.getitem(i)

    def val_data_generator_public():
        for i in range(val_data_generator.data_size):
            yield val_data_generator.getitem(i)


    train_dataset_interm = tf.data.Dataset.from_generator(generator=train_data_generator_public, output_signature=((
        tf.TensorSpec(shape=(original_shape), dtype=np.float32),
        tf.TensorSpec(shape=(original_mask), dtype=np.float32))))

    val_dataset_interm = tf.data.Dataset.from_generator(generator=val_data_generator_public, output_signature=((
        tf.TensorSpec(shape=original_shape, dtype=np.float32),
        tf.TensorSpec(shape=original_mask, dtype=np.float32))))


    train_dataset = train_dataset_interm.map(lambda x,y: tf.py_function(ImageDataGenerator.preprocsseing_aug, inp=[x,y, tf.constant([input_shape[1], input_shape[2]]), tf.constant(1)], Tout=[tf.float32, tf.float32])
        , num_parallel_calls=tf.data.AUTOTUNE)

    train_dataset = train_dataset.map(lambda x,y: (tf.ensure_shape(x,expected_imageshape), tf.ensure_shape(y,expected_maskshape)), num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = val_dataset_interm.map(lambda x,y: tf.py_function(ImageDataGenerator.preprocsseing_aug, inp=[x,y, tf.constant([input_shape[1], input_shape[2]])], Tout=[tf.float32, tf.float32])
        , num_parallel_calls=tf.data.AUTOTUNE)

    val_dataset = val_dataset.map(lambda x,y: (tf.ensure_shape(x,expected_imageshape), tf.ensure_shape(y,expected_maskshape)), num_parallel_calls=tf.data.AUTOTUNE)


    train_dataset = train_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    # for i in iter(train_dataset):
    #     print(i[0].shape)
    #     print(i[1].shape)
    #     img = i[0][0].numpy().astype(np.float32)
    #     plt.imshow(img)
    #     plt.show()
    #     mean = np.array([103.939, 116.779, 123.68])
    #     for x, mean_i in enumerate(mean):
    #         img[..., x] = img[..., x] + mean_i
    #     img = img[...,::-1]
    #     img = np.clip(img, 0, 255).astype('uint8')
    #     plt.imshow(img)
    #     plt.show()
    #     plt.imshow(i[1][0])
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
        min_lr = 1e-5
        patience=1
        factor = 0.5
        pre_train_model = VGG19()
        pre_train_model.load()
        pre_train_model.trainable=False
        epochs= 100
        model = Unet(pre_train_model)
        model.build(input_shape)
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
                loss=Utils_yeast.TverskyLoss(beta=0.5),
                metrics= [tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5)])

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
        batch_size=batch_size,
        callbacks=[callback_visual, reduce_lr, checkpoint])

    if not fine_tune_model:
        model.save('transfer_model')
    else:
        model.save('fine_tuned_model')
    print("Weights saved")
    print("Training Complete")



if __name__ == "__main__":
    main()
