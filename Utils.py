import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_data():
    # Set the seed for random operations. 
    # This let our experiments to be reproducible. 
    SEED = 1234
    tf.random.set_seed(SEED)  

    #Target directory
    training_dir = "Classification_Dataset\\training"

    # Batch size
    bs = 8

    # img shape
    img_h = 512
    img_w = 512

    #number of classes
    num_classes=20

    train_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            rescale=1./255)

    train_gen = train_data_gen.flow_from_directory(training_dir,batch_size=bs, 
                                                   class_mode='categorical',shuffle=True,seed=SEED)  # targets are directly converted into one-hot vectors

    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

    train_dataset = train_dataset.repeat()

    iterator = iter(train_dataset)

    """for _ in range(1000):
        augmented_img, target = next(iterator)
        
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize
   
        plt.imshow(np.uint8(augmented_img))
        plt.show()"""