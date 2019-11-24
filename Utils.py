import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import resize_images
from datetime import datetime


class_list = ['owl', 'galaxy', 'lightning', 'wine-bottle', 't-shirt', 'waterfall', 'sword', 'school-bus', 'calculator',
                 'sheet-music', 'airplanes', 'lightbulb' , 'skyscraper', 'mountain-bike', 'fireworks', 'computer-monitor',
                 'bear', 'grand-piano', 'kangaroo', 'laptop']

def load_dataset(img_h,img_w):
    # Set the seed for random operations. 
    # This let our experiments to be reproducible. 
    SEED = 1234
    tf.random.set_seed(SEED)  

    #Target directory
    training_dir = "Classification_Dataset\\training"
    validation_dir = "Classification_Dataset\\validation"


    # Batch size
    bs = 16

    # img shape
    #img_h = 512
    #img_w = 512

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
                                            rescale=1./255,
                                            )

    valid_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            rescale=1./255,
                                            )

    train_gen = train_data_gen.flow_from_directory(training_dir,batch_size=bs, target_size=(img_h, img_w),
                                                   class_mode='categorical',shuffle=True,seed=SEED,
                                                   classes=class_list)  # targets are directly converted into one-hot vectors

    valid_gen = valid_data_gen.flow_from_directory(validation_dir,batch_size=bs, target_size=(img_h, img_w),
                                                   class_mode='categorical',shuffle=False,seed=SEED,
                                                   classes=class_list)  # targets are directly converted into one-hot vectors


    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, num_classes]))

    train_dataset = train_dataset.repeat()
    valid_dataset = valid_dataset.repeat()

    """
    iterator = iter(train_dataset)
    
    for _ in range(3):
        augmented_img, target = next(iterator)    
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize

        plt.imshow(np.uint8(augmented_img))
        print(target)
        plt.show() 

    iterator = iter(valid_dataset)
    
    for _ in range(3):
        augmented_img, target = next(iterator)    
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize

        plt.imshow(np.uint8(augmented_img))
        print(target)
        plt.show() 
    """

    return train_dataset,valid_dataset




def test_model(model,to_show,img_h,img_w):
    path = 'Classification_Dataset\\test'
    model.load_weights('classification.h5')
    #image_filenames = next(os.walk('../Classification_Dataset/test))[2]

    results = {}
    for f in os.listdir(path):
        #Image loading
        ext = os.path.splitext(f)[1]
        img = cv2.imread(os.path.join(path,f))
        img = cv2.resize(img, (img_h, img_w))
        
        #Image preparation
        img_array = img_to_array (img)
        img_array = img_array / 255 
        img_array = np.expand_dims(img_array, 0) 

        #Image prediction
        res = model.predict(img_array)
        prediction = np.argmax(res) 

        results[str(f)] = int(prediction)
        #results = results + {str(f): int(prediction)}

        if (to_show == True):
            plt.imshow(np.uint8(img))
            plt.title(class_list[prediction])
            plt.show()
    
    create_csv (results,'')
    #print (results)

def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')
