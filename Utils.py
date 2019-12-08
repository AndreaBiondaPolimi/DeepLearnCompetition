import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import resize_images
from datetime import datetime



def load_dataset(img_h, img_w, batch_size):
    # Set the seed for random operations. 
    # This let our experiments to be reproducible. 
    SEED = 1234
    tf.random.set_seed(SEED)  

    #Target directory
    dataset_dir = "Segmentation_Dataset"

    # Batch size
    bs = batch_size

    #number of classes
    num_classes=2

    train_img_data_gen = ImageDataGenerator(rotation_range=10,
                                            width_shift_range=10,
                                            height_shift_range=10,
                                            zoom_range=0.3,
                                            horizontal_flip=True,
                                            vertical_flip=True,
                                            fill_mode='constant',
                                            cval=0,
                                            validation_split=0.2,
                                            rescale=1./255)


    train_mask_data_gen = ImageDataGenerator(rotation_range=10,
                                             width_shift_range=10,
                                             height_shift_range=10,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='constant',
                                             cval=0,
                                             validation_split=0.2,
                                             rescale=1./255)

    # Training
    # Two different generators for images and masks
    # ATTENTION: here the seed is important!! We have to give the same SEED to both the generator
    # to apply the same transformations/shuffling to images and corresponding masks
    training_dir = os.path.join(dataset_dir, 'training')
    
    ### TRAINING FLOW FROM DIRECTORY ###
    train_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED,
                                                       subset='training')  
    
    train_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         color_mode='grayscale',
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED,
                                                         subset='training')


    ### VALIDATION FLOW FROM DIRECOTRY ###
    valid_img_gen = train_img_data_gen.flow_from_directory(os.path.join(training_dir, 'images'),
                                                       target_size=(img_h, img_w),
                                                       batch_size=bs, 
                                                       class_mode=None, # Because we have no class subfolders in this case
                                                       shuffle=True,
                                                       interpolation='bilinear',
                                                       seed=SEED,
                                                       subset='validation')  
    
    valid_mask_gen = train_mask_data_gen.flow_from_directory(os.path.join(training_dir, 'masks'),
                                                         target_size=(img_h, img_w),
                                                         batch_size=bs,
                                                         class_mode=None, # Because we have no class subfolders in this case
                                                         color_mode='grayscale',
                                                         shuffle=True,
                                                         interpolation='bilinear',
                                                         seed=SEED,
                                                         subset='validation')

    

                                         
    train_gen = zip(train_img_gen, train_mask_gen)
    valid_gen = zip(valid_img_gen, valid_mask_gen)

    train_dataset = tf.data.Dataset.from_generator(lambda: train_gen,
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 1]))

    valid_dataset = tf.data.Dataset.from_generator(lambda: valid_gen, 
                                               output_types=(tf.float32, tf.float32),
                                               output_shapes=([None, img_h, img_w, 3], [None, img_h, img_w, 1]))

    def prepare_target(x_, y_):
        y_ = tf.cast(y_, tf.int32)
        return x_, y_

    train_dataset = train_dataset.map(prepare_target)
    train_dataset = train_dataset.repeat()

    valid_dataset = valid_dataset.map(prepare_target)
    valid_dataset = valid_dataset.repeat()

    """
    
    iterator = iter(train_dataset)
    
    for _ in range(3):
        augmented_img, target = next(iterator)    
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize
        augmented_img = tf.dtypes.cast(augmented_img, tf.int32)

        target_img = target[0]   # First element
        target_img = target_img * 255  # denormalize

        
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(augmented_img)
        f.add_subplot(1,2, 2)
        plt.imshow(np.reshape(target_img,(img_h,img_w)))
        plt.show(block=True)
        

    iterator = iter(valid_dataset)
  
    for _ in range(3):
        augmented_img, target = next(iterator)    
        augmented_img = augmented_img[0]   # First element
        augmented_img = augmented_img * 255  # denormalize
        augmented_img = tf.dtypes.cast(augmented_img, tf.int32)

        target_img = target[0]   # First element
        target_img = target_img * 255  # denormalize

        
        f = plt.figure()
        f.add_subplot(1,2, 1)
        plt.imshow(augmented_img)
        f.add_subplot(1,2, 2)
        plt.imshow(np.reshape(target_img,(img_h,img_w)))
        plt.show(block=True)
    """
    
    return train_dataset,valid_dataset



"""
def test_model(model,to_show,img_h,img_w):
    path = "Segmentation_Dataset\\test"
    
    test_img_dir = os.path.join(test_dir, 'images', 'img')

    results = {}
    for img_filename in img_filenames:
    
        mask_filename = img_filename[:-4] + '.png'
        
        img = Image.open(os.path.join(test_img_dir, img_filename))
        
        img_arr = np.expand_dims(np.array(img), 0)
        
        out_softmax = model.predict(x=img_arr / 255.)
        
        # Get predicted class as the index corresponding to the maximum value in the vector probability
        predicted_class = tf.argmax(out_softmax, -1)
        predicted_class = predicted_class[0]
        
        target = np.array(mask)
        target -= 1    ## to get classes 0,1,2 instead of 1,2,3
        
        print(target.shape)
        
        # Assign colors (just for visualization)
        target_img = np.zeros([target.shape[0], target.shape[1], 3])
        prediction_img = np.zeros([target.shape[0], target.shape[1], 3])
        
        target_img[np.where(target == 0)] = colors_dict[0]
        target_img[np.where(target == 1)] = colors_dict[1]
        target_img[np.where(target == 2)] = colors_dict[2]
        
        prediction_img[np.where(predicted_class == 0)] = colors_dict[0]
        prediction_img[np.where(predicted_class == 1)] = colors_dict[1]
        prediction_img[np.where(predicted_class == 2)] = colors_dict[2]
        
        ax[0].imshow(np.uint8(img_arr[0, ...]))
        ax[1].imshow(np.uint8(target_img))
        ax[2].imshow(np.uint8(prediction_img))
    
    create_csv (results,'')
    #print (results)

def create_csv(results, results_dir='./'):

    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(os.path.join(results_dir, csv_fname), 'w') as f:

        f.write('Id,Category\n')

        for key, value in results.items():
            f.write(key + ',' + str(value) + '\n')
"""