import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.backend import resize_images
from datetime import datetime



def load_dataset(img_h, img_w, batch_size, preprocess_type='None'):
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

    if (preprocess_type=='none'):
        train_img_data_gen = ImageDataGenerator(rotation_range=2,
                                                width_shift_range=2,
                                                height_shift_range=2,
                                                zoom_range=0.3,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                fill_mode='reflect',
                                                cval=0,
                                                validation_split=0.2,
                                                rescale=1./255)
    else:
        from tensorflow.keras.models import model_from_json
        if (preprocess_type == 'resnet50'):
            from tensorflow.keras.applications.resnet50 import preprocess_input
        if (preprocess_type == 'mobilenet'):
            from tensorflow.keras.applications.mobilenet import preprocess_input

        train_img_data_gen = ImageDataGenerator(rotation_range=2,
                                                width_shift_range=2,
                                                height_shift_range=2,
                                                zoom_range=0.3,
                                                horizontal_flip=True,
                                                vertical_flip=True,
                                                fill_mode='reflect',
                                                cval=0,
                                                validation_split=0.2,
                                                preprocessing_function=preprocess_input)   
        


    train_mask_data_gen = ImageDataGenerator(rotation_range=2,
                                             width_shift_range=2,
                                             height_shift_range=2,
                                             zoom_range=0.3,
                                             horizontal_flip=True,
                                             vertical_flip=True,
                                             fill_mode='reflect',
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



def test_model(model, to_show, img_h, img_w, preprocess_type='none'):
    path = 'Segmentation_Dataset\\test\\images\\img'
    model.load_weights('check.h5')

    results = {}

    for f in os.listdir(path):
        #Image loading
        ext = os.path.splitext(f)[1]
        img = cv2.imread(os.path.join(path,f))
        img = cv2.resize(img, (img_h,img_w))

        #Image preparation
        img_array = img_to_array (img)
        img_array = np.expand_dims(img_array, 0) 


        if (preprocess_type=='none'):
            img_array = img_array / 255 
        else:
            if (preprocess_type == 'mobilenet'):
                from tensorflow.keras.applications.mobilenet import preprocess_input
            if (preprocess_type == 'resnet50'):
                from tensorflow.keras.applications.resnet50 import preprocess_input

            img_arr = preprocess_input(img_arr)


        #Image prediction
        res = model.predict(img_array)
        
        res[np.where(res < 0.5)] = 0
        res[np.where(res >= 0.5)] = 1
        
        f_name = os.path.splitext(f)[0]
        print(f_name)
        results[str(f_name)] = rle_encode(res)

        if (to_show == True):
            res = res * 255
            f = plt.figure()
            f.add_subplot(1,2, 1)
            plt.imshow(img)
            f.add_subplot(1,2, 2)
            plt.imshow(np.reshape(res,(img_h,img_w)))
            plt.show(block=True)

    
    create_csv (results,'')
    #print (results)


def create_csv(results, results_dir='./'):
    print ('creating csv..')
    csv_fname = 'results_'
    csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

    with open(csv_fname, 'w') as f:

      f.write('ImageId,EncodedPixels,Width,Height\n')

      for key, value in results.items():
          f.write(key + ',' + str(value) + ',' + '256' + ',' + '256' + '\n')


def rle_encode(img):
      # Flatten column-wise
      pixels = img.T.flatten()
      pixels = np.concatenate([[0], pixels, [0]])
      runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
      runs[1::2] -= runs[::2]
      return ' '.join(str(x) for x in runs)
