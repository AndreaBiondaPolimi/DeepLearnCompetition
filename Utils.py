import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
import json
from sklearn.model_selection import train_test_split

from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator


#creare un iteratore che fissato il seed ritorna [question/img] e [answer] random ma fisso tutte le volte


def load_dataset():
    seed = 534
    np.random.seed(seed)
    folder = 'dataset_vqa'

    with open(folder + '/train_data.json', 'r') as f:
        SUBSET_data = json.load(f)
    f.close()

    data = SUBSET_data['questions']

    data_train, data_valid = train_test_split(data, test_size=0.2, random_state=seed)


    print (len(data_train))

    iterator = DatasetIterator(data_train, folder + '\\train\\' , 2)

    for i in range (0, 6, 1):
        print (i)
        print (next(iterator))


    


class DatasetIterator:
    def __init__(self, data, path, batch_size):
        self.data = data
        self.img_h = 320
        self.img_w = 480
        self.bs = batch_size
        self.base_path = path
        self.datagen = ImageDataGenerator(zoom_range=0.8)
        self.seed = 123

    def __iter__(self):
        return self

    def __next__(self): 
        idx = np.random.choice(len(self.data), self.bs)
        ret = []

        for i in idx:
            
            question = self.data[i]['question']
            answer = self.data[i]['answer']
            img_path = self.base_path + self.data[i]['image_filename']

            img = image.load_img(img_path, target_size=(self.img_h, self.img_w))
            img = image.img_to_array(img)
            img = self.datagen.random_transform(img, self.seed)
            
            #img = img * 255  # denormalize
            img = tf.dtypes.cast(img, tf.int32)
            plt.imshow(img)
            plt.show()

        return idx
        

load_dataset()