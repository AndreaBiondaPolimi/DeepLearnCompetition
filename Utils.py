import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
import json
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from tensorflow.keras.applications.resnet50 import preprocess_input


class DataLoader:
    def __init__(self, img_h, img_w, batch_size):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size

    def load_dataset(self,):
        seed = 534
        np.random.seed(seed)
        folder = 'dataset_vqa'

        with open(folder + '/train_data.json', 'r') as f:
            SUBSET_data = json.load(f)
        f.close()

        data = SUBSET_data['questions']

        data = self.tokenizator (data)

        data_train, data_valid = train_test_split(data, test_size=0.2, random_state=seed)

        train_dataset = DatasetIterator(data_train,self.img_h, self.img_w, folder + '\\train\\' , self.batch_size, 'none')
        train_dataset = iter(train_dataset)

        valid_dataset = DatasetIterator(data_valid,self.img_h, self.img_w, folder + '\\train\\' , self.batch_size, 'none')
        valid_dataset = iter(valid_dataset)

        
        for i in range (0, 3, 1):
            print ()
            [questions, images], answers = next(train_dataset)

            qst = questions[0]
            img = images [0]
            ans = answers[0]

            print (qst)
            print (ans)

            img = img * 255  # denormalize
            #img = preprocess_input(img) # denormalize
            img = tf.dtypes.cast(img, tf.int32)
            plt.imshow(np.reshape (img, (320,480,3)))
            plt.show()
            
    
        for i in range (0, 3, 1):
            print ()
            [questions, images], answers = next(valid_dataset)

            qst = questions[0]
            img = images [0]
            ans = answers[0]

            print (qst)
            print (ans)

            img = img * 255  # denormalize
            #img = preprocess_input(img) # denormalize
            img = tf.dtypes.cast(img, tf.int32)
            plt.imshow(np.reshape (img, (320,480,3)))
            plt.show()
        

        return train_dataset, valid_dataset

    def tokenizator (self, data):
        questions = []
        for dt in data:
            questions.append('<sos> ' + dt['question'] + ' <eos>')

        answers = []
        for dt in data:
            answers.append(dt['answer'])

        #Questions Tokenization
        quest_tokenizer = Tokenizer()
        quest_tokenizer.fit_on_texts(questions)
        questions_tokenized = quest_tokenizer.texts_to_sequences(questions)
        max_qst_length = max(len(sentence) for sentence in questions_tokenized)
        qst_encoder_inputs = pad_sequences(questions_tokenized, maxlen=max_qst_length)
        
        self.max_qst_length = max_qst_length
        self.quest_wtoi = quest_tokenizer.word_index

        #Answers Tokenization
        for i in range (len(answers)):
            if (answers[i] == '0'):
                answers[i] = 0
            elif (answers[i] == '1'):
                answers[i] = 1
            elif (answers[i] == '10'):
                answers[i] = 2
            elif (answers[i] == '2'):
                answers[i] = 3
            elif (answers[i] == '3'):
                answers[i] = 4
            elif (answers[i] == '4'):
                answers[i] = 5
            elif (answers[i] == '5'):
                answers[i] = 6
            elif (answers[i] == '6'):
                answers[i] = 7
            elif (answers[i] == '7'):
                answers[i] = 8
            elif (answers[i] == '8'):
                answers[i] = 9
            elif (answers[i] == '9'):
                answers[i] = 10
            elif (answers[i] == 'no'):
                answers[i] = 11
            elif (answers[i] == 'yes'):
                answers[i] = 12
            else:
                raise ('Invalid Answer format')

        #print (data[0]['question'])
        #print (data[1]['question'])

        for i in range (len(qst_encoder_inputs)):
            data[i]['question'] = qst_encoder_inputs[i]

        #print (data[0]['question'])
        #print (data[1]['question'])

        #print ()

        #print (data[0]['answer'])
        #print (data[1]['answer'])
        
        for i in range (len(answers)):
            data[i]['answer'] = answers[i]

        #print (data[0]['answer'])
        #print (data[1]['answer'])

        return data


    


class DatasetIterator:
    def __init__(self, data, img_h, img_w, path, batch_size, image_preprocessing='none'):
        self.data = data
        self.img_h = img_h
        self.img_w = img_w
        self.bs = batch_size
        self.base_path = path
        self.seed = 534

        if (image_preprocessing == 'resnet50'):
            self.datagen = ImageDataGenerator(rotation_range=8,
                                                width_shift_range=8,
                                                height_shift_range=8,
                                                zoom_range=0.3,
                                                cval=0,
                                                preprocessing_function=preprocess_input)
            
        else:
            self.datagen = ImageDataGenerator(rotation_range=8,
                                                width_shift_range=8,
                                                height_shift_range=8,
                                                zoom_range=0.3,
                                                cval=0,
                                                rescale=1./255)
            

        

    def __iter__(self):
        return self

    def __next__(self): 
        idx = np.random.choice(len(self.data), self.bs)
        questions = []
        images = []
        answers = []
        
        for i in idx:           
            qst = self.data[i]['question']
            ans = self.data[i]['answer']
            img_path = self.base_path + self.data[i]['image_filename']

            img = image.load_img(img_path, target_size=(self.img_h, self.img_w))
            img = np.expand_dims(image.img_to_array(img), axis=0)
            #img = self.datagen.random_transform(img, self.seed)

            iterator =  self.datagen.flow(img, batch_size=1)
            b_img = next(iterator)

            questions.append(qst)
            images.append(b_img)
            answers.append(ans)

        return [questions, images] , answers




"""

def load_dataset(img_h, img_w, batch_size):
    seed = 534
    np.random.seed(seed)
    folder = 'dataset_vqa'

    with open(folder + '/train_data.json', 'r') as f:
        SUBSET_data = json.load(f)
    f.close()

    data = SUBSET_data['questions']

    data = tokenizator (data)

    data_train, data_valid = train_test_split(data, test_size=0.2, random_state=seed)

    train_dataset = DatasetIterator(data_train,img_h, img_w, folder + '\\train\\' , batch_size)
    train_dataset = iter(train_dataset)

    valid_dataset = DatasetIterator(data_valid,img_h, img_w, folder + '\\train\\' , batch_size)
    valid_dataset = iter(valid_dataset)
    
    # for i in range (0, 3, 1):
    #     print ()
    #     [questions, images], answers = next(train_dataset)

    #     qst = questions[0]
    #     img = images [0]
    #     ans = answers[0]

    #     print (qst)
    #     print (ans)

    #     img = img * 255  # denormalize
    #     img = tf.dtypes.cast(img, tf.int32)
    #     plt.imshow(np.reshape (img, (320,480,3)))
    #     plt.show()
        
    
    # for i in range (0, 3, 1):
    #     print ()
    #     [questions, images], answers = next(valid_dataset)

    #     qst = questions[0]
    #     img = images [0]
    #     ans = answers[0]

    #     print (qst)
    #     print (ans)

    #     img = img * 255  # denormalize
    #     img = tf.dtypes.cast(img, tf.int32)
    #     plt.imshow(np.reshape (img, (320,480,3)))
    #     plt.show()

    return train_dataset, valid_dataset



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
def tokenizator (data):
    questions = []
    for dt in data:
        questions.append('<sos> ' + dt['question'] + ' <eos>')

    answers = []
    for dt in data:
        answers.append(dt['answer'])

    #Questions Tokenization
    quest_tokenizer = Tokenizer()
    quest_tokenizer.fit_on_texts(questions)
    questions_tokenized = quest_tokenizer.texts_to_sequences(questions)
    max_qst_length = max(len(sentence) for sentence in questions_tokenized)
    qst_encoder_inputs = pad_sequences(questions_tokenized, maxlen=max_qst_length)
    

    #Answers Tokenization
    for i in range (len(answers)):
        if (answers[i] == '0'):
            answers[i] = 0
        elif (answers[i] == '1'):
            answers[i] = 1
        elif (answers[i] == '10'):
            answers[i] = 2
        elif (answers[i] == '2'):
            answers[i] = 3
        elif (answers[i] == '3'):
            answers[i] = 4
        elif (answers[i] == '4'):
            answers[i] = 5
        elif (answers[i] == '5'):
            answers[i] = 6
        elif (answers[i] == '6'):
            answers[i] = 7
        elif (answers[i] == '7'):
            answers[i] = 8
        elif (answers[i] == '8'):
            answers[i] = 9
        elif (answers[i] == '9'):
            answers[i] = 10
        elif (answers[i] == 'no'):
            answers[i] = 11
        elif (answers[i] == 'yes'):
            answers[i] = 12
        else:
            raise ('Invalid Answer format')

    #print (data[0]['question'])
    #print (data[1]['question'])

    for i in range (len(qst_encoder_inputs)):
        data[i]['question'] = qst_encoder_inputs[i]

    #print (data[0]['question'])
    #print (data[1]['question'])

    #print ()

    #print (data[0]['answer'])
    #print (data[1]['answer'])
    
    for i in range (len(answers)):
        data[i]['answer'] = answers[i]

    #print (data[0]['answer'])
    #print (data[1]['answer'])

    return data
"""