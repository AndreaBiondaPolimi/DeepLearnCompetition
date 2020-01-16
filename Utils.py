import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os, os.path
import cv2
import json
from sklearn.model_selection import train_test_split
import cv2
from datetime import datetime

from tensorflow.keras.preprocessing.image import img_to_array

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
        seed = 1691
        np.random.seed(seed)
        folder = 'dataset_vqa'

        with open(folder + '/train_data.json', 'r') as f:
            SUBSET_data = json.load(f)
        f.close()
        data_train = SUBSET_data['questions']

        with open(folder + '/test_data.json', 'r') as f:
            SUBSET_data = json.load(f)
        f.close()
        data_test = SUBSET_data['questions']

        self.data_for_train, self.data_for_test = self.tokenizator (data_train, data_test)

        data_train, data_valid = train_test_split(self.data_for_train, test_size=0.2, random_state=seed)

        train_dataset = DatasetIterator(data_train,self.img_h, self.img_w, folder + '\\train\\' , self.batch_size, 'none')
        train_dataset = iter(train_dataset)

        valid_dataset = DatasetIterator(data_valid,self.img_h, self.img_w, folder + '\\train\\' , self.batch_size, 'none')
        valid_dataset = iter(valid_dataset)

        """
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
        """

        return train_dataset, valid_dataset

    def tokenizator (self, data_train, data_test):
        questions_train = []
        for dt in data_train:
            questions_train.append('<sos> ' + dt['question'] + ' <eos>')

        answers_train = []
        for dt in data_train:
            answers_train.append(dt['answer'])

        questions_test = []
        for dt in data_test:
            questions_test.append('<sos> ' + dt['question'] + ' <eos>')

        #Question Tokenization
        quest_tokenizer = Tokenizer()
        quest_tokenizer.fit_on_texts(questions_train + questions_test)
        
        questions_train_tokenized = quest_tokenizer.texts_to_sequences(questions_train)
        questions_test_tokenized = quest_tokenizer.texts_to_sequences(questions_test)

        #Question Padding
        max_qst_length = max(len(sentence) for sentence in (questions_train_tokenized + questions_test_tokenized))
        qst_train_encoder_inputs = pad_sequences(questions_train_tokenized, maxlen=max_qst_length)
        qst_test_encoder_inputs = pad_sequences(questions_test_tokenized, maxlen=max_qst_length)

        
        #Saving model train parameters
        self.max_qst_length = max_qst_length
        self.quest_wtoi = quest_tokenizer.word_index

        #Answers Train Tokenization
        for i in range (len(answers_train)):
            if (answers_train[i] == '0'):
                answers_train[i] = [1,0,0,0,0,0,0,0,0,0,0,0,0]
            elif (answers_train[i] == '1'):
                answers_train[i] = [0,1,0,0,0,0,0,0,0,0,0,0,0]
            elif (answers_train[i] == '10'):
                answers_train[i] = [0,0,1,0,0,0,0,0,0,0,0,0,0]
            elif (answers_train[i] == '2'):
                answers_train[i] = [0,0,0,1,0,0,0,0,0,0,0,0,0]
            elif (answers_train[i] == '3'):
                answers_train[i] = [0,0,0,0,1,0,0,0,0,0,0,0,0]
            elif (answers_train[i] == '4'):
                answers_train[i] = [0,0,0,0,0,1,0,0,0,0,0,0,0]
            elif (answers_train[i] == '5'):
                answers_train[i] = [0,0,0,0,0,0,1,0,0,0,0,0,0]
            elif (answers_train[i] == '6'):
                answers_train[i] = [0,0,0,0,0,0,0,1,0,0,0,0,0]
            elif (answers_train[i] == '7'):
                answers_train[i] = [0,0,0,0,0,0,0,0,1,0,0,0,0]
            elif (answers_train[i] == '8'):
                answers_train[i] = [0,0,0,0,0,0,0,0,0,1,0,0,0]
            elif (answers_train[i] == '9'):
                answers_train[i] = [0,0,0,0,0,0,0,0,0,0,1,0,0]
            elif (answers_train[i] == 'no'):
                answers_train[i] = [0,0,0,0,0,0,0,0,0,0,0,1,0]
            elif (answers_train[i] == 'yes'):
                answers_train[i] = [0,0,0,0,0,0,0,0,0,0,0,0,1]
            else:
                raise ('Invalid Answer format')

        #Save tokenized data
        for i in range (len(qst_train_encoder_inputs)):
            data_train[i]['question'] = qst_train_encoder_inputs[i]
        
        for i in range (len(answers_train)):
            data_train[i]['answer'] = answers_train[i]

        for i in range (len(qst_test_encoder_inputs)):
            data_test[i]['question'] = qst_test_encoder_inputs[i]

        return data_train, data_test


    def test_model (self,model, to_show=False):
        folder = 'dataset_vqa'
        data = self.data_for_test
        
        model.load_weights('vqa_model_0.h5')

        results = {}
        for i in range(len(data)):           
            qst = data[i]['question']
            qst_id = data[i]['question_id']
            img_path = folder + '/test/' + data[i]['image_filename']

            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_w, self.img_h))

        
            #Image preparation
            img_array = img_to_array (img)
            img_array = img_array / 255 
            img_array = np.expand_dims(img_array, 0) 

            qst = np.asarray(qst)

            print (qst)
            print (qst.shape)

            #Image prediction
            res = model.predict([img_array, qst], batch_size=1)
            prediction = np.argmax(res) 

            results[str(qst_id)] = int(prediction)

            if (to_show == True):
                plt.imshow(np.uint8(img))
                #plt.title(class_list[prediction])
                plt.title(qst + ' || ' + prediction)
                plt.show()
    
        self.create_csv (results,'')



    def create_csv(self, results, results_dir='./'):
        csv_fname = 'results_'
        csv_fname += datetime.now().strftime('%b%d_%H-%M-%S') + '.csv'

        with open(os.path.join(results_dir, csv_fname), 'w') as f:

            f.write('Id,Category\n')

            for key, value in results.items():
                f.write(key + ',' + str(value) + '\n')






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

            iterator =  self.datagen.flow(img, batch_size=1)

            b_img = next(iterator)

            b_img = np.reshape (b_img, (320,480,3))

            questions.append(qst)
            images.append(b_img)
            answers.append(ans)

        questions = np.asarray(questions)
        images = np.asarray(images)
        answers = np.asarray(answers)

        #from keras.utils import to_categorical
        #answers = to_categorical(answers, num_classes=13)

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