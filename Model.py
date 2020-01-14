from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D, Multiply
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import numpy as np

def Word2VecModel(num_words, seq_length, dropout_rate):
    EMBEDDING_SIZE = 512

    model = Sequential()
    model.add(Embedding(num_words, EMBEDDING_SIZE, input_length=seq_length))
    model.add(LSTM(units=512, return_sequences=True, input_shape=[seq_length]))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))

    #model.summary()
    return model



from tensorflow.keras.applications import ResNet50   
def img_model(input_shape):
    model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    x = model.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dense(1024, activation='tanh')(x)

    model = Model(inputs=model.input, outputs=x)

    #model.summary()
    return model




def vqa_model(input_shape, num_words, seq_length, dropout_rate, num_classes):
    resnet_model = img_model(input_shape)
    lstm_model = Word2VecModel(num_words, seq_length, dropout_rate)

    fc_model = Multiply()([resnet_model.output, lstm_model.output])
    fc_model = Dropout(dropout_rate) (fc_model)

    fc_model = Dense(1000, activation='tanh')(fc_model)
    fc_model = Dropout(dropout_rate) (fc_model)

    fc_model = Dense(num_classes, activation='softmax')(fc_model)

    fc_model = Model(inputs=[resnet_model.input, lstm_model.input], outputs=fc_model)
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])

    fc_model.summary()
    return fc_model


from tqdm import tqdm
from tqdm import trange

def train (model, train_dataset, valid_dataset, epochs, batch_size):
    callbacks = []

    checkpoint_loss = ModelCheckpoint(filepath='check_val_loss{epoch:02d}.h5', monitor='val_loss',mode='min', period=1, save_best_only=True) 
    checkpoint_iou = ModelCheckpoint(filepath='check_val_iou{epoch:02d}.h5', monitor='val_my_IoU',mode='max', period=1, save_best_only=True) 


    #callbacks.append(es_callback)
    callbacks.append(checkpoint_loss)
    callbacks.append(checkpoint_iou)

    steps_per_epochs = 300
    validation_steps = 200

    

    for i in range (epochs):
        print ('Epoch: ' + str(i) + '/' + str(epochs))
        loss = 0
        accuracy = 0

        #Train
        t_train = trange(steps_per_epochs, desc='Bar desc', leave=True)
        for j in t_train:
            [questions, images], answers = next(train_dataset)
            
            l, a = model.train_on_batch([images, questions], answers)
            loss += l
            accuracy += a

            loss_print = round(loss/(j+1), 4)
            accuracy_print = round(accuracy/(j+1), 4)
            t_train.set_description('Train_Loss: ' + str(loss_print) + ' , Train_Accuracy: ' + str(accuracy_print))    


        #Validation
        accuracy = 0
        t_val = trange(validation_steps, desc='Bar desc', leave=True)
        for j in t_val:
            [questions, images], answers = next(valid_dataset)
            
            res = model.predict([images, questions],  batch_size=batch_size)
            
            acc = 0
            for b in range (batch_size):
                prediction = np.argmax(res[b]) 
                acc += answers[b][prediction]

                #print (prediction)
                #print (answers[b])
                #print ()

            accuracy += (acc/batch_size)            
            accuracy_print = round(accuracy/(j+1), 4)


            t_val.set_description('Valid_Accuracy: ' + str(accuracy_print))    



        #loss = loss / steps_per_epochs
        #accuracy = accuracy / steps_per_epochs
        #print ('epochs ' + str(i) + ' of ' + str(epochs) + '---> loss ' + str(loss) + ' accuracy ' + str(accuracy))

    """
    model.fit(x=train_dataset,
          epochs=epochs,  #### set repeat in training dataset
          steps_per_epoch=300,
          validation_data=valid_dataset,
          validation_steps=200,
          callbacks=callbacks)
    """