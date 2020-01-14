from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM, Flatten, Embedding
from tensorflow.keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, 

def Word2VecModel(num_words, seq_length, dropout_rate):
    EMBEDDING_SIZE = 512

    model = Sequential()
    model.add(Embedding(num_words, EMBEDDING_SIZE, input_length=seq_length))
    model.add(LSTM(units=512, return_sequences=True, input_shape=[seq_length]))
    model.add(Dropout(dropout_rate))
    model.add(LSTM(units=512, return_sequences=False))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1024, activation='tanh'))

    model.summary()
    return model



from tensorflow.keras.applications import ResNet50   
def img_model(input_shape):
    model = ResNet50(input_shape=input_shape, weights='imagenet', include_top=False)
    x = model.output
    x = Flatten()(x)
    x = Dense(500, activation='tanh')(x)

    model = Model(inputs=model.input, outputs=x)

    model.summary()
    return model




def vqa_model(input_shape, num_words, seq_length, dropout_rate, num_classes):
    resnet_model = img_model(input_shape)
    lstm_model = Word2VecModel(num_words, seq_length, dropout_rate)

    fc_model = Sequential()
    fc_model.add(Merge([resnet_model, lstm_model], mode='mul'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(1000, activation='tanh'))
    fc_model.add(Dropout(dropout_rate))
    fc_model.add(Dense(num_classes, activation='softmax'))
    fc_model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
        metrics=['accuracy'])
    return fc_model
