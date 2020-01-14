from Utils import DataLoader
import sys
import tensorflow as tf
import Model as mod

def start ():

    img_h=320
    img_w=480
    batch_size=8

    dl = DataLoader(img_h, img_w, batch_size)

    #train, valid = util.load_dataset(img_h,img_w,batch_size,preprocess_type)
    train, valid = dl.load_dataset()
    
    num_words = len(dl.quest_wtoi)+1
    seq_length = dl.max_qst_length
    dropout_rate = 0.2
    input_shape = (img_h, img_w, 3)
    num_classes = 13

    #model = mod.vqa_model(input_shape, num_words, seq_length, dropout_rate, num_classes)

    #mod.train(model, train, valid, 100, batch_size)

    dl.test_model()

start()