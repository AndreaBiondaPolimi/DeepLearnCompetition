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
    
    #model = mod.img_model(input_shape=(img_h, img_w, 3))


    mod.Word2VecModel(len(dl.quest_wtoi)+1, dl.max_qst_length, 0.2)

    #util.test_model(model,False,img_h,img_w,preprocess_type)

start()