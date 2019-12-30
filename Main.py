import Utils as util
import sys
import tensorflow as tf

def start ():

    img_h=320
    img_w=480
    batch_size=8

    #train, valid = util.load_dataset(img_h,img_w,batch_size,preprocess_type)
    train, valid = util.load_dataset(img_h,img_w,batch_size)



    #util.test_model(model,False,img_h,img_w,preprocess_type)

start()