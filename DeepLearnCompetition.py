import tensorflow as tf
import Utils as util
import Classificator as cl
import sys

def start ():

    model_type = sys.argv[1]
    action = sys.argv[2]
    
    print (model_type)
    print (action)

    print (tf.__version__)

    img_h = 400
    img_w = 400

    train_dataset,valid_dataset = util.load_dataset(img_h,img_w)


    if (model_type == 'lenet_mine'):# ~40% accuracy
        model = cl.lenet_mod_model((img_h,img_w,3),20)
    elif (model_type == 'resnet_mine'):# ~55% accuracy
        model = cl.ResNet50((img_h,img_w,3),20)
    elif (model_type == 'resnet_tr_lr'):# ~70% accuracy
        model = cl.ResNet50_transf_learning((img_h,img_w,3),20)
    else:
        print ('No available model selected')


    if (action == 'train'):
        cl.train (model,train_dataset,valid_dataset,200)
    elif (action == 'csv'):
        util.test_model (model,False,img_h,img_w)
    else:
        print ("No available action selected")

   

start()