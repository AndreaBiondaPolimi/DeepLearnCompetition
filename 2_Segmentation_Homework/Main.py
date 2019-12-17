import Utils as util
import Segmentator as seg
import sys
import tensorflow as tf



#preprocess_type = 'resnet50' #resnet50 encoder preporcessing + unet decoder
#preprocess_type = 'mobilenet' #mobilenet encoder preporcessing + unet decoder
#preprocess_type = 'none' #unet model without preprocessing

def start ():

    preprocess_type = sys.argv[1]
    action = sys.argv[2]
    
    img_h=256
    img_w=256
    batch_size=8

    if (action == 'train'):

        train, valid = util.load_dataset(img_h,img_w,batch_size,preprocess_type)

        model = seg.get_segmentation_model(preprocess_type=preprocess_type)

        seg.train(model, train, valid, 50)

        #util.test_model(model,False,img_h,img_w,preprocess_type)

    elif (action == 'csv'):
        model = seg.get_segmentation_model(preprocess_type=preprocess_type)

        util.test_model(model,False,img_h,img_w,preprocess_type)

    else:
        print ("No available action selected")
   

start()