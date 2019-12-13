import Utils as util
import Segmentator as seg

img_h=256
img_w=256
batch_size=8

preprocess_type = 'resnet50' #resnet50 encoder preporcessing + unet decoder
#preprocess_type = 'mobilenet' #mobilenet encoder preporcessing + unet decoder
#preprocess_type = 'none' #unet model without preprocessing

train, valid = util.load_dataset(img_h,img_w,batch_size,preprocess_type)

model = seg.get_segmentation_model(preprocess_type=preprocess_type)

seg.train(model, train, valid, 40)

util.test_model(model,True,img_h,img_w,preprocess_type)