import tensorflow as tf
import Utils as util
import Classificator as cl


print (tf.__version__)

img_h = 512
img_w = 512

train_dataset,valid_dataset = util.load_dataset(img_h,img_w)



#model = cl.lenet_mod_model((512,512,3),20)

model = cl.ResNet50((img_h,img_w,3),20)


cl.train (model,train_dataset,valid_dataset,300)

util.test_model (model,False,img_h,img_w)
