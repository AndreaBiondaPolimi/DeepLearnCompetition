import tensorflow as tf
import Utils as util
import Classificator as cl


print (tf.__version__)
train_dataset,valid_dataset = util.load_dataset()

model = cl.lenet_mod_model((512,512,3),20)

#cl.train (model,train_dataset,valid_dataset,500)

util.test_model (model,False)
