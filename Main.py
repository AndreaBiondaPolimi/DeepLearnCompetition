import Utils as util
import Segmentator as seg

img_h=256
img_w=256
batch_size=1

#train, valid = util.load_dataset(img_h,img_w,batch_size)

model = seg.unet_model()

#seg.train(model, train, valid, 20)

util.test_model(model,False,img_h,img_w)