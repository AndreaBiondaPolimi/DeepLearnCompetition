In order to execute the script you have to call:

python Main.py [arg1] [arg2]

where:
arg1: type of model to execute
	"resnet50" loads the ResNet50 encoder and unet decoder 
	"mobilenet" loads the MobileNet encoder and unet decoder 
	"none" loads the Unet model

arg2: if you want to train the model or to create the csv file
	"train" if you want to train the model
	"csv" if you want to create the csv starting from "seg_final.h5" file created at the end of the train phase