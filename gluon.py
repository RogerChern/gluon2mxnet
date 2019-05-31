import mxnet as mx
import gluoncv

# you can change it to your image filename
filename = 'classification-demo.png'
# you may modify it to switch to another model. The name is case-insensitive
model_name = 'ResNet152_v1b'
# download and load the pre-trained model
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)


