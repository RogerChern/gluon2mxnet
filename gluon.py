import mxnet as mx
import gluoncv

# you may modify it to switch to another model. The name is case-insensitive
model_name = 'resnet101_v1d'
# download and load the pre-trained model
net = gluoncv.model_zoo.get_model(model_name, pretrained=True)
# hybridize
net.hybridize()
# run forward pass to obtain the predicted score for each class
pred = net.forward(mx.nd.zeros(shape=[1, 3, 224, 224]))
# export pretrain model
net.export(model_name, epoch=0)
