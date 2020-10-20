import tensorflow as tf
from mmdnn.conversion.examples.extractor import get_command_line_args
networks_map = {
    'inception_v3'      : lambda : tf.keras.applications.inception_v3.InceptionV3(input_shape=(299, 299, 3)),
    'vgg16'             : lambda : tf.keras.applications.vgg16.VGG16(),
    'vgg19'             : lambda : tf.keras.applications.vgg19.VGG19(),
    'resnet'            : lambda : tf.keras.applications.resnet50.ResNet50(),
    'mobilenet'         : lambda : tf.keras.applications.mobilenet.MobileNet(),
    'xception'          : lambda : tf.keras.applications.xception.Xception(input_shape=(299, 299, 3)),
    'inception_resnet'  : lambda : tf.keras.applications.inception_resnet_v2.InceptionResNetV2()
}

def _main():
    args = get_command_line_args(networks_map.keys())

    model = networks_map.get(args.network)
    model = model()
    json_string = model.to_json()
    with open("imagenet_{}.json".format(args.network), "w") as of:
        of.write(json_string)




