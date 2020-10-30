import os
from six import string_types as _string_types
import tensorflow as tf

from mmdnn.conversion.kerasCommon.kerasCommon_parser import Keras2CommonParser
from mmdnn.conversion.tensorflow2.tf2_graph import Tf2Graph


class Tf2Parser(Keras2CommonParser):

    def __init__(self, model_name):
        super(Tf2Parser, self).__init__()
        if isinstance(model_name, _string_types):
            model = tf.keras.models.load_model(model_name)
        else:
            raise NotImplementedError('non string type model_name load not supported')

        self.weight_loaded = True
        self.graph = self.build_graph(Tf2Graph,model)
        self.data_format = tf.keras.backend.image_data_format()
        self.lambda_layer_count = 0

    @property
    def src_graph(self):
        return self.graph