import os
from six import string_types as _string_types
import tensorflow as tf

from mmdnn.conversion.kerasCommon.kerasCommon_parser import Keras2CommonParser
from mmdnn.conversion.tensorflow2.tf2_graph import Tf2Graph


class Tf2Parser(Keras2CommonParser):

    def __init__(self, model_name):
        if isinstance(model_name, _string_types):
            model = tf.keras.models.load_model(model_name)
            self.tf2_graph = self.build_graph(Tf2Graph,model)

    def gen_IR(self):
        for layer in self.tf2_graph.topological_sort:
            current_node = self.tf2_graph.get_node(layer)
            node_type = current_node.type

