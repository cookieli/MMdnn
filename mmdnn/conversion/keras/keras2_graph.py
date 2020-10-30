#----------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License. See License.txt in the project root for license information.
#----------------------------------------------------------------------------------------------
import os
import keras as _keras
from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph
from mmdnn.conversion.kerasCommon.kerasCommon_graph import Keras2CommonGraphNode, Keras2CommonGraph


class Keras2GraphNode(Keras2CommonGraphNode):

    def __init__(self, layer):
        super(Keras2GraphNode, self).__init__(layer)

    @property
    def keras_layer(self):
        return self.layer



class Keras2Graph(Keras2CommonGraph):

    def __init__(self, model):
        # sanity check.
        if not (type(model) == _keras.models.Sequential or type(model) == _keras.models.Model):
            raise TypeError("Keras layer of type %s is not supported." % type(model))
        super(Keras2Graph, self).__init__(model)

    def _connect(self, node, layer):
        for pred in node.inbound_layers:
            if pred.name not in self.layer_map:
                # self.layer_map[pred.name] = Keras2GraphNode(pred)
                # self.layer_name_map[pred.name] = pred.name
                self.add_layer(pred.name, Keras2GraphNode(pred))
            self._make_connection(pred.name, layer.name)

    def build(self):
        # Kit: TODO
        # Duplicate models for weight sharing
        # Expand the sub-models
        super(Keras2Graph, self).build()

    @property
    def src_graph(self):
        return self.graph