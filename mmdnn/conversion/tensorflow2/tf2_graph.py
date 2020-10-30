from mmdnn.conversion.kerasCommon.kerasCommon_graph import Keras2CommonGraphNode, Keras2CommonGraph
class Tf2GraphNode(Keras2CommonGraphNode):

    def __init__(self, layer):
        super(Tf2GraphNode, self).__init__(layer)


class Tf2Graph(Keras2CommonGraph):

    def __init__(self, model):
        super(Tf2Graph, self).__init__(model)

    def _connect(self, node, layer):
        if not node.inbound_layers:
            return
        in_layer = node.inbound_layers
        self.add_layer(layer.name, Tf2GraphNode(in_layer))
        self._make_connection(in_layer.name, layer.name)
        #print("it's in graph debug")
