from mmdnn.conversion.common.DataStructure.graph import GraphNode, Graph


class Keras2CommonGraphNode(GraphNode):

    def __init__(self, layer):
        super(Keras2CommonGraphNode, self).__init__(layer)

    @property
    def name(self):
        return self.layer.name

    @property
    def type(self):
        return self.layer.__class__.__name__

class Keras2CommonGraph(Graph):

    def __init__(self, model):
        # sanity check.
        super(Keras2CommonGraph, self).__init__(model)
        self.model = model

    def _connect(self, node, layer):
        pass

    def build(self):
        self.input_layers = list()
        for i, layer in enumerate(self.model.layers):
            self.layer_map[layer.name] = Keras2CommonGraphNode(layer)
            self.layer_name_map[layer.name] = layer.name
            for node in layer._inbound_nodes:
                self._connect(node, layer)
                # for pred in node.inbound_layers:
                #     if pred.name not in self.layer_map:
                #         self.layer_map[pred.name] = Keras2CommonGraphNode(pred)
                #         self.layer_name_map[pred.name] = pred.name
                #     self._make_connection(pred.name, layer.name)

        # Kit: TODO
        # Duplicate models for weight sharing
        # Expand the sub-models
        super(Keras2CommonGraph, self).build()
