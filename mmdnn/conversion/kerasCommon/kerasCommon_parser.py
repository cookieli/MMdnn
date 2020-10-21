
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.utils import assign_IRnode_values, compute_tf_same_padding, assign_attr_value

from enum import Enum

ConvEnum = Enum(
    value='Conv',
    names=[
        ('SeparableConv', 1),
        ('ConvTranspose', 2),
        ('DepthwiseConv', 3),
        ('Conv', 4),
    ]
)

class Keras2CommonParser(Parser):

    dtype_map = {
        "float16": graph_pb2.DT_FLOAT16,
        "float32": graph_pb2.DT_FLOAT32,
        "float64": graph_pb2.DT_FLOAT64,
        "int16": graph_pb2.DT_INT16,
        "int32": graph_pb2.DT_INT32,
        "int64": graph_pb2.DT_INT64,
        "uint8": graph_pb2.DT_UINT8,
        "uint16": graph_pb2.DT_UINT16
    }

    activation_map = {
        "relu": "Relu",
        'softmax': "Softmax",
        'sigmoid': "Sigmoid",
        "tanh": "Tanh",
        "elu": "Elu",
        "relu6": "Relu6",
        'softplus': 'Softplus',
        'softsign': 'Softsign',
        'hard_sigmoid': 'HardSigmoid'
    }

    def __init__(self, model):
        pass

    def build_graph(self, graph_init, model):
        graph = graph_init(model)
        graph.build()
        return graph

    def rename_InputLayer(self, source_node):
        IR_node =  self.IR_graph.node.add()

        self._transfer_op_attr(self, IR_node, "DataInput", True)
        self.convert_inedge(source_node, IR_node)


    @classmethod
    def _transfer_op_attr(cls, source_node, IR_node, new_op = None, need_shape_attr=False):
        IR_node.name = source_node.name
        IR_node.op = source_node.type if new_op == None else new_op

        if hasattr(source_node.layer, "dtype"):
            IR_node.attr["dtype"].type = Keras2CommonParser.dtype_map[source_node.layer.dtype]

        cls._set_output_shape(source_node, IR_node, need_shape_attr)

    @classmethod
    def _set_output_shape(cls, source_node, IR_node, need_shape_attr= False):
        shape = graph_pb2.TensorShape()
        for dim in source_node.layer.output_shape:
            new_dim = shape.dim.add()
            new_dim.size = dim if dim else -1

        IR_node.attr["_output_shapes"].list.shape.extend([shape])

        if need_shape_attr:
            IR_node["shape"].shape = shape

    def _convert_convolution(self, source_node, dim):
        IR_node = self.IR_graph.node.add()
        pass

    def _convert_padding(self, source_node, IR_node):
        dims = len(source_node.layer.input_shape)
        if source_node.layer.padding == 'valid':
            assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})

        elif source_node.layer.padding == 'same':
            kernel_shape = source_node.layer.kernel_size if hasattr(source_node.layer,
                                                                    'kernel_size') else source_node.layer.pool_size
            padding = compute_tf_same_padding(
                source_node.layer.input_shape,
                kernel_shape,
                list(source_node.layer.strides))
            assign_IRnode_values(IR_node, {'auto_pad': "SAME_LOWER", 'pads': padding})

        else:
            assert False

    def _set_conv_weight(self, source_node, isSeparable=False):
        if self.weight_loaded:
            if isSeparable:
                self.set_weight(source_node.name, 'depthwise_filter', source_node.layer.get_weights()[0])
                self.set_weight(source_node.name, 'pointwise_filter', source_node.layer.get_weights()[1])
            self.set_weight(source_node.name, "weights", source_node.layer.get_weights()[0])
        if source_node.layer.use_bias:
            self.set_weight(source_node.name, "bias", source_node.layer.get_weights()[1])

    def _parse_conv_node(self, source_node):
        if source_node.type.startswith('Separable'):
            return ConvEnum.SeparableConv
        elif source_node.type.startswith('Conv'):
            if source_node.type.endswith('Transpose'):
                return ConvEnum.ConvTranspose
            return ConvEnum.Conv
        elif source_node.type.startswith('Deconv'):
            return ConvEnum.ConvTranspose
        elif source_node.type.startswith('Depthwise'):
            return ConvEnum.DepthwiseConv
        else:
            raise NotImplementedError("Convolution layer [{}] is not supported.".format(source_node.type))

    def _create_IR_conv(self, source_node, dim):
        IR_node = self.IR_graph.node.add()

        #input edge
        self.convert_inedge(source_node, IR_node)
        new_op = self._parse_conv_node(source_node)
        isSeparable = (new_op == ConvEnum.SeparableConv)
        self._transfer_op_attr(source_node, IR_node, new_op.name)
        self._set_conv_weight(source_node, isSeparable)
        self._convert_padding(source_node, IR_node)
        self._set_conv_kwargs(source_node, IR_node, dim)
        self._defuse_activation(source_node)

    def _convert_pooling(self, source_node, dim, pooling_type, is_global):
        IR_node = self.IR_graph.node.add()

        # input edge
        self.convert_inedge(source_node, IR_node)
        name = 'Pool'
        self._transfer_op_attr(source_node, IR_node, name)
        pass



    def _convert_nonGlobal_pooling(self, source_node, dim):
        layer_pool_size = source_node.layer.pool_size
        layer_strides   = source_node.layer.strides
        if isinstance(layer_pool_size, int):
            layer_pool_size *= dim
        pass 

    def get_dim_related_attr(self, layer, name, dim):
        if hasattr(layer, name):
            attr = getattr(layer, name)
            if isinstance(attr, int):
                attr *= dim
            return attr
        else:
            raise AttributeError("layer do not have this attr {}".name)





    def _defuse_activation(self, source_node):
        if source_node.layer.activation is None or source_node.layer.activation.__name__ == "linear":
            return

        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.real_name + "_activation"
        IR_node.op = self.activation_map[source_node.layer.activation.__name__]
        IR_node.input.append(source_node.real_name)
        self._set_output_shape(source_node, IR_node)

        # TODO: More activation functions
        # for ELU
        if hasattr(source_node.layer, 'alpha'):
            assign_attr_value(IR_node['alpha'], source_node.layer.alpha)

        source_node.real_name = IR_node.name



    def _set_conv_kwargs(self, source_node, IR_node, dim):
        kwargs = dict()

        layer_kernel_size = source_node.layer.kernel_size
        if isinstance(layer_kernel_size, int):
            layer_kernel_size *= dim
        layer_strides = source_node.layer.strides
        if isinstance(layer_strides, int):
            layer_strides *= dim
        layer_dilation_rate = source_node.layer.dilation_rate
        if isinstance(layer_dilation_rate, int):
            layer_dilation_rate *= dim

        in_channel = source_node.layer.input_shape[-1] if self.data_format == "channels_last" else source_node.layer.input_shape[1]
        out_channel = source_node.layer.filters or source_node.layer.depth_multiplier

        if source_node.type.startswith("Deconv"):
            kwargs['kernel_shape'] = list(layer_kernel_size) + [out_channel, in_channel]
        else:
            kwargs['kernel_shape'] = list(layer_kernel_size) + [in_channel, out_channel]

        # use_bias
        kwargs['use_bias'] = source_node.layer.use_bias

        # strides
        # [1, sd, sh, sw, 1]
        kwargs['strides'] = [1] + list(layer_strides) + [1]

        # dilations
        # [1, dd, dh, dw, 1]
        kwargs['dilations'] = [1] + list(layer_dilation_rate) + [1]

        assign_IRnode_values(IR_node, kwargs)





