
import mmdnn.conversion.common.IR.graph_pb2 as graph_pb2
from mmdnn.conversion.common.DataStructure.parser import Parser
from mmdnn.conversion.common.Runtime.env import Env
from mmdnn.conversion.common.utils import assign_IRnode_values, compute_tf_same_padding, assign_attr_value, ConvEnum
from mmdnn.conversion.tensorflow2.attr_policy import execute_to_kwargs_policy
from mmdnn.conversion.tensorflow2.tf2_op_map import op_ir_name, op_attr_map

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

    def __init__(self):
        super(Keras2CommonParser, self).__init__()

    def build_graph(self, graph_init, model):
        graph = graph_init(model)
        graph.build()
        return graph

    def gen_IR(self):
        for layer in self.graph.topological_sort:
            current_node =  self.graph.get_node(layer)
            self.gen_IR_from(current_node)

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
        tmp_shape = source_node.layer.output_shape
        if isinstance(tmp_shape, list):
            output_shape = tmp_shape[0]
        else:
            output_shape = tmp_shape
        for dim in output_shape:
            new_dim = shape.dim.add()
            new_dim.size = dim if dim else -1
        if need_shape_attr:
            for dim in output_shape:
                new_dim = IR_node.attr["shape"].shape.dim.add()
                new_dim.size = -1 if dim == None else dim
        else:
            IR_node.attr['shape'].shape.unknown_rank = True
        IR_node.attr["_output_shapes"].list.shape.extend([shape])

    def get_dim_related_attr(self,layer, name, dim):
        if hasattr(layer, name):
            attr = getattr(layer, name)
            if isinstance(attr, int):
                return (attr,) * dim
            return attr
        else:
            raise AttributeError("layer do not have this attr {}".name)

    #TODO if you want to add separate padding layer to ir, you should add it to all
    def add_padding_layer(self):
        pass

    def _convert_padding(self, source_node, IR_node):
        dims = len(source_node.layer.input_shape)
        if source_node.layer.padding == 'valid':
            assign_IRnode_values(IR_node, {'auto_pad': "VALID", 'pads': [0, 0] * dims})

        elif source_node.layer.padding == 'same':
            kernel_shape = source_node.layer.kernel_size if hasattr(source_node.layer,
                                                                    'kernel_size') else source_node.layer.pool_size
            # padding = compute_tf_same_padding(
            #     source_node.layer.input_shape,
            #     kernel_shape,
            #     list(source_node.layer.strides))
            is_channels_first = source_node.layer.data_format == 'channels_first'
            is_channels_last  = source_node.layer.data_format == 'channels_last'
            if is_channels_first:
                input_shape = source_node.layer.input_shape[2:]
            elif is_channels_last:
                input_shape = source_node.layer.input_shape[1:-1]
            else:
                raise NotImplementedError('the data_format not in range')
            strides = source_node.layer.strides
            padding = []
            for i in range(len(input_shape)):
                if input_shape[i] % strides[i] == 0:
                    pad = max(kernel_shape[i] - strides[i], 0)
                else:
                    pad = max(kernel_shape[i] - (input_shape[i] % strides[i]))
                pad_left  = pad // 2
                pad_right = pad - pad_left
                padding.extend([pad_left, pad_right])
            if is_channels_first:
                padding = [0,0,0,0] + padding
            else:
                padding = [0,0] + padding + [0,0]
            assign_IRnode_values(IR_node, {'auto_pad': "SAME_LOWER", 'pads': padding})
        else:
            assert False


    def _set_weight(self, source_node, isSeparable=False):
        if self.weight_loaded:
            if isSeparable:
                self.set_weight(source_node.name, 'depthwise_filter', source_node.layer.get_weights()[0])
                self.set_weight(source_node.name, 'pointwise_filter', source_node.layer.get_weights()[1])
            self.set_weight(source_node.name, "weights", source_node.layer.get_weights()[0])
        if source_node.layer.use_bias:
            self.set_weight(source_node.name, "bias", source_node.layer.get_weights()[1])

    @classmethod
    def _parse_conv_node(cls, source_node):
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


    def gen_IR_from(self, source_node):
        IR_node = self.IR_graph.node.add()
        self.convert_inedge(source_node, IR_node)
        node_type = source_node.type
        if 'InputLayer' in node_type:
            self._transfer_op_attr(source_node, IR_node, "DataInput", True)
        elif 'Conv' in node_type:
            self._set_op_attr(source_node, IR_node, op_attr_map['Conv'])
            # dim_idx = node_type.find('Conv') + 4 # 4 is conv length, conv node_type often is Conv2d
            # dim     = int(node_type[dim_idx])
            # self._create_IR_conv(source_node, dim, IR_node)
        elif 'Pool' in node_type:
           dim, is_global, pooling_type = self.parse_pool_type(node_type)
           self._convert_pooling(source_node, dim, pooling_type, is_global, IR_node)
        elif 'Dense' in node_type:
            self._set_op_attr(source_node, IR_node, op_attr_map['Dense'])
        elif 'Flatten' in node_type:
            self._set_op_attr(source_node, IR_node, op_attr_map['Flatten'])
            # self._transfer_op_attr(source_node, IR_node)
            # assign_IRnode_values(IR_node, {'data_format': source_node.layer.data_format})
        elif 'ZeroPadding'  in node_type:
            self._transfer_op_attr(source_node, IR_node)
            assign_IRnode_values(IR_node, {'data_format': source_node.layer.data_format})

        else:
            raise NotImplementedError('it\'s not implemented {}'.format(node_type))

    def _set_Dense_attr(self, source_node, IR_node):
        self._transfer_op_attr(source_node, IR_node, 'FullyConnected')
        IR_node.attr['units'].i    = source_node.layer.units
        IR_node.attr['use_bias'].b = source_node.layer.use_bias

        #weights
        self._set_weight(source_node)

        #activation
        self._defuse_activation(source_node)


    def _set_op_attr(self, source_node, IR_node, attrs):
        op_name = source_node.type
        is_separable_conv = 'SeparableConv' in op_name
        if op_name in op_ir_name:
            op_name = op_ir_name[op_name]
        self._transfer_op_attr(source_node, IR_node, op_name)
        kwargs = {}
        for attr in attrs:
            if attr != 'weight' and attr != 'activation':
                execute_to_kwargs_policy(source_node, attr, kwargs)
        assign_IRnode_values(IR_node, kwargs)
        self._set_weight_activation(attrs, source_node, is_separable_conv)

    def set_op_attr(self, source_node, IR_node, env:Env):
        op_exe, kwargs = env.evaluate(source_node)
        self._transfer_op_attr(source_node, IR_node, op_exe.unify_name)
        assign_IRnode_values(IR_node, kwargs)
        if op_exe.has_weight:
            self._set_weight(source_node, op_exe.is_separable_conv)
        if op_exe.has_activaiton:
            self._defuse_activation(source_node)

    def _set_weight_activation(self, attrs, source_node, is_separable_conv=False):
        if 'weight' in attrs:
            self._set_weight(source_node, is_separable_conv)
        if 'activation' in attrs:
            self._defuse_activation(source_node)

    @classmethod
    def parse_pool_type(cls, node_type):
        tmp_idx = node_type.find('Pool')
        dim = int(node_type[tmp_idx + 7])
        pooling_type = None
        is_global = False
        if 'Global' in node_type:
            is_global = True
        if 'Max' in node_type:
            pooling_type = 'MAX'
        elif 'Average' in node_type:
            pooling_type = 'AVG'
        return dim, is_global, pooling_type

    def _create_IR_conv(self, source_node, dim, IR_node):
        new_op = self._parse_conv_node(source_node)
        isSeparable = (new_op == ConvEnum.SeparableConv)
        self._transfer_op_attr(source_node, IR_node, new_op.name)#fixme
        self._set_weight(source_node, isSeparable)
        self._convert_padding(source_node, IR_node)
        self._set_conv_kwargs(source_node, IR_node, dim)
        self._defuse_activation(source_node)

    def _convert_pooling(self, source_node, dim, pooling_type, is_global, IR_node):
        name = 'Pool'
        self._transfer_op_attr(source_node, IR_node, name)
        kwargs = {'pooling_type': pooling_type, 'dim': dim}
        # global kernel size is H,W
        if is_global:
            kwargs['global_pooling'] = True
            kwargs['strides']        = [1] * (dim + 2)

            #add flatten node
            flatten_node = self.IR_graph.node.add()
            flatten_node.name = source_node.name + '_flatten'
            flatten_node.op = 'Flatten'
            flatten_node.input.append(source_node.name)
            self._set_output_shape(source_node, flatten_node)
            source_node.real_name = flatten_node.name
        else:
            kwargs['global_pooling'] = False
            layer_pool_size = self.get_dim_related_attr(source_node.layer, 'pool_size', dim)
            layer_strides   = self.get_dim_related_attr(source_node.layer, 'strides', dim)
            #it's a wrong implementation,but for compatibility i won't fix it now
            kwargs['strides'] = [1] + list(layer_strides) + [1]
            kwargs['kernel_shape'] = [1] + list(layer_pool_size) + [1]
        assign_IRnode_values(IR_node, kwargs)

    def _defuse_activation(self, source_node):
        if source_node.layer.activation is None or source_node.layer.activation.__name__ == "linear":
            return

        IR_node = self.IR_graph.node.add()
        IR_node.name = source_node.real_name + "_activation"
        IR_node.op = self.activation_map[source_node.layer.activation.__name__]
        IR_node.input.append(source_node.real_name)
        #self._set_output_shape(source_node, IR_node)

        # TODO: More activation functions
        # for ELU
        if hasattr(source_node.layer, 'alpha'):
            assign_attr_value(IR_node['alpha'], source_node.layer.alpha)

        source_node.real_name = IR_node.name


    def _set_conv_kwargs(self, source_node, IR_node, dim):
        kwargs = dict()

        layer_kernel_size   = self.get_dim_related_attr(source_node.layer, 'kernel_size', dim)
        layer_strides       = self.get_dim_related_attr(source_node.layer, 'strides',     dim)
        layer_dilation_rate = self.get_dim_related_attr(source_node.layer, 'dilation_rate', dim)
        in_channel = source_node.layer.input_shape[-1] if self.data_format == "channels_last" else source_node.layer.input_shape[1]
        out_channel = source_node.layer.filters or source_node.layer.depth_multiplier

        if source_node.type.startswith("Deconv"):
            kwargs['kernel_shape'] = list(layer_kernel_size) + [out_channel, in_channel]
        else:
            kwargs['kernel_shape'] = list(layer_kernel_size) + [in_channel, out_channel]
        kwargs['rank'] = dim
        # use_bias
        kwargs['use_bias'] = source_node.layer.use_bias

        # strides
        # [1, sd, sh, sw, 1]
        kwargs['strides'] = [1] + list(layer_strides) + [1]

        # dilations
        # [1, dd, dh, dw, 1]
        kwargs['dilations'] = [1] + list(layer_dilation_rate) + [1]

        assign_IRnode_values(IR_node, kwargs)

    def get_op(self, name):
        raise NotImplementedError('should implemented in subclass')
