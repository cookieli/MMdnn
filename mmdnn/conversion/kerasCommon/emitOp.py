from typing import Dict

from mmdnn.conversion.common.IR import graph_pb2
from mmdnn.conversion.common.IR.IR_graph import IRGraph

def model_out(inputs, outputs, tab_str, weights_path = None):
    name = 'tf.keras.Model'
    arg = "(" + "inputs={}, outputs={}, name='output_model'".format(inputs, outputs) +")"
    ret =  tab_str + 'model = ' + name + arg
    if weights_path:
        ret += '\n'
        ret += tab_str + 'model' + '.' + 'load_weights' + '(' +weights_path  + ')'
    ret += '\n'
    ret += tab_str + 'return model'
    return ret

def add_tf_import():
    return 'import tensorflow as tf ' + '\n'

def construct_arg(args:Dict[str, str])->str:
    ret = ""
    for k, v in args.items():
        ret += "{}={}, ".format(k, v)
    return ret[:-2]

class BaseOp(object):
    header = 'tf.keras.layers'
    args   = ''
    layer_call_map = {
        'input'       : header + '.' + 'Input',
        'conv2D'      : header + '.' + 'Conv2D',
        'maxpool2D'   : header + '.' + 'MaxPool2D',
        'maxpool1D'   : header + '.' + 'MaxPool1D',
        'maxpool3D'   : header + '.' + 'MaxPool3D',
        'avgpool1D'   : header + '.' + 'AveragePooling1D',
        'avgpool2D'   : header + '.' + 'AveragePooling2D',
        'avgpool3D'   : header + '.' + 'AveragePooling3D',
        'gblmaxpool1D': header + '.' + 'GlobalMaxPooling1D',
        'gblmaxpool2D': header + '.' + 'GlobalMaxPooling2D',
        'gblmaxpool3D': header + '.' + 'GlobalMaxPooling3D',
        'gblavgpool1D': header + '.' + 'GlobalAveragePooling1D',
        'gblavgpool2D': header + '.' + 'GlobalAveragePooling2D',
        'gblavgpool3D': header + '.' + 'GlobalAveragePooling3D',
        'flatten'     : header + '.' + 'Flatten',
        'dense'       : header + '.' + 'Dense',
        'zeropadding' : header + '.' + 'ZeroPadding',
        'batchnorm'   : header + '.' + 'BatchNormalization',
        'activation'  : header + '.' + 'Activation'
    }

    dtype_map = {
        graph_pb2.DT_FLOAT16: "float16",
        graph_pb2.DT_FLOAT32: "float32",
        graph_pb2.DT_FLOAT64: "float64",
        graph_pb2.DT_INT16: "int16",
        graph_pb2.DT_INT32: "int32",
        graph_pb2.DT_INT64: "int64",
        graph_pb2.DT_UINT8: "uint8",
        graph_pb2.DT_UINT16: "uint16"
    }

    def __init__(self, lvalue, input_val):
        self.input_val  = input_val
        self.lvalue = lvalue

    @property
    def call_header(self):
        raise NotImplementedError("you should implement it in subclass")

    def need_reserve(self, key, **kwargs):
        ret = self.kwargs[key] is not None
        if self.args:
            return ret and key in self.args
        return ret

    def construct_arg(self):
        pass

    def emit_code(self, activation = None):
        func = None
        if activation is None:
            func = self.call_header + '(' + self.construct_arg()  + ')'
        else:
            func = self.call_header \
               + '(' + self.construct_arg() + ", activation = '{}'".format(activation.lower()) + ')'
        return self.lvalue + '=' +  func + '(' + self.input_val + ')'

class InputOp(BaseOp):
    def __init__(self, IR_node, lvalue, input_val=None):
        self.IR_node = IR_node
        super(InputOp, self).__init__(lvalue, input_val)

    def construct_arg(self):
        shape_str = IRGraph.shapeToStr(self.IR_node.layer.attr['shape'].shape)
        dtype_str =  self.dtype_map[self.IR_node.layer.attr['dtype'].type] \
                            if 'dtype' in self.IR_node.layer.attr else ""
        arg = "name = '{}', shape= ({}), dtype = '{}'".format(
            self.IR_node.variable_name,
            shape_str,
            dtype_str)
        return arg

    @property
    def call_header(self):
        return self.layer_call_map['input']

    def emit_code(self, activation = None):
        func = None
        if activation is None:
            func = self.call_header + '(' + self.construct_arg()  + ')'
        else:
            func = self.call_header \
               + '(' + self.construct_arg() + ", activation = '{}'".format(activation.lower()) + ')'
        return self.lvalue + '=' + func

class ConvOp(BaseOp):
    args = """filters, kernel_size, strides=(1, 1), padding='valid', data_format=None,
               dilation_rate=(1, 1), groups=1, activation=None, use_bias=True,
               kernel_initializer='glorot_uniform', bias_initializer='zeros',
               kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
               kernel_constraint=None, bias_constraint=None, **kwargs"""
    CONV1D    = 1
    CONV2D    = 2
    CONV3D    = 3

    # def __init__(self, dim, **kwargs):
    #     self.dim  = dim
    #     super(ConvOp, self).__init__(kwargs)
    #     self.transpose = False
    #     if 'output_padding' in self.attr:
    #         self.transpose = True

    def __init__(self, IR_node, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.IR_node = IR_node
        self.conv_type = IR_node.type

    def construct_arg(self):
        group = self.IR_node.get_attr("group", 1)

        if self.conv_type.endswith('Transpose'):
            filters = self.IR_node.get_attr('kernel_shape')[-2]
        else:
            filters = self.IR_node.get_attr('kernel_shape')[-1]

        filters_str = 'filters={}'.format(filters) if not self.conv_type.endswith(
            'DepthwiseConv2D') else 'depth_multiplier={}'.format(filters)
        # change dw from filters to 1
        padding = 'VALID'

        dilations = self.IR_node.get_attr('dilations')

        if not dilations or len(dilations) == 2:
            # reset the default dilation
            dilations = [1] * len(self.IR_node.get_attr('kernel_shape'))

        args = "name='{}', group={}, conv_type='{}', {}, kernel_size={}, strides={}, dilation_rate={}, padding='{}', use_bias={}".format(
            self.IR_node.name,
            group,
            self.conv_type,
            filters_str,
            tuple(self.IR_node.get_attr('kernel_shape')[:-2]),
            tuple(self.IR_node.get_attr('strides')[1:-1]),
            tuple(dilations[1:-1]),
            padding,
            self.IR_node.get_attr('use_bias'))
        return args

    @property
    def call_header(self):
        name = 'conv' + str(self.IR_node.get_attr('rank')) + 'D'
        #name += 'Transpose'
        return self.layer_call_map[name]

class PadOp(BaseOp):
    def __init__(self, padding, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.paddings = [padding[i:i+2] for i in range(0, len(padding), 2)]

    @property
    def call_header(self):
        return 'tf.pad'

    def construct_arg(self):
        return "tensor={}, paddings = {}, mode = 'CONSTANT'".format(self.input_val, str(self.paddings))

    def emit_code(self, activation = None):
        func = None
        if activation is None:
            func = self.call_header + '(' + self.construct_arg()  + ')'
        else:
            func = self.call_header \
               + '(' + self.construct_arg() + ", activation = '{}'".format(activation.lower()) + ')'
        return self.lvalue + '=' + func

class PoolingOp(BaseOp):
    def __init__(self, IR_node, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        dim = self.IR_node.get_attr('dim')
        name = 'pool' + str(dim) + 'D'
        kind = self.IR_node.get_attr('pooling_type')
        if kind == 'MAX':
            name = 'max' + name
        elif  kind == 'AVG':
            name = 'avg' + name
        else:
            raise NotImplementedError('not implemented')
        is_global = self.IR_node.get_attr('global_pooling')
        if is_global:
            name = 'gbl' + name
        return self.layer_call_map[name]

    def construct_arg(self):
        pool_size = tuple(self.IR_node.get_attr('kernel_shape')[1:-1])
        strides   = tuple(self.IR_node.get_attr('strides')[1:-1])
        arg = "pool_size = {}, strides = {}, name = '{}'".format(pool_size, strides, self.IR_node.name)
        return arg

class FlattenOp(BaseOp):
    def __init__(self, IR_node, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        name = 'flatten'
        return self.layer_call_map[name]

    def construct_arg(self):
        arg = "name='flatten', data_format = '{}'".format(self.IR_node.get_attr('data_format'))
        return arg

class DenseOp(BaseOp):
    args = """units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
              activity_regularizer=None, kernel_constraint=None, bias_constraint=None,"""
    def __init__(self, IR_node, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        name = 'dense'
        return self.layer_call_map[name]

    def construct_arg(self):
        arg = "units = {}, use_bias = {}, name = '{}'".format(self.IR_node.get_attr('units'),
                                                                self.IR_node.get_attr('use_bias'),
                                                                self.IR_node.name)
        return arg

class ZeroPaddingOp(BaseOp):

    def __init__(self, IR_node, lvalue, input_val):
        super().__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        name =  'zeropadding'
        dim  =  self.IR_node.get_attr('dim')
        return self.layer_call_map[name] + str(dim) + 'D'

    def construct_arg(self):
        args = {'padding':tuple(self.IR_node.get_attr('padding')),
                'data_format': self.IR_node.get_attr('data_format')}
        ret = ""
        for k, v in args.items():
            ret +="{}={}, ".format(k, v)
        return ret[:-2]

class BatchNormOp(BaseOp):

    def __init__(self, IR_node, lvalue, input_val):
        super(BatchNormOp, self).__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        name = 'batchnorm'
        return self.layer_call_map[name]

    def construct_arg(self):
        args = {'axis'    : self.IR_node.get_attr('axis'),
                'momentum': self.IR_node.get_attr('momentum'),
                'epsilon' : self.IR_node.get_attr('epsilon'),
                'center'  : self.IR_node.get_attr('center'),
                'scale'   : self.IR_node.get_attr('scale')}
        return construct_arg(args)

class ActivationOp(BaseOp):

    def __init__(self, IR_node, lvalue, input_val):
        super(BatchNormOp, self).__init__(lvalue, input_val)
        self.IR_node = IR_node

    @property
    def call_header(self):
        name = 'activation'
        return self.layer_call_map[name]

    def construct_arg(self):
        args = {'activation' : self.IR_node.get_attr('activation')}
        return construct_arg(args)


