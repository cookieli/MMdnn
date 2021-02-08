from tensorflow.python.keras.layers import Activation

from mmdnn.conversion.common.Runtime.Registry import BatchNormalizationOp, DenseOp, ConvOp, SeparableConvOp, \
    ActivationOp
from mmdnn.conversion.common.Runtime.env import Env
from mmdnn.conversion.tensorflow2.attr_policy import get_tf_padding_attr, get_conv_kwargs, get_layer_dim
from mmdnn.conversion.tensorflow2.tf2_op_map import PoolRegistry


class Tf2Env(Env):
    def __init__(self):
        super(Tf2Env, self).__init__()
        self.register_op('Dense',registry=DenseOpTf2())
        self.register_op('Flatten', easy_attr_lst=['data_format'])
        self.register_op('Conv',
                         policy_lst=[('padding', get_tf_padding_attr), get_conv_kwargs],
                         has_weight=True, has_activation=True)
        self.register_op('Pool', registry=PoolRegistry())
        self.register_op('DataInput').add_determining_func(lambda node_type: 'InputLayer' in node_type)
        self.register_op('ZeroPadding',
                         easy_attr_lst=['data_format', 'padding'],
                         policy_lst=[('dim', get_layer_dim)])
        self.register_op('BatchNormalization',
                         registry=BatchNormalizationOpTf2())
        self.register_op('Activation', registry=ActivationOpTf2())
        self.register_op('Add')

class BatchNormalizationOpTf2(BatchNormalizationOp):
    def __init__(self):
        super(BatchNormalizationOpTf2, self).__init__()
        self.easy_get_attr=['axis', 'momentum', 'epsilon', 'center', 'scale']

class DenseOpTf2(DenseOp):
    def __init__(self):
        super(DenseOpTf2, self).__init__()
        self.easy_get_attr = ['units', 'use_bias']

class ConvOpTf2(ConvOp):
    def __init__(self):
        super(ConvOpTf2, self).__init__()
        self.add_policy(get_tf_padding_attr, 'padding')
        self.add_policy(get_conv_kwargs, 'rank', 'kernel_shape', 'use_bias', 'strides', 'dilations')

class SeparableConvTf2(SeparableConvOp):
    def __init__(self):
        super(SeparableConvTf2, self).__init__()
        self.add_policy(get_tf_padding_attr, 'padding')
        self.add_policy(get_conv_kwargs, 'rank', 'kernel_shape', 'use_bias', 'strides', 'dilations')

class ActivationOpTf2(ActivationOp):
    def __init__(self):
        super(ActivationOpTf2, self).__init__()
        self.add_easy_attr('activation')




