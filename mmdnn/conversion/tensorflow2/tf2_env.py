from mmdnn.conversion.common.Runtime.env import Env
from mmdnn.conversion.tensorflow2.attr_policy import get_tf_padding_attr, get_conv_kwargs, get_layer_dim
from mmdnn.conversion.tensorflow2.tf2_op_map import PoolRegistry


class Tf2Env(Env):
    def __init__(self):
        super(Tf2Env, self).__init__()
        self.register_op('Dense', has_weight=True, has_activation=True).add_easy_attr('units', 'use_bias')
        self.register_op('Flatten', easy_attr_lst=['data_format'])
        self.register_op('Conv',
                         policy_lst=[('padding', get_tf_padding_attr), get_conv_kwargs],
                         has_weight=True, has_activation=True)
        self.register_op('Pool', registry=PoolRegistry())
        self.register_op('DataInput').add_determining_func(lambda node_type: 'InputLayer' in node_type)
        self.register_op('ZeroPadding',
                         easy_attr_lst=['data_format', 'padding'],
                         policy_lst=[('dim', get_layer_dim)])

