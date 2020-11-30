from mmdnn.conversion.common.Runtime.Registry import OpRegistry
from mmdnn.conversion.tensorflow2.attr_policy import get_tf_padding_attr, get_conv_kwargs, get_pool_attr

op_attr_map = {
    'Dense':   ['units', 'use_bias', 'weight', 'activation'],
    'Flatten': ['data_format'],
    'Conv':    ['weight', 'activation', ('padding', get_tf_padding_attr), get_conv_kwargs]

}

op_ir_name = {
    'Dense': 'FullyConnected'
}

class PoolRegistry(OpRegistry):
    def __init__(self):
        super(PoolRegistry, self).__init__('Pool')
        self.add_policy(get_pool_attr,"dim", 'pooling_type', 'global_pooling', 'strides', 'kernel_shape')


