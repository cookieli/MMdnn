from six import text_type, binary_type, integer_types, string_types

#from mmdnn.conversion.common.utils import _parse_conv_node


def is_default_type(val):
    return isinstance(val, bool) or isinstance(val, integer_types) or isinstance(val, float) \
           or isinstance(val, binary_type) or isinstance(val, text_type) or isinstance(val, list) \
           or isinstance(val, tuple)

def get_conv_op(source_node, attr=None):
    #return _parse_conv_node(source_node).name
    pass

def get_tf_padding_attr(source_node, kwargs):
    dims = len(source_node.layer.input_shape)
    if source_node.layer.padding == 'valid':
        kwargs['auto_pad'] = "VALID"
        kwargs['pads'] = [0, 0] * dims

    elif source_node.layer.padding == 'same':
        kernel_shape = source_node.layer.kernel_size if hasattr(source_node.layer,'kernel_size') else source_node.layer.pool_size
        is_channels_first = source_node.layer.data_format == 'channels_first'
        is_channels_last = source_node.layer.data_format == 'channels_last'
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
            pad_left = pad // 2
            pad_right = pad - pad_left
            padding.extend([pad_left, pad_right])
        if is_channels_first:
            padding = [0, 0, 0, 0] + padding
        else:
            padding = [0, 0] + padding + [0, 0]
        kwargs['auto_pad'] = "SAME_LOWER"
        kwargs['pads']     =  padding
    else:
        assert False

def execute_to_kwargs_policy(source_node, attr, kwargs):
    if isinstance(attr, string_types):
        kwargs[attr] = getattr(source_node.layer, attr)
    elif isinstance(attr, tuple):
        attr[-1](source_node, kwargs)
    else:
        attr(source_node, kwargs)

def get_conv_kwargs(source_node, kwargs):
    dim = getattr(source_node.layer, 'rank')
    layer_kernel_size   = get_dim_related_attr(source_node.layer, 'kernel_size', dim)
    layer_strides       = get_dim_related_attr(source_node.layer, 'strides',     dim)
    layer_dilation_rate = get_dim_related_attr(source_node.layer, 'dilation_rate', dim)
    in_channel = source_node.layer.input_shape[-1] if source_node.layer.data_format == "channels_last" \
                                                      else source_node.layer.input_shape[-1 - dim]
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

def get_dim_related_attr(layer, name, dim):
    if hasattr(layer, name):
        attr = getattr(layer, name)
        if isinstance(attr, int):
            return (attr,) * dim
        return attr
    else:
        raise AttributeError("layer do not have this attr {}".name)

#for pool
def get_pool_type(source_node, kwargs):
    node_type = source_node.type
    if 'Max' in node_type:
        kwargs['pooling_type'] = 'MAX'
    elif 'Average' in node_type:
        kwargs['pooling_type'] = 'AVG'
    else:
        raise NotImplementedError("Not cover all pool type {}".format(node_type))

def get_global_pool_attr(source_node, kwargs, dim):
    kwargs['global_pooling'] = True
    kwargs['strides']        = [1] * (dim + 2)
    kwargs['kernel_shape']   = [1] + [-1] * dim + [1]# why here is -1: because global pool which other dim is
                                                     # same as hw, and we don't need to set it until gen code
                                                     #  but for compatibility, we must have this key word

def get_non_global_pool_attr(source_node, kwargs, dim):
    kwargs['global_pooling'] = False
    layer_pool_size          = get_dim_related_attr(source_node.layer, 'pool_size', dim)
    layer_strides            = get_dim_related_attr(source_node.layer, 'strides', dim)
    kwargs['strides']        = [1] + list(layer_strides) + [1]
    kwargs['kernel_shape']   = [1] + list(layer_pool_size) + [1]

def get_pool_attr(source_node, kwargs):
    get_pool_type(source_node, kwargs)
    node_type = source_node.type
    is_global_pooling = 'Global' in node_type
    dim = int(node_type[-2])
    kwargs['dim'] = dim
    if is_global_pooling:
        get_global_pool_attr(source_node, kwargs, dim)
    else:
        get_non_global_pool_attr(source_node, kwargs, dim)

def get_layer_dim(source_node, kwargs):
    kwargs['dim'] = int(source_node.type[-2])
