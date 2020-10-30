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
        'dense'       : header + '.' + 'Dense'

    }
    def __init__(self, **kwargs):
        self.attr = {k:v for k, v in kwargs if self.need_reserve(k,kwargs)}

    @property
    def call_header(self):
        pass

    def need_reserve(self, key, **kwargs):
        ret = self.kwargs[key] is not None
        if self.args:
            return ret and key in self.args
        return ret

    def emit_code(self):
        ret = '('
        attr_map =  {k:v for k, v in self.attr if self.need_reserve(k)}
        for key, value in attr_map:
            ret += (key + ' = ' + str(vars(self)[key])) + ' , '
        ret = ret[:-1] + ')'
        return self.call_header + ret

class InputOp(BaseOp):
    def __init__(self, shape=None, batch_size=None,
                 name=None, dtype=None, sparse=False,
                 tensor=None,ragged=False):
        self.shape      = shape
        self.batch_size = batch_size
        self.name       = name
        self.dtype      = dtype
        self.sparse     = sparse
        self.tensor     = tensor
        self.ragged     = ragged
        super(InputOp, self).__init__()

    @property
    def call_header(self):
        return self.layer_call_mapp['input']

class ConvOp(BaseOp):
    args = """filters, kernel_size, strides=1, padding='valid', output_padding=None,
                   data_format=None, dilation_rate=1, activation=None, use_bias=True,
                   kernel_initializer='glorot_uniform', bias_initializer='zeros',
                   kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None,
                   kernel_constraint=None, bias_constraint=None,"""
    CONV1D    = 1
    CONV2D    = 2
    CONV3D    = 3

    def __init__(self, dim, **kwargs):
        self.dim  = dim
        super(ConvOp, self).__init__(kwargs)
        self.transpose = False
        if 'output_padding' in self.attr:
            self.transpose = True

    def call_header(self):
        name = 'conv' + str(self.dim) + 'D'
        name += 'Transpose'
        return self.layer_call_map[name]

class PoolingOp(BaseOp):
    MAX = 1
    AVG = 2

    def __init__(self, dim, kind, is_global, **kwargs):
        self.dim = dim
        self.kind = kind
        self.is_global = is_global
        super(PoolingOp, self).__init__(kwargs)

    def call_header(self):
        name = 'pool' + self.dim + 'D'
        if self.kind == self.MAX:
            name = 'max' + name
        elif  self.kind == self.AVG:
            name = 'avg' + name
        else:
            raise NotImplementedError('not implemented')
        if self.is_global:
            name = 'gbl' + name
        return self.layer_call_map[name]

class FlattenOp(BaseOp):
    def __init__(self, **kwargs):
        super(FlattenOp, self).__init__(kwargs)

    def call_header(self):
        name = 'flatten'
        return self.layer_call_map[name]

class DenseOp(BaseOp):
    args = """units, activation=None, use_bias=True, kernel_initializer='glorot_uniform',
              bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None,
              activity_regularizer=None, kernel_constraint=None, bias_constraint=None,"""
    def __init__(self, **kwargs):
        super(DenseOp, self).__init__(kwargs)

    def call_header(self):
        name = 'dense'
        return self.layer_call_map[name]



class ModelCostructor







