from typing import Callable, Dict

from mmdnn.conversion.common.IR.IR_graph import IRGraph
from mmdnn.conversion.kerasCommon.emitOp import InputOp, ConvOp, PoolingOp, FlattenOp, DenseOp, PadOp, model_out, \
    add_tf_import, ZeroPaddingOp, BatchNormOp


class CodeConstructor(object):
    def __init__(self, model):
        from six import string_types as _string_types
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self.weight_path = model[1]
            # self._load_weights(weight_path)
        # op_map maps from op name to initialize function
        self.op_map: Dict[str, Callable] = {'Conv': ConvOp, 'Pool': PoolingOp,
                                            'Flatten': FlattenOp, 'FullyConnected': DenseOp,
                                            'ZeroPadding': ZeroPaddingOp, 'BatchNorm': BatchNormOp}
        self.IR_graph = IRGraph(network_path)
        self.IR_graph.build()

    def add_padding(self, node, input_val, lvalue, cnt, tab_str):
        padding = node.get_attr('pads')
        if padding is not None:
            pad_op = PadOp(padding, lvalue=lvalue, input_val=input_val)
            input_val = pad_op.lvalue
            cnt = self.code_printer(pad_op, cnt, activation=None, tab_str=tab_str);
        return input_val, cnt

    def code_printer(self, op, cnt, activation, tab_str):
        print(tab_str + op.emit_code(activation))
        return cnt + 1

    def method_constructor(self):
        method_hdr = add_tf_import()
        method_hdr += 'def net(inputs, weight_path):\n'
        print(method_hdr)
        self.topo_traversal("inputs", "weight_path", tab_num=1)

    def topo_traversal(self, inputs, weight_path, tab_num=None):
        op = None
        activation = None
        pad_op = None
        input_val = None
        cnt = 1
        name = 'var_'
        op_map = {}
        tab_str = ""
        if tab_num is not None:
            tab_str = '\t' * tab_num
        for layer in self.IR_graph.topological_sort:
            node = self.IR_graph.get_node(layer)
            if 'activation' in layer:
                activation = node.type
            else:
                if op is not None:
                    cnt = self.code_printer(op, cnt, activation, tab_str)
                activation = None
                lvalue = name + str(cnt)
                if node.type == 'DataInput':
                    op = InputOp(node, lvalue=lvalue)
                else:
                    if len(node.in_edges) != 1:
                        raise NotImplementedError("not handle multi input edge error")
                    else:
                        input_val = op_map[node.in_edges[0]].lvalue
                    input_val, cnt = self.add_padding(node, input_val, lvalue, cnt, tab_str)
                    if node.type in self.op_map:
                        op = self.op_map[node.type](node, lvalue, input_val)
                    else:
                        raise NotImplementedError("Unsupported op {}".format(node.type))
            op_map[layer] = op
        print(tab_str + op.emit_code(activation))
        print(model_out(inputs=inputs, outputs=op.lvalue, weights_path=weight_path, tab_str=tab_str))
