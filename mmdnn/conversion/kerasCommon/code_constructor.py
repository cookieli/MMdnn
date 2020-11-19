from mmdnn.conversion.common.IR.IR_graph import IRGraph
from mmdnn.conversion.kerasCommon.emitOp import InputOp, ConvOp, PoolingOp, FlattenOp, DenseOp, PadOp, model_out, \
    add_tf_import


class CodeConstructor(object):
    def __init__(self, model):
        from six import string_types as _string_types
        if isinstance(model, _string_types):
            network_path = model
        else:
            network_path = model[0]
            self.weight_path = model[1]
            # self._load_weights(weight_path)

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
                        raise NotImplementedError("not handle multiinput edge error")
                    else:
                        input_val = op_map[node.in_edges[0]].lvalue
                    input_val, cnt = self.add_padding(node, input_val, lvalue, cnt, tab_str)
                    if node.type == 'Conv':
                        op = ConvOp(node, node.type, lvalue=lvalue, input_val=input_val)
                    elif node.type == 'Pool':
                        op = PoolingOp(node, lvalue=lvalue, input_val=input_val)
                    elif node.type == 'Flatten':
                        op = FlattenOp(node, lvalue=lvalue, input_val=input_val)
                    elif node.type == 'FullyConnected':
                        op = DenseOp(node, lvalue=lvalue, input_val=input_val)
                    else:
                        raise NotImplementedError("Unsupported op")
            op_map[layer] = op
        print(tab_str + op.emit_code(activation))
        print(model_out(inputs=inputs, outputs=op.lvalue, weights_path=weight_path, tab_str=tab_str))
