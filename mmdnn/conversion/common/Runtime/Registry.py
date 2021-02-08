from typing import Dict, Callable, List, Set, Union, Optional, Tuple
from collections import OrderedDict
from mmdnn.conversion.common.DataStructure.graph import GraphNode
from mmdnn.conversion.common.utils import flatten_tuple

class OpRegistry:
    def __init__(self, op_name:str, init_func:Callable=None, has_weight:bool=False, has_activation:bool=False):
        self.op_name                                             = op_name
        self.init_func                                           = init_func
        self.attr_policy:Dict[str, int]                          = dict() # it's map about string attr name to policy's index
        self.policies: List[Callable[[GraphNode, Dict], None]]   = list() # a list of function
        self.attributes: List[str]                               = list()
        self.has_anonymous_policy                                = False
        self.easy_get_attr:Set[str]                              = set() # a set of string name
        self.is_this_op:Callable[[str], bool]                    = lambda node_type: self.op_name in node_type
        self.has_weight                                          = has_weight
        self.has_activation                                      = has_activation
        self.unify_name                                          = self.op_name
        self.is_separable_conv:Callable[[str], bool]             = lambda node_type: 'Separable' in node_type and 'Conv' in node_type
        self.is_data_input:Callable[[str], bool]                 = lambda node_type: "InputLayer" in node_type
        self.need_weight:List[str]                               = list()
        self.trainable: OrderedDict[str, Optional[str]]       = OrderedDict()

    def add_policy(self, policy_func, *argv):
        if policy_func is None:
            for arg in argv:
                self.easy_get_attr.add(arg)
            return self
        self.policies.append(policy_func)
        if len(argv) == 0 and not self.has_anonymous_policy:
            self.has_anonymous_policy = True
        for arg in argv:
            self.easy_get_attr.discard(arg)
            self.attr_policy[arg] = len(self.policies) - 1
        return self

    def add_policies(self, policies:List[Union[Tuple, Callable]]):
        for p in policies:
            if isinstance(p, tuple):
                self.add_policy(p[-1], *p[:-1])
            else:
                self.add_policy(p)

    def add_easy_attr(self, *argv):
        return self.add_policy(None, *argv)

    def add_determining_func(self, checker):
        self.is_this_op = checker
        return self

    def to_ir(self, source_node):
        kwargs = {}
        for attr in self.easy_get_attr:
            tmp_attr     = getattr(source_node.layer, attr)
            if isinstance(tmp_attr, tuple):
                tmp_attr = flatten_tuple(tmp_attr)
            elif callable(tmp_attr):
                tmp_attr = tmp_attr.__name__
            kwargs[attr] = tmp_attr
        for policy in self.policies:
            policy(source_node, kwargs)
        return kwargs

    def add_weight(self):
        self.has_weight = True

    def add_activation(self):
        self.has_activation = True

    def set_sepa_conv_checker(self, checker):
        self.is_separable_conv = checker

    def set_data_input_checker(self, checker):
        self.is_data_input = checker

class BatchNormalizationOp(OpRegistry):
    def __init__(self):
        super(BatchNormalizationOp, self).__init__('BatchNorm')
        self.trainable = {"scale": "scale", "bias": "center", "mean": None, "var": None}

class DenseOp(OpRegistry):
    def __init__(self):
        super(DenseOp, self).__init__('Dense')
        self.trainable = {"weights":None, "bias":"use_bias"}

class ConvOp(OpRegistry):
    def __init__(self):
        super(ConvOp, self).__init__('Conv')
        self.trainable                        = {"weights":None, "bias":"use_bias"}
        self.is_this_op:Callable[[str], bool] = lambda node_type: 'Conv' in node_type and 'Separable' not in node_type

class SeparableConvOp(ConvOp):
    def __init__(self):
        super(SeparableConvOp, self).__init__()
        self.op_name = self.unify_name = 'SeparableConv'
        self.trainable = {"depthwise_filter":None, "pointwise_filter":None}

class ActivationOp(OpRegistry):
    def __init__(self):
        super(ActivationOp, self).__init__('Activation')





