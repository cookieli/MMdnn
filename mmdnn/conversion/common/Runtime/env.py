from typing import Tuple, Dict

from mmdnn.conversion.common.Runtime.Registry import OpRegistry


class Env:
    def __init__(self):
        self.registry_map = {}#key is op name, value is OpRegistry

    def register_op(self, op_name, easy_attr_lst=None, policy_lst=None, has_weight=False, has_activation=False,
                    enable_override=False, registry=None) ->OpRegistry:
        """
        easy_attr_lst is lst of string
        policy_lst is lst of tuple or function: the last of tuple is function and others are string
        registry is instance of OpRegistry
        """
        if policy_lst is None:
            policy_lst = []
        if easy_attr_lst is None:
            easy_attr_lst = []
        if registry is not None:
            self.registry_map[op_name] = registry
        else:
            if enable_override:
                self.registry_map[op_name] = OpRegistry(op_name, has_weight, has_activation)
            else:
                if op_name not in self.registry_map:
                    self.registry_map[op_name] = OpRegistry(op_name, has_weight, has_activation)
        reg = self.registry_map[op_name]
        for attr in easy_attr_lst:
            reg.add_easy_attr(attr)
        for tp in policy_lst:
            if isinstance(tp, tuple):
                reg.add_policy(tp[-1], *tp[:-1])
            else:
                reg.add_policy(tp)
        return reg

    def add_op(self, op_name, op_func):
        pass

    def evaluate(self, source_node) -> Tuple[OpRegistry, Dict]:
        for v in self.registry_map.values():
            if v.is_this_op(source_node.type):
                return v, v.to_ir(source_node)
        raise RuntimeError("this op {} isn't in env".format(source_node.type))
