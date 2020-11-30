class OpRegistry:
    def __init__(self, op_name, has_weight=False, has_activation=False):
        self.op_name               = op_name
        self.attr_policy           = dict() # it's map about string attr name to policy's index
        self.policies              = list() # a list of
        self.has_anonymous_policy  = False
        self.easy_get_attr         = set() # a set of string name
        self.is_this_op            = lambda node_type: self.op_name in node_type
        self.has_weight            = has_weight
        self.has_activation        = has_activation
        self.unify_name            = self.op_name

    def add_policy(self, policy_func, *argv):
        if policy_func is None:
            for arg in argv:
                self.easy_get_attr.add(arg)
            return self
        self.policy.append(policy_func)
        if len(argv) == 0 and not self.has_anonymous_policy:
            self.has_anonymous_policy = True
        for arg in argv:
            self.easy_get_attr.discard(arg)
            self.attr_policy[arg] = len(self.policy) - 1
        return self

    def add_easy_attr(self, *argv):
        return self.add_policy(None, argv)

    def add_determining_func(self, checker):
        self.is_this_op = checker
        return self

    def to_ir(self, source_node):
        kwargs = {}
        for attr in self.easy_get_attr:
            tmp_attr     = getattr(source_node.layer, attr)
            if isinstance(tmp_attr, tuple):
                tmp_attr = list(tmp_attr)
            kwargs[attr] = tmp_attr
        for policy in self.policies:
            policy(source_node, kwargs)
        return kwargs

    def add_weight(self):
        self.has_weight = True

    def add_activation(self):
        self.has_activation = True






