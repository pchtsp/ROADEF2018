# These are auxiliary functions for nodes of trees (see ete3).


def move_node(node, movement, axis):
    node_pos = getattr(node, axis)
    setattr(node, axis, node_pos + movement)
    for children in node.get_children():
        move_node(children, movement, axis)
    return True


def change_feature(node, feature, value):
    setattr(node, feature, value)
    for children in node.get_children():
        change_feature(children, feature, value)
    return True


def find_ancestor_level(node, level):
    for anc in node.iter_ancestors():
        if anc.CUT == level:
            return anc
    return None


def find_waste_sibling(node):
    for n in node.get_sisters():
        if n.TYPE in [-1, -3]:
            return n
    return None


def order_children(node):
    for v in node.traverse():
        if not len(v.children):
            continue
        # we inverse this condition because we're dealing
        # with the children
        axis, dim = get_orientation_from_cut(v, inv=True)
        v.children.sort(key=lambda x: getattr(x, axis))
    return True


def get_orientation_from_cut(node, inv=False):
    # inv: means inverse the result.
    result = node.CUT % 2
    if inv:
        result = not result
    if result:  # cuts 1 and 3
        dim = 'WIDTH'
        axis = 'X'
    else:  # cut 2 and 4
        dim = 'HEIGHT'
        axis = 'Y'
    return axis, dim


def get_node_leaves(node, min_type=0):
    return [leaf for leaf in node.get_leaves() if leaf.TYPE >= min_type]


def get_node_pos(node):
    return node.PLATE_ID, node.up.children.index(node)