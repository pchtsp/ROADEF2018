import ete3

# These are auxiliary functions for nodes of trees (see ete3).
# TODO: this should be a subclass of TreeNode...


def create_node(**kwargs):
    node = ete3.Tree(name=kwargs['NODE_ID'])
    node.add_features(**kwargs)
    return node


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


def resize_node(node, dim, quantity):
    waste = find_waste(node, child=True)
    if waste is None:
        return False
    setattr(waste, dim, getattr(waste, dim) + quantity)
    return True


def rotate_node(node, subnodes=True):
    node.WIDTH, node.HEIGHT = node.HEIGHT, node.WIDTH
    root = node.get_tree_root()
    pos_rel_i = {'Y': node.X - root.X + root.Y,
                 'X': node.Y - root.Y + root.X}
    for axis in ['X', 'Y']:
        setattr(node, axis, pos_rel_i[axis])
    for children in node.get_children():
        rotate_node(children, subnodes)
    return node


def find_ancestor_level(node, level, incl_node=True):
    if incl_node:
        if node.CUT == level:
            return node
    for anc in node.iter_ancestors():
        if anc.CUT == level:
            return anc
    return None


def get_descendant(node, which="first"):
    assert which in ['first', 'last']
    if which == 'first':
        pos = 0
    else:
        pos = -1
    children = node.children
    if not children:
        return node
    else:
        return get_descendant(children[pos], which=which)


def find_waste(node, child=False):
    # We assume waste is at the end. Always.
    # If child=True we look in children instead of siblings
    if not child:
        node = node.up
    children = node.children
    if not children:
        return None
    waste = children[-1]
    if waste.TYPE not in [-1, -3]:
        return None
    return waste


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


def get_node_leaves(node, min_type=0, max_type=99999, type_options=None):
    if type_options is None:
        return [leaf for leaf in node.get_leaves() if min_type <= leaf.TYPE <= max_type]
    if type(type_options) is not list:
        raise ValueError("type_options needs to be a list instead of {}".
                         format(type(type_options)))
    return [leaf for leaf in node.get_leaves() if leaf.TYPE in type_options]


def get_node_pos(node):
    return node.PLATE_ID, node.up.children.index(node)


def node_to_square(node):
    """
    Reformats a piece to a list of two points
    :param node: a leaf from ete3 tree.
    :return: list of two points {'X': 1, 'Y': 1}
    """
    axis = ['X', 'Y']
    axis_dim = {'X': 'WIDTH', 'Y': 'HEIGHT'}
    return [{a: getattr(node, a) for a in axis},
     {a: getattr(node, a) + getattr(node, axis_dim[a]) for a in axis}]


def get_features(node):
    features = ['X', 'Y', 'NODE_ID', 'PLATE_ID', 'CUT', 'TYPE', 'WIDTH', 'HEIGHT']
    attrs = {k: int(getattr(node, k)) for k in features}
    parent = node.up
    if parent is not None:
        parent = parent.NODE_ID
    attrs['PARENT'] = parent
    return attrs


def duplicate_node_as_child(node):
    features = get_features(node)
    features['NODE_ID'] += 200
    features['CUT'] += 1
    child = create_node(**features)
    node.add_child(child)
    node.TYPE = -2
    return child


def delete_only_child(node):
    if len(node.children) != 1:
        return False
    child = node.children[0]
    features = get_features(child)
    features['CUT'] -= 1
    child.detach()
    node.add_features(**features)
    return True


def add_child_waste(node, fill):
    axis, dim_i = get_orientation_from_cut(node, inv=True)
    node_size = getattr(node, dim_i)
    child_size = fill - node_size
    if child_size <= 0:
        # sometimes we want a node without children
        # (because it had waste as only other child and now it hasn't).
        if len(node.children) == 1:
            delete_only_child(node)
        return False
    if node.is_leaf():
        # sometimes we have a node without children
        # (because it had no waste and now it has).
        duplicate_node_as_child(node)
    features = get_features(node)
    features[axis] += node_size
    features[dim_i] = child_size
    features['TYPE'] = -1
    features['CUT'] += 1
    # TODO: get a better NODE_ID
    features['NODE_ID'] += 100
    child = create_node(**features)
    node.add_child(child)
    setattr(node, dim_i, fill)
    return True


def get_size_without_waste(node, dim):
    waste = find_waste(node, child=True)
    if waste is None:
        return getattr(node, dim)
    return getattr(node, dim) - getattr(waste, dim)


def del_child_waste(node):
    axis, dim_i = get_orientation_from_cut(node, inv=True)
    child = find_waste(node, child=True)
    if child is None:
        return False
    child.detach()
    new_size = getattr(node, dim_i) - getattr(child, dim_i)
    setattr(node, dim_i, new_size)
    return True


def get_node_position_cost_unit(node, plate_width):
    return node.PLATE_ID * plate_width + node.X + node.Y/10


def get_node_position_cost(node, plate_width):
    # we'll give more weight to things that are in the right and up.
    # I guess it depends on the size too...
    return get_node_position_cost_unit(node, plate_width) * (node.WIDTH * node.HEIGHT)


def filter_defects(node, defects, previous=True):
    # filter defects if to the left of node.
    # return defects to the right. Unless previous=False
    if previous:
        return [d for d in defects if d['X'] >= node.X and d['Y'] >= node.Y]
    return [d for d in defects if d['X'] <= node.X + node.WIDTH and d['Y'] <= node.Y + node.HEIGHT]


def split_waste(node1, cut):
    # first, we split one waste in two.
    # then we make a swap to one of the nodes.
    # if child=True: the resulting wastes are children of the first waste
    assert node1.TYPE == -1, "node {} needs to be waste!".format(node1.name)
    parent = node1.up
    axis, dim = get_orientation_from_cut(node1)
    axis_i, dim_i = get_orientation_from_cut(node1, inv=True)
    attributes = [axis, axis_i, dim, dim_i]
    size = getattr(node1, dim)
    assert size > cut, "cut for node {} needs to be smaller than size".format(node1.name)
    node2 = node1.copy()
    nodes = {1: node1, 2: node2}
    pos = {k: {a: getattr(nodes[k], a) for a in attributes} for k in nodes}
    pos[2][axis] = pos[1][axis] + cut
    pos[2][dim] = pos[1][dim] - cut
    pos[1][dim] = cut

    for k, node in nodes.items():
        setattr(node, axis, pos[k][axis])
        setattr(node, dim, pos[k][dim])
    node2.NODE_ID += 300
    parent.add_child(node2)
    order_children(parent)
    return nodes