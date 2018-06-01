import ete3
import package.geometry as geom

# These are auxiliary functions for nodes of trees (see ete3).
# TODO: this should be a subclass of TreeNode...


def item_to_node(item):
    args = {'WIDTH': item['WIDTH_ITEM'],
            'HEIGHT': item['LENGTH_ITEM'],
            'CUT': 0,
            'X': 0,
            'Y': 0,
            'TYPE': item['ITEM_ID'],
            'NODE_ID': item['ITEM_ID'],
            'PLATE_ID': 0
            }
    return create_node(**args)


def create_plate(width, height, id):
    args = {'WIDTH': width,
            'HEIGHT': height,
            'CUT': 0,
            'X': 0,
            'Y': 0,
            'TYPE': -3,
            'NODE_ID': 0,
            'PLATE_ID': id
            }
    return create_node(**args)


def create_node(**kwargs):
    node = ete3.Tree(name=kwargs['NODE_ID'])
    node.add_features(**kwargs)
    return node


def mod_feature_node(node, quantity, feature):
    node_pos = getattr(node, feature)
    setattr(node, feature, node_pos + quantity)
    for children in node.get_children():
        mod_feature_node(children, quantity, feature)
    return True


def change_feature(node, feature, value):
    setattr(node, feature, value)
    for children in node.get_children():
        change_feature(children, feature, value)
    return True


def resize_node(node, dim, quantity):
    waste = find_waste(node, child=True)
    # if we need to create a waste: no problemo.
    if waste is None:
        if quantity <= 0:
            return False
        # We need to add a waste at the end...
        if not node.children:
            return False
        last = node.children[-1]
        add_sibling_waste(last, quantity, dim)
        return True
    setattr(waste, dim, getattr(waste, dim) + quantity)
    # if we eliminate the waste to 0, we delete the node.
    if not getattr(waste, dim):
        waste.detach()
    delete_only_child(node)
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
    if incl_node and node.CUT == level:
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
    children = node.get_children()
    if not children:
        return node
    else:
        return get_descendant(children[pos], which=which)


def find_waste(node, child=False):
    # We assume waste is at the end. Always.
    # If child=True we look in children instead of siblings

    # special case! if the node IS a waste, I'll create a waste child
    # and return it...
    parent = node
    if not child:
        parent = node.up
    if parent is None:
        # this means we were dealing with the first node
        children = [node]
    else:
        children = parent.get_children()
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
        # I sort mainly according to axis but I want the empty cuts to
        # come before... (that's why the dim in second place)
        v.children.sort(key=lambda x: (getattr(x, axis), getattr(x, dim)))
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


def get_axis_of_dim(dim):
    r = {
        'HEIGHT': 'Y',
        'WIDTH': 'X'
    }
    return r[dim]


def get_node_leaves(node, min_type=0, max_type=99999, type_options=None):
    if type_options is None:
        return [leaf for leaf in node.get_leaves() if min_type <= leaf.TYPE <= max_type]
    if type(type_options) is not list:
        raise ValueError("type_options needs to be a list instead of {}".
                         format(type(type_options)))
    return [leaf for leaf in node.get_leaves() if leaf.TYPE in type_options]


def get_node_pos(node):
    pos = 0
    if node.up is not None:
        pos = node.up.get_children().index(node)
    return node.PLATE_ID, pos


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


def node_to_plate(node):
    return (node.WIDTH, node.HEIGHT)


def default_features():
    return ['X', 'Y', 'NODE_ID', 'PLATE_ID', 'CUT', 'TYPE', 'WIDTH', 'HEIGHT']


def get_features(node, features=None):
    if features is None:
        features = default_features()
    attrs = {k: int(getattr(node, k)) for k in features}
    parent = node.up
    if parent is not None:
        parent = int(parent.NODE_ID)
    attrs['PARENT'] = parent
    return attrs


def duplicate_node_as_its_parent(node, node_mod=900, return_both=False):
    features = get_features(node)
    features['NODE_ID'] += node_mod
    parent = create_node(**features)
    mod_feature_node(node=node, quantity=1, feature="CUT")
    grandparent = node.up
    node.detach()
    parent.add_child(node)
    parent.TYPE = -2
    if grandparent is not None:
        grandparent.add_child(parent)
        order_children(grandparent)
    if return_both:
        return node, parent
    return parent


def duplicate_node_as_child(node, node_mod=500):
    features = get_features(node)
    features['NODE_ID'] += node_mod
    child = create_node(**features)
    # we increase the cut recursively among all children
    mod_feature_node(child, 1, "CUT")
    node.add_child(child)
    # print('created in node ID={}, TYPE={} a child with ID={}, TYPE={}'.
    #       format(node.NODE_ID, node.TYPE, child.NODE_ID, child.TYPE))
    node.TYPE = -2
    return child


def duplicate_waste_into_children(node):
    assert node.TYPE in [-1, -3], \
        'node {} needs to be a waste!'.format(node.name)
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    child1 = duplicate_node_as_child(node, node_mod=200)
    setattr(child1, dim_i, 0)
    child2 = duplicate_node_as_child(node, node_mod=400)
    child2.TYPE = -1
    return child1


# def get_sister_waste(node):
#     assert node.TYPE in [-1, -3], \
#         'node {} needs to be a waste!'.format(node.name)
#     axis, dim = get_orientation_from_cut(node)
#     child1 = duplicate_node_as_child(node, node_mod=200)
#     setattr(child1, dim, 0)
#     return child1


def delete_only_child(node):
    if len(node.children) != 1:
        return False
    child = node.get_children()[0]
    parent = node.up
    if parent is None:
        return False
    # copying features is not enough.
    # I have to move the child one level above.
    # and delete the node
    mod_feature_node(child, quantity=-1, feature='CUT')
    if child.children:
        mod_feature_node(child, quantity=-1, feature='CUT')
        for ch in child.get_children():
            ch.detach()
            parent.add_child(ch)
    else:
        child.detach()
        parent.add_child(child)
    parent.remove_child(node)
    order_children(parent)
    return True


def add_sibling_waste(node, size, dim):
    axis = get_axis_of_dim(dim)
    features = get_features(node)
    features[axis] += getattr(node, dim)
    features[dim] = size
    features['TYPE'] = -1
    features['NODE_ID'] += 100
    waste = create_node(**features)
    node.add_sister(waste)
    return True


def add_child_waste(node, fill):
    recalculate = False
    axis, dim_i = get_orientation_from_cut(node, inv=True)
    node_size = getattr(node, dim_i)
    child_size = fill - node_size
    if child_size <= 0:
        # sometimes we want a node without children
        # (because it had waste as only other child and now it hasn't).
        if len(node.children) == 1:
            delete_only_child(node)
        return False, recalculate
    if node.is_leaf():
        # not sure about this though...
        if node.TYPE == -2:
            node.TYPE = -1
        # sometimes we have a node without children
        # (because it had no waste and now it has).
        duplicate_node_as_child(node, 600)
        recalculate = True
    features = get_features(node)
    features[axis] += node_size
    features[dim_i] = child_size
    features['TYPE'] = -1
    features['CUT'] += 1
    features['NODE_ID'] += 100
    child = create_node(**features)
    # print('created child in node {} with ID={}'.
    #       format(node.NODE_ID, child.NODE_ID))
    node.add_child(child)
    setattr(node, dim_i, fill)
    return True, recalculate


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


def split_waste(node1, cut, min_size):
    # first, we split one waste in two.
    # then we make a swap to one of the nodes.
    assert node1.TYPE in [-1, -3], "node {} needs to be waste!".format(node1.name)
    parent = node1.up
    axis, dim = get_orientation_from_cut(node1)
    attributes = [axis, dim]
    size = getattr(node1, dim)
    assert size > cut, "cut for node {} needs to be smaller than size".format(node1.name)
    features = get_features(node1)
    features['NODE_ID'] += 300
    node2 = create_node(**features)
    nodes = {1: node1, 2: node2}
    attr = {k: {a: getattr(nodes[k], a) for a in attributes} for k in nodes}
    attr[2][axis] = attr[1][axis] + cut
    attr[2][dim] = attr[1][dim] - cut
    attr[1][dim] = cut

    for k, info in attr.items():
        if info[dim] < min_size:
            return {}

    for k, node in nodes.items():
        setattr(node, axis, attr[k][axis])
        setattr(node, dim, attr[k][dim])
    parent.add_child(node2)
    order_children(parent)
    return nodes


def reduce_children(node):
    axis, dim = get_orientation_from_cut(node)
    node_size = min_size = getattr(node, dim)
    if len(node.children) < 2:
        return False
    for ch in node.get_children():
        # if it has a children which is already a item... nothing to do
        waste = None
        if ch.TYPE >= 0:
            return False
        elif ch.TYPE in [-1, -3]:
            waste = ch
        elif ch.TYPE == -2:
            waste = find_waste(ch, child=True)
        if waste is None:
            return False
        size = getattr(waste, dim)
        if size < min_size:
            min_size = size
    if min_size <= 0:
        return False
    # ok: we got here: we reduce all children and the node.
    # print("node and children {} are being reduced by {}".
    #       format(node.name, min_size))
    for ch in node.get_children():
        # print("redimensiono {}".format(ch.name))
        setattr(ch, dim, getattr(ch, dim) - min_size)
        if ch.TYPE == -2:
            resize_node(ch, dim, -min_size)
    setattr(node, dim, node_size - min_size)
    # now we need to create a sibling waste with the correct size:
    features = get_features(node)
    features[axis] += features[dim]
    features[dim] = min_size
    features['NODE_ID'] += 700
    features['TYPE'] = -1
    node2 = create_node(**features)
    node.add_sister(node2)
    order_children(node.up)
    return True


def check_children_fit(node):
    if not node.children:
        return True
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    return sum(getattr(n, dim_i) for n in node.get_children()) == getattr(node, dim_i)


def node_inside_node(node1, node2, **kwargs):
    square1 = node_to_square(node1)
    square2 = node_to_square(node2)
    return geom.square_inside_square(square1, square2, **kwargs)


def check_children_inside(node):
    return [n for n in node.get_children() if not node_inside_node(n, node, both_sides=False)]


# def check_children_sibling(node):
#
#     return [n for n in node.get_children() if not node_inside_node(n, node, both_sides=False)]


def assign_cut_numbers(node, cut=0, update=True):
    result = {node: cut}
    if update:
        node.CUT = cut
    for n in node.get_children():
        result = {**result, **assign_cut_numbers(n, cut=cut+1, update=update)}
    return result


def search_node_of_defect(node, defect):
    # TODO: I could give a max_level. If I arrive, I return the type=2 node
    def before_defect(_node):
        axis, dim = get_orientation_from_cut(_node)
        return defect[axis] > getattr(_node, axis) + getattr(_node, dim)
    for n in node.traverse('preorder', is_leaf_fn=before_defect):
        if before_defect(n):
            continue
        if n.TYPE != -2:
            return n
    return None


def get_node_pos_tup(node):
    return (node.PLATE_ID, node.X, node.Y)


def defects_in_node(node, defects):
    """
    :param node:
    :param defects: defects to check
    :return: [defect1, defect2]
    """
    square = node_to_square(node)
    defects_in_node = []
    for defect in defects:
        square2 = geom.defect_to_square(defect)
        if geom.square_intersects_square(square2, square):
            defects_in_node.append(defect)
    return defects_in_node



if __name__ == "__main__":

    import package.params as pm
    import package.solution as sol

    case = 'A2'
    path = pm.PATHS['experiments'] + case + '/'
    solution = sol.Solution.from_io_files(path=path, case_name=case)
    defects = solution.get_defects_per_plate()
    defect = defects[0][0]
    node = solution.trees[0]

    search_node_of_defect(node, defect)
