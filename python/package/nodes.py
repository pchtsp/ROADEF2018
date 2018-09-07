import ete3
import package.geometry as geom
import package.superdict as sd
import logging as log
import random as rn
import numpy as np
from collections import deque
import copy
import functools as ft

# These are auxiliary functions for nodes of trees (see ete3).
# TODO: this should be a subclass of TreeNode...


def item_to_node(item):
    equivalence = \
        {'WIDTH': 'WIDTH_ITEM',
         'HEIGHT': 'LENGTH_ITEM',
         'TYPE': 'ITEM_ID',
         'NODE_ID': 'ITEM_ID',
         }

    default  =\
        {'CUT': 0,
            'X': 0,
            'Y': 0,
            'PLATE_ID': 0
         }
    features = {k: item[v] for k, v in equivalence.items()}
    args = {**default, **features}
    return create_node(**args)


def node_to_item(node):
    equivalence = \
        {'WIDTH': 'WIDTH_ITEM',
         'HEIGHT': 'LENGTH_ITEM',
         'TYPE': 'ITEM_ID',
         'NODE_ID': 'ITEM_ID',
         }
    return {v: getattr(node, k) for k, v in equivalence.items()}


def create_plate(width, height, id, defects):
    args = {'WIDTH': width,
            'HEIGHT': height,
            'CUT': 0,
            'X': 0,
            'Y': 0,
            'TYPE': -3,
            'NODE_ID': 0,
            'PLATE_ID': id
            }
    plate = create_node(**args)
    duplicate_waste_into_children(plate)
    plate.add_feature('DEFECTS', defects)
    return plate


def create_dummy_tree(nodes, id=-1):
    """
    This function creates a very big tree and puts the nodes list inside as children.
    The does not comply with space and size requirements.
    :param nodes: list of nodes to insert as children
    :return: tree object.
    """
    dummyTree = create_plate(width=999999, height=999999, id=id, defects=[])
    dummyWaste = dummyTree.copy()
    dummyWaste.TYPE = -1
    for n in nodes:
        dummyTree.add_child(n)
    dummyTree.add_child(dummyWaste)
    return dummyTree


def create_node(**kwargs):
    node = ete3.Tree(name=kwargs['NODE_ID'])
    node.add_features(**kwargs)
    return node


def mod_feature_node(node, quantity, feature):
    if not quantity:
        return True
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


def resize_waste(waste, dim, quantity):
    assert is_waste(waste), "node is not a waste! {}".format(waste.name)
    parent = waste.up
    if parent is None:
        return False
    log.debug('waste {} is being reduced by {}'.format(waste.name, quantity))
    setattr(waste, dim, getattr(waste, dim) + quantity)
    plate, pos = get_node_plate_pos(waste)
    for ch in parent.children[pos+1:]:
        mod_feature_node(ch, quantity, get_axis_of_dim(dim))
    if not getattr(waste, dim):
        log.debug('waste {} is being removed'.format(waste.name))
        waste.detach()
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

#
# def resize_node_2(node, dim, quantity):
#


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
        children = parent.children
    if not children:
        return None
    waste = children[-1]
    if waste.TYPE not in [-1, -3]:
        return None
    if waste == node:
        return None
    return waste


def get_size_last_waste(node, dim='WIDTH'):
    waste = find_waste(node)
    if waste is None:
        return 0
    return getattr(node, dim)


def find_all_wastes(node):
    # same as find_waste but returns all wastes
    return [w for w in node.children if is_waste(w)]

# @profile
def find_all_wastes_after_defect(node):
    """
    :param node: the node to search wastes
    :return: list of wastes that are next to the defects
    """
    # We assume that the last child of the node, if it's a waste,
    # it's available
    if node is None:
        return []
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    defects = defects_in_node(node)
    up_right = [x[dim_i] + x[axis_i] for x in defects]
    last_defect = max(up_right, default=-1)
    if not len(node.children):
        return []
    last_children = node.children[-1]
    # TODO: optimise
    return [w for w in node.children
            if is_waste(w) and (getattr(w, axis_i) > last_defect or
            w == last_children)
            ]


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


def get_code_node(node):
    return rn.randint(1, 100000)


def get_dim_of_node(node, inv):
    result = node.CUT % 2
    if inv:
        result = not result
    if result:  # cuts 1 and 3
        return 'WIDTH'
    # cut 2 and 4
    return 'HEIGHT'


def get_orientation_from_cut(node, inv=False):
    # inv: means inverse the result.
    dim = get_dim_of_node(node, inv)
    axis = get_axis_of_dim(dim)
    return axis, dim


def get_axis_of_dim(dim):
    r = {
        'HEIGHT': 'Y',
        'WIDTH': 'X'
    }
    return r[dim]


def get_node_leaves(node, min_type=0, max_type=99999, type_options=None):
    # print(node)
    if type_options is None:
        return [leaf for leaf in node.get_leaves() if min_type <= leaf.TYPE <= max_type]
    if type(type_options) is not list:
        raise ValueError("type_options needs to be a list instead of {}".
                         format(type(type_options)))
    return [leaf for leaf in node.get_leaves() if leaf.TYPE in type_options]


def get_node_plate_pos(node):
    pos = get_node_pos(node)
    return node.PLATE_ID, pos


def get_node_pos(node):
    if node.up is None:
        return 0
    pos = node.up.children.index(node)
    return pos


def get_next_sibling(node):
    if node.up is None:
        return None
    new_pos = get_node_pos(node) + 1
    children = node.up.children
    if len(children) == new_pos:
        return None
    return children[new_pos]


def get_node_position_cost_unit(node, plate_width):
    return ((node.PLATE_ID+1) * plate_width + node.X + node.Y/10)**2


def get_node_position_cost(node, plate_width):
    # we'll give more weight to things that are in the right and up.
    # I guess it depends on the size too...
    return get_node_position_cost_unit(node, plate_width) * (node.WIDTH * node.HEIGHT)


def get_size_without_waste(node, dim):
    waste = find_waste(node, child=True)
    if waste is None:
        return getattr(node, dim)
    return getattr(node, dim) - getattr(waste, dim)


def get_size_without_wastes(node, dim):
    wastes = find_all_wastes(node)
    sum_waste_dims = sum(getattr(waste, dim) for waste in wastes)
    return getattr(node, dim) - sum_waste_dims


def get_node_pos_tup(node):
    return node.PLATE_ID, node.X, node.Y


def get_children_names(node):
    return [ch.name for ch in node.children]


def get_last_sibling(node):
    assert node.up is not None, 'node is root node'
    return node.up.children[-1]


def get_features(node, features=None):
    if features is None:
        features = default_features()
    attrs = {k: int(getattr(node, k)) for k in features}
    parent = node.up
    if parent is not None:
        parent = int(parent.NODE_ID)
    attrs['PARENT'] = parent
    return attrs


def get_surplus_dim(node):
    if node.is_leaf():
        return 0
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    return sum(getattr(n, dim_i) for n in node.get_children()) - getattr(node, dim_i)


def get_item_density_node(node):
    return sum(item.HEIGHT * item.WIDTH for item in get_node_leaves(node)) /\
           ((node.HEIGHT+1) * (node.WIDTH+1))


def get_waste_density_node(node):
    return sum(waste.HEIGHT * waste.WIDTH for waste in get_node_leaves(node, type_options=[-1, -3])) /\
           ((node.HEIGHT+1) * (node.WIDTH+1))


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


def get_defects(node):
    return node.get_tree_root().DEFECTS

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


def duplicate_node_as_its_parent(node, node_mod=900, return_both=False):
    features = get_features(node)
    features['NODE_ID'] = get_code_node(node)
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
    features['NODE_ID'] = get_code_node(node)
    child = create_node(**features)
    # we increase the cut recursively among all children
    mod_feature_node(child, 1, "CUT")
    node.add_child(child)
    log.debug('created in node ID={}, TYPE={} a child with ID={}, TYPE={}'.format(
        node.NODE_ID, node.TYPE, child.NODE_ID, child.TYPE)
    )
    node.TYPE = -2
    return child


def duplicate_waste_into_children(node):
    assert is_waste(node), \
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


def collapse_node(node):
    # I will only collapse it if it has incorrect children
    if len(node.children) <= 1:
        # node has only on child, cannot collapse
        return [node]
    axis, dim = get_orientation_from_cut(node)
    if getattr(node, dim) == getattr(node.children[0], dim):
        # child is okay: no need to collapse
        return [node]
    log.debug('We collapse node {} into its children: {}'.
              format(node.name, get_children_names(node)))
    parent = node.up
    assert parent is not None
    mod_feature_node(node, quantity=-1, feature='CUT')
    children = node.children
    node.delete()
    # TODO: the following I can avoid I think:
    order_children(parent)
    return children


def delete_only_child(node, check_parent=True):
    # TODO: I think this is easier if I use node.delete()
    if len(node.children) != 1:
        return node
    child = node.children[0]
    parent = node.up
    if parent is None:
        if check_parent:
            return node
        else:
            node.delete()
            return child
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
    return child


def add_sibling_waste(node, size, dim):
    log.debug('adding a waste on node {}'.format(node.name))
    axis = get_axis_of_dim(dim)
    features = get_features(node)
    features[axis] += getattr(node, dim)
    features[dim] = size
    features['TYPE'] = -1
    features['NODE_ID'] = get_code_node(node)
    waste = create_node(**features)
    node.add_sister(waste)
    return True


def add_child_waste(node, child_size, waste_pos=None, increase_node=True):
    recalculate = False
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    node_size = getattr(node, dim_i)
    # child_size = fill - node_size
    if increase_node:
        fill = node_size + child_size
    else:
        fill = node_size
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
    if waste_pos is None:
        # if position not assigned, we assume is at the end
        waste_pos = node_size
    features[axis_i] += waste_pos
    features[dim_i] = child_size
    features['TYPE'] = -1
    features['CUT'] += 1
    features['NODE_ID'] = get_code_node(node)
    child = create_node(**features)
    log.debug('created waste inside node {} with ID={}'.
              format(node.NODE_ID, child.NODE_ID))
    node.add_child(child)
    setattr(node, dim_i, fill)
    return True, recalculate


def del_child_waste(node):
    axis, dim_i = get_orientation_from_cut(node, inv=True)
    child = find_waste(node, child=True)
    if child is None:
        return False
    child.detach()
    new_size = getattr(node, dim_i) - getattr(child, dim_i)
    setattr(node, dim_i, new_size)
    return True


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
    features['NODE_ID'] = get_code_node(node1)
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
    log.debug("node and children {} are being reduced by {}".format(
        node.name, min_size)
    )
    for ch in node.get_children():
        setattr(ch, dim, getattr(ch, dim) - min_size)
        if ch.TYPE == -2:
            resize_node(ch, dim, -min_size)
    setattr(node, dim, node_size - min_size)
    # now we need to create a sibling waste with the correct size:
    features = get_features(node)
    features[axis] += features[dim]
    features[dim] = min_size
    features['NODE_ID'] = get_code_node(node)
    features['TYPE'] = -1
    node2 = create_node(**features)
    node.add_sister(node2)
    order_children(node.up)
    return True


def check_children_fit(node):
    return get_surplus_dim(node) == 0


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


def defects_in_node(node):
    """
    :param node:
    :return: [defect1, defect2]
    """
    square = node_to_square(node)
    defects = get_defects(node)
    if node.CUT == 0:
        return defects
    defects_in_node = []
    for defect in defects:
        square2 = geom.defect_to_square(defect)
        if geom.square_intersects_square(square2, square):
            defects_in_node.append(defect)
    return defects_in_node


def is_waste(node):
    return node.TYPE in [-1, -3]


def is_last_sibling(node):
    return get_last_sibling(node) == node


def draw(node, *attributes):
    if attributes is None:
        attributes = ['NODE_ID']
    print(node.get_ascii(show_internal=True, attributes=attributes))
    return


def extract_node_from_position(node):
    # take node out from where it is (detach and everything)
    # update all the positions of siblings accordingly
    parent = node.up
    plate, ch_pos = get_node_plate_pos(node)
    axis, dim = get_orientation_from_cut(node)
    for sib in parent.children[ch_pos+1:]:
        mod_feature_node(node=sib, quantity=-getattr(node, dim), feature=axis)
    log.debug('We extract node {} from its parent: {}'.
              format(node.name, parent.name))
    node.detach()
    # In case there's any waste at the end: I want to trim it.
    del_child_waste(node)
    return node


def repair_dim_node(node, min_waste):
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    change = get_surplus_dim(node)
    node_size = getattr(node, dim_i)
    # if surplus is negative: we need to add a waste, it's easier
    # if surplus is positive: we start deleting wastes.
    if change < 0:
        add_child_waste(node, child_size=-change,
                        waste_pos=node_size+change,
                        increase_node=False)
        return True
    wastes = find_all_wastes_after_defect(node)
    # wastes = find_all_wastes(node)
    # TODO: get better ways to eat wastes.
    # we revert to the previous change
    # we want the farthest at the end:
    # but we want to eliminate really small wastes before
    # wastes.sort(key=lambda x: (getattr(x, dim_i) < 20, getattr(x, axis_i)))

    # we want the smallest at the end:
    wastes.sort(key= lambda x: getattr(x, dim_i), reverse=True)
    remaining = change
    comply_min_waste = True
    while wastes and remaining:
        waste = wastes.pop()
        size = getattr(waste, dim_i)
        quantity = size
        if remaining < size:
            waste_rem = size - remaining
            if min_waste > waste_rem > 0 and comply_min_waste:
                quantity = size - min_waste
            else:
                quantity = remaining
        resize_waste(waste, dim_i, -quantity)
        remaining -= quantity
        # If we did all we could and still have remaining.
        # we relax the min size constraint and do one last turn.
        if not len(wastes) and comply_min_waste:
            wastes = find_all_wastes(node)
            wastes.sort(key=lambda x: getattr(x, axis_i))
            comply_min_waste = False
    if remaining > 0:
        assert remaining == 0, "repair_dim_node did not eliminate all waste. Left={}".format(remaining)
    return True


def get_nodes_between_nodes_in_tree(node1=None, node2=None):
    """
    This procedure searches a tree for all nodes between node1 and node2.
    node1 is assumed to come before node2.
    In case node1 is None: it should start in the first node
    If node2 is None: it should end in the last node
    :param node1:
    :param node2:
    :return: list: the order of the nodes and a list of nodes in between.
    """
    if node1 is None and node2 is None:
        raise ValueError("node1 and node2 cannot be None at the same time")

    ancestors = set()
    if node1:
        ancestors |= set(node1.get_ancestors() + [node1])
    if node2:
        ancestors |= set(node2.get_ancestors() + [node2])
    if node1 and node2:
        assert node1.get_tree_root() == node2.get_tree_root(), \
            "node {} does not share root with node {}".format(node1.name, node2.name)
        # if one of the nodes is the ancestor: there's no nodes in between
        ancestor = node1.get_common_ancestor(node2)
        if ancestor in [node1, node2]:
            return []

    if not node1:
        node1 = get_descendant(node2.get_tree_root(), which='first')

    def is_leaf_fn(node):
        return node not in ancestors

    nodes = [n for n in
             post_traverse_from_node(node1,
                                     is_leaf_fn=is_leaf_fn,
                                     last_node=node2)
             ]

    return list(set(nodes) - ancestors)


def pre_traverse_from_node(node1, is_leaf_fn=None):
    """
    If it's an ancestor, iterate starting in the children that's to the right of the previous ancestor-children. And then go up.
    If it's not an ancestor, traverse normally from left to right.
    For this we need to store the position of each node in the children list.
    :return:
    """
    to_visit = deque()
    node = node1
    pos = get_node_pos(node1)
    pos += 1
    node1_ancestors = set(node1.get_ancestors() + [node1])
    while node is not None:
        yield node
        if node in node1_ancestors and node.up:
            pos_new = get_node_pos(node)
            to_visit.appendleft((node.up, pos_new+1))
        if not is_leaf_fn or not is_leaf_fn(node):
            nodes_to_add = node.children
            if pos:
                nodes_to_add = nodes_to_add[pos:]
            to_visit.extendleft(zip(reversed(nodes_to_add), [0]*len(nodes_to_add)))
        try:
            node, pos = to_visit.popleft()
        except:
            node = None


def post_traverse_from_node(node1, is_leaf_fn=None, last_node=None):
    to_visit = [(node1, 0)]
    pos = get_node_pos(node1)
    pos += 1
    node1_ancestors = set(node1.get_ancestors() + [node1])

    if last_node:
        top_root = node1.get_common_ancestor(last_node)

        def not_top_root(node):
            return node != top_root
    else:
        def not_top_root(node):
            return node.up

    if is_leaf_fn is not None:
        _leaf = is_leaf_fn
    else:
        _leaf = node1.__class__.is_leaf

    while to_visit:
        node, pos = to_visit.pop(-1)
        try:
            node = node[1]
        except TypeError:
            # PREORDER ACTIONS
            if node in node1_ancestors and not_top_root(node):
                pos_new = get_node_pos(node)
                to_visit.append((node.up, pos_new + 1))
            if not _leaf(node):
                # ADD CHILDREN
                nodes_to_add = node.children
                if pos:
                    nodes_to_add = nodes_to_add[pos:]
                to_visit.extend(
                    zip(
                        reversed(node.children + [[1, node]]),
                        np.zeros(len(nodes_to_add) + 1)
                    )
                )
            else:
                yield node
        else:
            #POSTORDER ACTIONS
            yield node
        if last_node and node == last_node:
            break


def check_assumptions_swap(node1, node2, insert):
    siblings = node1.up == node2.up
    if siblings and insert:
        return False
    if is_waste(node1) and is_waste(node2):
        return False
    if is_waste(node1) and insert:
        return False
    if node1.up == node2 or node1 == node2.up:
        return False
    # for now, we do not allow swapping between different levels
    # or inserting a node from a higher to a lower level
    if node1.CUT != node2.CUT and not insert:
        return False
    if node1.CUT < node2.CUT:
        return False
    return True


def check_node_order(node1, node2):
    """
    These nodes are assumed to belong to the same tree.
    :param node1:
    :param node2:
    :return: True if node2 comes before node2
    """
    ancestor = node1.get_common_ancestor(node2)
    n1ancestors = set([node1] + node1.get_ancestors())
    n2ancestors = set([node2] + node2.get_ancestors())
    for n in ancestor.children:
        if n in n1ancestors:
            return True
        if n in n2ancestors:
            return False
    return True


def order_nodes(node1, node2):
    first_node = node1
    second_node = node2
    if not check_node_order(node1, node2):
        first_node, second_node = second_node, first_node
    return first_node, second_node


def insert_node_at_position(node, destination, position):
    """
    :param node: node I'm going to insert.
    :param destination: parent node were I want to insert it.
    :param position: position of a children on the parent node (1 : num_children)
    :return:
    """
    # move all nodes at destination starting from the marked position
    # to make space and assign new plate, and new axis
    # add node to new parent
    axis_i, dim_i = get_orientation_from_cut(node, inv=True)
    axis, dim = get_orientation_from_cut(node)
    ref = {axis: dim, axis_i: dim_i}
    # sib_axis, sib_dim = get_orientation_from_cut(destination.children[position])
    # dest_axis, dest_dim = get_orientation_from_cut(destination)
    dest_axis_i, dest_dim_i = get_orientation_from_cut(destination, inv=True)

    if destination.children:
        if position < len(destination.children):
            # we get the destination position and then make space:
            axis_dest = {a: getattr(destination.children[position], a) for a in ref}
            # because siblings could have different level than my node, the movement
            # needs to be according to the siblings dimensions
            # which are the opposite of the destination (parent)
            for sib in destination.children[position:]:
                mod_feature_node(node=sib,
                                    quantity=getattr(node, dest_dim_i),
                                    feature=dest_axis_i)
        else:
            # we're puting the node in a new, last place:
            # so we do not move anything.
            last_node = destination.children[-1]
            axis_dest = {a: getattr(last_node, a) for a in ref}
            axis_dest[dest_axis_i] += getattr(last_node, dest_dim_i)
    else:
        axis_dest = {a: getattr(destination, a) for a in ref}

    # when doing weird multilevel swaps,
    # we need to keep the CUT level and hierarchy:
    # or we put the node inside itself.
    # or we take all nodes out of the node and insert them separately.
    if (node.CUT >= destination.CUT + 2) and node.children:
        node = duplicate_node_as_its_parent(node)

    # we make the move:
    change_feature(node, 'PLATE_ID', destination.PLATE_ID)
    dist = {a: axis_dest[a] - getattr(node, a) for a in ref}
    for k, v in dist.items():
        if v:
           mod_feature_node(node=node, quantity=v, feature=k)

    # In case we're moving nodes from different levels, we need to update the CUT:
    cut_change = destination.CUT + 1 - node.CUT
    if cut_change:
        mod_feature_node(node, feature='CUT', quantity=cut_change)

    log.debug('We insert node {} into destination {}'.
              format(node.name, destination.name))
    destination.add_child(node)

    # 3. update parents order:
    order_children(destination)
    return node


def check_swap_size(nodes, min_waste, insert=False, rotate=None):
    if rotate is None:
        rotate = []
        # ideally, there should be not reference to the tree here
        # so we can test nodes that are not part of a tree
    dims_i = {
        k: get_orientation_from_cut(node, inv=True)[1]
        for k, node in nodes.items()
    }
    dims = {
        k: get_orientation_from_cut(node)[1]
        for k, node in nodes.items()
    }

    wastes = {k: [
        w for w in find_all_wastes_after_defect(node.up)
        if w != node or (k == 2 and insert)
    ] for k, node in nodes.items()}
    # wastes = {k: find_waste(node) for k, node in nodes.items()}
    # if there's no waste, I'll just say it's a 0 length node?
    # for k, waste in wastes.items():
    #     if waste is None:
    #         wastes[k] = create_node(NODE_ID=12345, HEIGHT=0, WIDTH=0)
    waste_space = {k: sum(getattr(w, dims[k]) for w in wastes[k]) for k in nodes}

    node_space = {
        k: {
            dims[k]: getattr(node, dims[k]),
            dims_i[k]: get_size_without_waste(node, dims_i[k])
        }
        for k, node in nodes.items()
    }

    space = {
        k: {
            dims[k]: waste_space[k] + node_space[k][dims[k]],
            dims_i[k]: getattr(node, dims_i[k])
        }
        for k, node in nodes.items()
    }
    # if not swapping, we have less space in node2
    if insert:
        space[2][dims[2]] -= node_space[2][dims[2]]
        # # if node2 is a waste, I can use it as the destination's waste
        # # but on
        # if is_waste(nodes[2]) and wastes[2] is None:
        #     space[2][dims[2]] = max(space[2][dims[2]], node_space[2][dims[2]])

    # rotate can be a list with the index of nodes to reverse.
    # this way, we can check different combinations of rotation
    # it's usually en empty list
    for pos in rotate:
        _dim_i = dims_i[pos]
        _dim = dims[pos]
        node_space[pos][_dim], node_space[pos][_dim_i] = \
            node_space[pos][_dim_i], node_space[pos][_dim]

    return geom.check_nodespace_in_space(node_space, space, insert, min_waste)


def swap_nodes_same_level(node1, node2, min_waste, insert=False, rotation=None, debug=False):
    if rotation is None:
        rotation = []
    nodes = {1: node1, 2: node2}
    other_node = {1: 2, 2: 1}
    parents = {k: node.up for k, node in nodes.items()}
    parent1 = parents[1]
    parent2 = parents[2]
    ch_pos = {k: get_node_pos(node) for k, node in nodes.items()}

    recalculate = False
    nodes_to_move = []
    if debug:
        pass
        # self.draw(node1.PLATE_ID, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
        # self.draw(node2.PLATE_ID, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
    if node1.up != parent2 or ch_pos[1] != ch_pos[2]:
        nodes_to_move.append(1)
    if not insert and (node2.up != parent1 or ch_pos[1] + 1 != ch_pos[2]):
        nodes_to_move.append(2)

    for k in nodes_to_move:
        node = nodes[k]
        other_k = other_node[k]
        destination = parents[other_k]
        ch_pos_dest = ch_pos[other_k]
        node = extract_node_from_position(node)  # 1: take out children waste
        node = delete_only_child(node, check_parent=False)  # 1.5: collapse if only child
        if k in rotation:
            rotate_node(node)  # 2: rotate
        node = insert_node_at_position(node, destination, ch_pos_dest)  # 3, 4: insert+child

        # 5: if necessary, we open the node to its children
        inserted_nodes = collapse_node(node)
        for _node in inserted_nodes:
            for ch in _node.children:
                collapse_node(ch)

        # 6: now we check if we need to create a waste children on the node:
        _, dest_dim = get_orientation_from_cut(destination)
        for _node in inserted_nodes:
            dest_size = getattr(destination, dest_dim)
            node_size = getattr(node, dest_dim)
            status, recalculate = add_child_waste(node=_node, child_size=dest_size-node_size)
        nodes[k] = node

    # 7: we need to update the waste at both sides
    if parent1 != parent2:
        for parent in parents.values():
            repair_dim_node(parent, min_waste)

    return nodes
    # return recalculate


def check_swap_size_rotation(node1, node2, min_waste, params, insert=False):
    if node1.up == node2.up:
        return []
    rotations = [[]]
    nodes = {1: node1, 2: node2}
    rotation_tries = params['rotation_tries']
    try_rotation = params['try_rotation']
    rotation_probs = params['rotation_probs']
    if rotation_tries is None:
        rotation_tries = 1
    if try_rotation:
        rotations_av = [[], [1], [2], [1, 2]]
        if rotation_probs is None:
            rotation_probs = [0.8, 0.1, 0.05, 0.05]
        rotations = np.random.choice(a=rotations_av, p=rotation_probs, size=rotation_tries, replace=False)
    for rotation in rotations:
        if check_swap_size(nodes, min_waste=min_waste, insert=insert, rotate=rotation):
            return rotation
    return None


def try_change_node_simple(node, candidates, insert, min_waste, params, reverse=False):
    good_candidates = {}
    rotation = {}
    weights = params.get('weights', None)
    debug = params.get('debug', False)
    assert weights is not None, 'weights argument cannot be empty or None!'
    for candidate in candidates:
        node1, node2 = node, candidate
        if reverse:
            node1, node2 = candidate, node
        if not check_assumptions_swap(node1, node2, insert):
            continue
        result = check_swap_size_rotation(node1, node2, insert=insert,
                                          min_waste=min_waste, params=params)
        if result is None:
            continue
        rotation[node2] = result
        good_candidates[node2] = 0
    if len(good_candidates) == 0:
        return False
    candidates_prob = sd.SuperDict({k: v for k, v in good_candidates.items()}).to_weights()
    node2 = np.random.choice(a=candidates_prob.keys_l(), size=1, p=candidates_prob.values_l())[0]
    rot = rotation[node2]
    inserted_nodes = swap_nodes_same_level(node1, node2, insert=insert, rotation=rot,
                                           debug=debug, min_waste=min_waste)
    return inserted_nodes


def insert_node_inside_node_traverse(node1, node_start, min_waste, params):
    # we want to insert node1 at the first available space in node_start's tree
    # but never before node_start.
    # If i just want to traverse the whole tree, just need to put node1= root
    def is_leaf_fn(node2):
        return not \
            geom.plate_inside_plate(
                node_to_plate(node1),
                node_to_plate(node2),
                turn=True
            ) or node2.CUT >= 3

    for node2 in post_traverse_from_node(node_start, is_leaf_fn=is_leaf_fn):
        node1.CUT = node2.CUT
        if is_waste(node2):
            inserted_nodes = try_change_node_simple(node=node1, candidates=[node2], min_waste=min_waste, params=params, insert=True)
            if inserted_nodes:
                return inserted_nodes
            continue
        # If I failed inserting in the children or if node2 is an item:
        # I try to insert next to the node2
        next_sibling = get_next_sibling(node2)
        if next_sibling is None:
            continue
        inserted_nodes = try_change_node_simple(node=node1, candidates=[next_sibling], min_waste=min_waste, params=params, insert=True)
        if inserted_nodes:
            return inserted_nodes
    return False


def get_node_by_type(node, type):
    for n in node.traverse():
        if n.TYPE == type:
            return n
    return None


def iter_insert_items_on_trees(num_iters, **kwgs):
    result = []
    for x in range(num_iters):
        result.append(insert_items_on_trees(**kwgs))
    return result

def insert_items_on_trees(params, global_params, items_by_stack, defects, sorting_function, seed, limit_trees=None):
    """
    This algorithm just iterates over the items in the order of the sequence
    and size to put everything as tight as possible.
    Respects sequence.
    :return:
    """
    rn.seed(seed)
    np.random.seed(seed)
    values = sorting_function(items_by_stack)
    plate_W = global_params['widthPlates']
    plate_H = global_params['heightPlates']
    min_waste = global_params['minWaste']
    ordered_nodes = [item_to_node(v) for v in values]
    for n in ordered_nodes:
        if n.WIDTH > n.HEIGHT:
            rotate_node(n)
        if n.HEIGHT > plate_H:
            rotate_node(n)
    dummy_tree = create_dummy_tree(ordered_nodes, id=-1)
    tree_id = 0
    tree = create_plate(width=plate_W, height=plate_H, id=tree_id, defects=defects.get(tree_id, []))
    trees = [dummy_tree, tree]

    # For each item, I want the previous item.
    # Two parts:
    # 1. for each item we want it's previous item => this doesn't change
    # But we have to guarantee that the list of items is SORTED.
    item_prec = {}
    items_by_stack = {s: sorted(items, key=lambda x: x['SEQUENCE']) for s, items in items_by_stack.items()}
    for stack, items in items_by_stack.items():
        for i, i2 in zip(items, items[1:]):
            item_prec[i2['ITEM_ID']] = i['ITEM_ID']

    # 2. for each item we've placed, we want it's node => this changes
    item_node = {}

    for n in ordered_nodes:
        item_id = n.TYPE
        inserted_nodes = False
        t_start = 1
        # We first search the tree where the previous node in the sequence
        # only starting from the position of the previous node
        if item_id in item_prec:
            node2 = item_node[item_prec[item_id]]
            inserted_nodes = insert_node_inside_node_traverse(n, node2, min_waste=min_waste, params=params)
            if inserted_nodes:
                node = get_node_by_type(inserted_nodes[1], item_id)
                item_node[item_id] = node
                continue
            t_start = node2.PLATE_ID + 2
        # The first tree is our dummy tree so we do not want to use it.
        for tree in trees[t_start:]:
            inserted_nodes = insert_node_inside_node_traverse(n, tree, min_waste=min_waste, params=params)
            if inserted_nodes:
                break
        if inserted_nodes:
            node = get_node_by_type(inserted_nodes[1], item_id)
            item_node[item_id] = node
            continue
        tree_id = len(trees) - 1
        # If we arrive to the limit, it means we lost.
        # because we are about to create another tree.
        if limit_trees and tree_id == limit_trees:
            return None
        tree = create_plate(width=plate_W, height=plate_H, id=tree_id, defects=defects.get(tree_id, []))
        trees.append(tree)
        inserted_nodes = insert_node_inside_node_traverse(n, tree, min_waste=min_waste, params=params)
        # TODO: warning, in the future this could be possible due to defects checking
        assert inserted_nodes, "node {} apparently doesn't fit in a blank new tree".format(n.name)
        node = get_node_by_type(inserted_nodes[1], item_id)
        item_node[item_id] = node

    # we take out the dummy tree
    return trees[1:]


def sorting_items(items_by_stack):
    # I get a list of stacks of items.
    # The first items on the stack are at the end of the list.
    items_list_stack = [sorted(items, key=lambda x: -x['SEQUENCE'])
                        for stack, items in items_by_stack.items()]
    items_list = []
    # I get a random stack at every iteration and
    # get the first remaining sequence element from the stack.
    while len(items_list_stack):
        stack_num = rn.randrange(len(items_list_stack))
        stack = items_list_stack[stack_num]
        items_list.append(stack.pop())
        if not len(stack):
            items_list_stack.pop(stack_num)
    return items_list


def sorting_function_2(items_by_stack):
        # This is an implementation by comparing two elements
        # cmp = ft.cmp_to_key()
        # batch_data.sort(key=cmp)
        pass


def get_nodes_between_nodes(node1, node2, solution):
    """
    :param node1:
    :param node2:
    :return: (n1, n2, list): the order of the nodes and a list of nodes in between.
    """
    plate1 = node1.PLATE_ID
    plate2 = node2.PLATE_ID

    if plate1 == plate2:
        # if they're in the same plate: I just get the nodes between them
        node1, node2 = order_nodes(node1, node2)
        nodes = get_nodes_between_nodes_in_tree(node1=node1, node2=node2)
        return node1, node2, nodes

    if plate1 > plate2:
        node1, node2 = node2, node1
        plate1, plate2 = plate2, plate1
    # if not in the same plate: i have three parts:
    # the rest of node1's plate:
    nodes1 = get_nodes_between_nodes_in_tree(node1=node1)
    # the beginning of node2's plate:
    nodes2 = get_nodes_between_nodes_in_tree(node2=node2)
    nodes = nodes1 + nodes2
    # nodes in intermediate plates:
    i = plate1 + 1
    while i < plate2:
        nodes += solution[i]
        i += 1
    return node1, node2, nodes


def check_swap_nodes_seq(node1, node2, solution, precedence, precedence_inv, insert=False):
    """
    checks if a change is beneficial in terms of sequence violations
    :param node1:
    :param node2:
    :param insert: type of swap can be insert or swap
    :return: balance of violations. Bigger is better.
    """
    # get all leaves in node1 and node2
    nodes = {1: node1, 2: node2}
    moved_items = {k: get_node_leaves(v) for k, v in nodes.items()}
    # precedence = self.get_previous_nodes(type_node_dict=self.type_node_dict)
    # precedence_inv = self.get_next_nodes(type_node_dict=self.type_node_dict)
    # items1 = get_node_leaves(node1)
    # items2 = get_node_leaves(node2)
    # get all leaves between the two nodes
    first_node, second_node, neighbors = get_nodes_between_nodes(node1, node2, solution=solution)
    first_i, second_i = 1, 2
    if first_node != node1:
        first_i, second_i = 2, 1
    # print(neighbors)
    neighbor_items = set(leaf for node in neighbors for leaf in get_node_leaves(node))
    crossings = {k: {'items_after': set(), 'items_before': set()} for k in nodes}
    # neighbors between nodes are almost the same.
    # The sole difference is that the second node arrives *before* the first node
    neighbor_items_k = {1: neighbor_items.copy(), 2: neighbor_items}
    neighbor_items_k[second_i] |= set(moved_items[first_i])
    nodes_iter = [1]
    if not insert:
        nodes_iter = [1, 2]
    # items_after means items that are after in the sequence.
    # for each leaf in the first node:
        # because items are "going to the back":
        # if I find nodes that precede it: good
        # if I find nodes that follow it: bad
    for k in nodes_iter:
        for item in moved_items[k]:
            crossings[k]['items_before'] |= neighbor_items_k[k] & set(precedence.get(item, set()))
            crossings[k]['items_after'] |= neighbor_items_k[k] & set(precedence_inv.get(item, set()))
    balance = (
               len(crossings[first_i]['items_before']) -
               len(crossings[first_i]['items_after'])
           ) +\
           (
               len(crossings[second_i]['items_after']) -
               len(crossings[second_i]['items_before'])
           )
    return balance


def check_swap_nodes_defect(node1, node2, insert=False):
    # we'll only check if exchanging the nodes gets a
    # positive defect balance.
    # We need to:
    # 1. list the nodes to check (re-use the 'nodes between nodes'?)
    # here there can be two sets of nodes if two plates.
    # 1b. list the possible defects (for each the plate)
    # 1c. calculate the present defect violations in each plate.
    # 2. calculate the distance to move them.
    # 3. create squares with new positions.
    # 4. calculate new defect violations
    if node1 == node2:
        return 0
    if node1.up is None or node2.up is None:
        return -10000
    nodes = {1: node1, 2: node2}
    plate1, ch_pos1 = get_node_plate_pos(node1)
    plate2, ch_pos2 = get_node_plate_pos(node2)
    positions = {1: ch_pos1, 2: ch_pos2}
    plates = {1: plate1, 2: plate2}
    dims = {}
    axiss = {}
    dims_i = {}
    axiss_i = {}
    for k, node in nodes.items():
        axiss[k], dims[k] = get_orientation_from_cut(node)
        axiss_i[k], dims_i[k] = get_orientation_from_cut(node, inv=True)
    # axis, dim = get_orientation_from_cut(node1)
    # axis_i, dim_i = get_orientation_from_cut(node1, inv=True)
    siblings = node1.up == node2.up
    if siblings:
        # if they're siblings: I just get the nodes between them
        first_node, second_node = 1, 2
        if positions[1] > positions[2]:
            first_node, second_node = 2, 1
        neighbors = {
            first_node: node1.up.children[positions[first_node]+1:positions[second_node]],
            second_node: []
        }
        defects = {
            first_node: get_defects(nodes[1]),
            second_node: []
        }
        # If a defect is to the left of the first node
        # or to the right of the second node: take it out.
        right = filter_defects(nodes[first_node], defects[first_node])
        left = filter_defects(nodes[second_node], defects[first_node], previous=False)
        defects[first_node] = [d for d in right if d in left]
        # defects[second_node] = filter_defects(nodes[second_node], defects[second_node], previous=False)
    else:
        neighbors = {k: nodes[k].up.children[positions[k]+1:] for k in nodes}

        # If a defect is to the left / down of the node: take it out.
        defects = {k: get_defects(nodes[k]) for k in nodes}
        defects = {k: filter_defects(nodes[k], defects[k]) for k in nodes}

    # if there's no defects to check: why bother??
    if not defects[1] and not defects[2]:
        return 0

    # TODO: this can be generalized even more to make it shorter and (possibly)
    # easier to understand
    # and correct...

    if insert:
        move_neighbors = {
            1: getattr(nodes[1], dims[1]),
            2: getattr(nodes[1], dims[2])
        }
        pos_dif = {
            1: {'X': nodes[2].X - nodes[1].X, 'Y': nodes[2].Y - nodes[1].Y},
            2: {axiss[2]: move_neighbors[2], axiss_i[2]: 0}
        }
    else:  # complete swap
        # to get the movement for neighbors we need to compare the diff among
        # each node's dimension.
        move_neighbors = {
            1: getattr(nodes[1], dims[1]) - getattr(nodes[2], dims[1]),
            2: getattr(nodes[1], dims[2]) - getattr(nodes[2], dims[2])
        }
        pos_dif = {
            1: {'X': nodes[2].X - nodes[1].X, 'Y': nodes[2].Y - nodes[1].Y},
            2: {'X': - nodes[2].X + nodes[1].X, 'Y': - nodes[2].Y + nodes[1].Y}
        }
        # this works but does not make *any* sense:
        if siblings:
            pos_dif[first_node][axiss[first_node]] -= move_neighbors[first_node] * (2*(first_node == 1)-1)

    # get squares
    # first of the nodes involved
    nodes_sq = {k: [(node_to_square(p), pos_dif[k]) for p in get_node_leaves(nodes[k])]
                for k in nodes
                }
    # the destination node needs to be included in the first one in case of siblings:
    if siblings:
        nodes_sq[first_node] += nodes_sq[second_node]

    # second of the nodes in between
    dif = {
        1: {axiss[1]: -move_neighbors[1], axiss_i[1]: 0},
        2: {axiss[2]: move_neighbors[2], axiss_i[2]: 0}
    }
    for k, _neighbors in neighbors.items():
        nodes_sq[k] += [(node_to_square(p), dif[k]) for n in _neighbors for p in get_node_leaves(n)]

    # finally of the defects to check
    defects_sq = {k: [geom.defect_to_square(d) for d in defects[k]] for k in nodes}

    # here we edit the squares we created in (1) and (2)
    # squares is a list of two dictionaries.
    # We have for 'before' and 'after' the nodes affected indexed by 1 and 2.
    squares = [{k: [] for k in nodes_sq} for r in range(2)]
    for k, squares_changes in nodes_sq.items():
        for (sq, change) in squares_changes:
            _sq = copy.deepcopy(sq)
            squares[0][k].append(sq)
            for n in range(2):
                # we change the position in the two corners of the square
                for a in ['X', 'Y']:
                    _sq[n][a] += change[a]
            squares[1][k].append(_sq)

    # here I count the number of defects that collide with squares. Before and now.
    defects_found = [[] for r in range(2)]
    for pos in range(2):  # for (before) => after
        for k, sq_list in squares[pos].items():  # for each node
            for d in defects_sq[k]:  # for each defect
                for sq in sq_list:  # for each neighbor
                    if geom.square_intersects_square(d, sq):
                        defects_found[pos].append(d)
                        # if it's inside some node, it's not in the rest:
                        break

    # as the rest of checks: the bigger the better.
    return len(defects_found[0]) - len(defects_found[1])


def check_swap_space(node1, node2, global_params, insert=False):
    # (dif density) * (dif position)
    nodes = {1: node1, 2: node2}
    node_density = {
        n: get_item_density_node(node) for n, node in nodes.items()
    }

    if insert:
        node_density[2] = 0

    cost = {
        n: get_node_position_cost_unit(node, global_params['widthPlates'])
        for n, node in nodes.items()
    }
    # we divide it over cost[1] to scale it.
    # the bigger the better
    gains = (node_density[1] - node_density[2]) * (cost[1] - cost[2]) / cost[1]
    return gains


def evaluate_swap(weights, solution, precedence, precedence_inv, global_params, **kwargs):
    # the bigger the better
    components = {
        'space': check_swap_space(global_params=global_params, **kwargs)
        ,'seq': check_swap_nodes_seq(solution=solution, precedence=precedence,
                                     precedence_inv=precedence_inv, **kwargs)
        ,'defects': check_swap_nodes_defect(**kwargs)
    }
    gains = {k: v * weights[k] for k, v in components.items()}
    return sum(gains.values())


def check_swap_two_nodes(nodes, args_check_size, args_evaluate, evaluate, params):
    tolerance = params['tolerance']
    args_check_size = {**args_check_size, **nodes}
    result = check_swap_size_rotation(**args_check_size)
    if result is None:
        return None
    args_evaluate = {**args_evaluate, **nodes}
    if not evaluate:
        return 0
    balance = evaluate_swap(**args_evaluate)
    if tolerance is not None and balance <= tolerance:
        return None
    return (balance, result)


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