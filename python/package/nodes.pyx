# cython: profile=False

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


def create_plate(width, height, id, defects, cut_defects=False):
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
    plate.NODE_ID = get_code_node(plate)
    duplicate_waste_into_children(plate)
    plate.add_feature('DEFECTS', defects)
    if cut_defects:
        pass
    return plate


# def node_cut_wastes(node, min_width):
#     for defect in get_defects_from_plate(node):
#         node_to_cut = search_node_of_defect(node, defect)
#         position = defect.X
#         pos_node = (node_to_cut.X, node_to_cut.X + node_to_cut.WIDTH)
#         pos_range = (position - min_width/2, position + min_width/2)
#         if pos_range[0] < pos_node[0]:
#             pos_range = (pos_node[0], pos_node[0] + min_width)
#         if pos_range[1] < pos_node[1]:
#             pos_range = (pos_node[1] - min_width, pos_node[1])
#         # TODO: still need to do check neighbor nodes sizes...
#
#         # nodes = split_waste(node_to_cut, pos_range[0], min_width)
#         split_waste(node_to_cut, pos_range[1], min_width)
#         pass
#     return node


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


def resize_waste(waste, dim, quantity, delete_if_empty=True):
    """
    :param waste:
    :param dim:
    :param quantity:
    :param delete_if_empty:
    :return: 2 if succesully deleted waste. 1 if only edited
    """
    if not is_waste(waste):
        a = 1
    assert is_waste(waste), "node is not a waste! {}".format(waste.name)
    parent = waste.up
    if parent is None:
        return False
    new_size = getattr(waste, dim) + quantity
    log.debug('waste {} in node {} is being changed by {} to {}'.format(waste.name, parent.name, quantity, new_size))
    setattr(waste, dim, new_size)
    plate, pos = get_node_plate_pos(waste)
    for ch in parent.children[pos+1:]:
        mod_feature_node(ch, quantity, get_axis_of_dim(dim))
    if not getattr(waste, dim) and delete_if_empty:
        log.debug('waste {} is being removed from node {}'.format(waste.name, parent.name))
        waste.detach()
        return 2
    return 1


def resize_node(node, dim, quantity, delete_if_empty=True):
    waste = find_waste(node, child=True)
    if waste is None:
        # if we need to create a waste: no problemo.
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
    if not getattr(waste, dim) and delete_if_empty:
        waste.detach()
    delete_only_child(node, collapse_child=True)
    return True

#
# def resize_node_2(node, dim, quantity):
#


def rotate_node(node, subnodes=True):
    log.debug('Node {} is being rotated'.format(node.name))
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
    if not is_waste(waste):
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


#TODO: stop passing strings all around
#cpdef char* get_dim_of_node(node, bint inv=False):
cpdef object get_dim_of_node(node, bint inv=False):
    cdef bint result
    result = node.CUT % 2
    if inv:
        result = not result
    if result:  # cuts 1 and 3
        return 'WIDTH'
    # cut 2 and 4
    return 'HEIGHT'

#Encoding text to bytes¶
#py_byte_string = py_unicode_string.encode('UTF-8')
#cdef char* c_string = py_byte_string

#Decoding bytes to text¶
#ustring = some_c_string.decode('UTF-8')


def get_orientation_from_cut(node, bint inv=False):
    # inv: means inverse the result.
#    cdef char* dim
#    cdef char* axis
    dim = get_dim_of_node(node, inv)
    axis = get_axis_of_dim(dim)
#    print(dim, axis)
#    return axis.decode('UTF-8'), dim.decode('UTF-8')
    return axis, dim


#cpdef char* get_axis_of_dim(char* dim):
cpdef object get_axis_of_dim(dim):
#    print(dim)
    cdef dict r
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
    next(get_next_sibling_iter(node), None)
    # if node.up is None:
    #     return None
    # new_pos = get_node_pos(node) + 1
    # children = node.up.children
    # if len(children) == new_pos:
    #     return None
    # return children[new_pos]


def get_next_sibling_iter(node):
    parent = node.up
    if parent is None:
        yield
    new_pos = get_node_pos(node) + 1
    return get_next_children_iter(parent, new_pos)


def get_next_children_iter(node, pos_start=0, pos_end=None):
    children = node.children
    if pos_end is None:
        pos_end = len(children)
    while pos_start < pos_end:
        yield children[pos_start]
        pos_start += 1


def get_node_position_cost_unit(node, plate_width):
    return (node.PLATE_ID+1) * plate_width + node.X


def get_node_position_cost(node, plate_width):
    # we'll give more weight to things that are in the right and up.
    # I guess it depends on the size too...
    return get_node_position_cost_unit(node, plate_width) * (node.WIDTH * node.HEIGHT)


#cdef int get_size_without_waste(node, char* dim):
def get_size_without_waste(node, dim):
    waste = find_waste(node, child=True)
    if waste is None:
        return getattr(node, dim)
    return getattr(node, dim) - getattr(waste, dim)


#cdef int  get_size_without_wastes(node, char* dim):
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


def get_defects_from_plate(node):
    return node.get_tree_root().DEFECTS

def node_to_square(node):
    """
    Reformats a piece to a list of two points
    :param node: a leaf from ete3 tree.
    :return: list of two points {'X': 1, 'Y': 1}
    """
    axis = ['X', 'Y']
    axis_dim = {'X': 'WIDTH', 'Y': 'HEIGHT'}
#    return [{a: getattr(node, a) for a in axis},
#     {a: getattr(node, a) + getattr(node, axis_dim[a]) for a in axis}]
    return {
     'DL': {a: getattr(node, a) for a in axis},
     'UR': {a: getattr(node, a) + getattr(node, axis_dim[a]) for a in axis}
    }


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
#    print(dim_i)
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
    # node_dim = getattr(node, dim)
    # if np.all(node_dim==getattr(child, dim) for child in node.children):
    if getattr(node, dim) == getattr(node.children[0], dim):
        # children are okay: no need to collapse
        return [node]
    log.debug('We collapse node {} into its children: {}'.format(node.name, get_children_names(node)))
    parent = node.up
    assert parent is not None
    mod_feature_node(node, quantity=-1, feature='CUT')
    children = node.children
    node.delete()
    order_children(parent)
    return children


def delete_only_child(node, check_parent=True, collapse_child=False):
    if len(node.children) != 1:
        return node
    child = node.children[0]
    parent = node.up
    if parent is None:
        if check_parent:
            return node
    log.debug('We collapse node {} into its one children: {}'. format(node.name, child.name))
    # if child.TYPE == -2:
    #     a = 1
    mod_feature_node(node, quantity=-1, feature='CUT')
    node.delete()
    if parent is not None:
        order_children(parent)
    if collapse_child:
        collapse_node(child)
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
            delete_only_child(node, collapse_child=True)
        return False, recalculate

    if node.is_leaf():
        # not sure about this though...
        if node.TYPE == -2:
            node.TYPE = -1
        # sometimes we have a node without children
        # (because it had no waste and now it has).
        if is_waste(node):
            # it its a waste, we expand only:
            mod_feature_node(node, child_size, dim_i)
            return True, recalculate
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
    log.debug('created waste of size {} inside node {} with ID={}'.
              format(child_size, node.NODE_ID, child.NODE_ID))
    node.add_child(child)
    order_children(node)
    setattr(node, dim_i, fill)
    return True, recalculate


def del_child_waste(node):
    axis, dim_i = get_orientation_from_cut(node, inv=True)
    child = find_waste(node, child=True)
    if child is None:
        return False
    child.detach()
    log.debug('deleted waste inside node {} with ID={}'.format(node.name, child.name))
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
    assert is_waste(node1), "node {} needs to be waste!".format(node1.name)
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


def reduce_children(node, min_waste):
    """
    cuts level one nodes that have extra waste.
    :param node:
    :return:
    """
    axis, dim = get_orientation_from_cut(node)
    node_size = getattr(node, dim)
    if len(node.children) <= 1:
        return False
    all_wastes = []
    for ch in node.get_children():
        waste = None
        # if it has a child which is already an item... nothing to do
        if ch.TYPE >= 0:
            return False
        elif is_waste(ch):
            waste = ch
        elif ch.TYPE == -2:
            waste = find_waste(ch, child=True)
        if waste is None:
            return False
        all_wastes.append(waste)
        # size = getattr(waste, dim)
        # if size < min_size:
        #     min_size = size
    sizes = sorted(getattr(waste, dim) for waste in all_wastes)
    min_size, second_min_size = sizes[:2]
    if min_size <= 0:
        return False
    if second_min_size - min_size < min_waste:
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
        result.update(assign_cut_numbers(n, cut=cut+1, update=update))
    return result


def search_nodes_of_defect(node, defect):
    nodes = []
    def before_defect(_node):
        axis, dim = get_orientation_from_cut(_node)
        return defect[axis] >= getattr(_node, axis) + getattr(_node, dim)

    def after_defect(_node):
        axis, dim = get_orientation_from_cut(_node)
        return defect[axis] + defect[dim] <= getattr(_node, axis)

    for n in node.traverse('preorder', is_leaf_fn=before_defect):
        # we have not arrived yet to the defect
        if before_defect(n):
            continue
        # we have passed the defect
        if after_defect(n):
            return nodes
        # we have found a node that intersects the defect
        if n.TYPE != -2:
            nodes.append(n)
    return nodes


def defects_in_node(node):
    """
    :param node:
    :return: [defect1, defect2]
    """
    square = node_to_square(node)
    defects = get_defects_from_plate(node)
    if node.CUT == 0:
        return defects
    defects_in_node = []
    for defect in defects:
        square2 = geom.defect_to_square(defect)
        if geom.square_intersects_square(square2, square):
            defects_in_node.append(defect)
    return defects_in_node


cpdef bint is_waste(object node):
    return node.TYPE in [-1, -3]


cdef bint is_last_sibling(object node):
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
    ch_pos = get_node_pos(node)
    axis, dim = get_orientation_from_cut(node)
    for sib in parent.children[ch_pos+1:]:
        mod_feature_node(node=sib, quantity=-getattr(node, dim), feature=axis)
    log.debug('We extract node {} from its parent: {}'.
              format(node.name, parent.name))
    node.detach()
    # In case there's any waste at the end: I want to trim it.
    del_child_waste(node)
    return node


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


def check_node_order(node1, node2):
    """
    These nodes are assumed to belong to the same tree.
    :param node1:
    :param node2:
    :return: True if node2 comes before node2
    """
    n1ancestors = set([node1] + node1.get_ancestors())
    n2ancestors = set([node2] + node2.get_ancestors())
    common_ancestors = n1ancestors & n2ancestors
    ancestor = max(common_ancestors, key=lambda x: x.CUT)
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


def get_node_by_type(node, type):
    for n in node.traverse():
        if n.TYPE == type:
            return n
    return None

def normalize_weights(weights):
    min_value = min(weights) - 0.1
    weights_temp = [w - min_value for w in weights]
    total = sum(weights_temp)
    return [w/total for w in weights_temp]


def debug_node(node):
    print("name={}\nPLATE_ID={}\nX={}\nY={}\nchildren={}\nCUT={}\nTYPE={}".format(
        node.name, node.PLATE_ID, node.X, node.Y, [ch.name for ch in node.children],
        node.CUT, node.TYPE)
    )


def debug_nodes( nodes):
    for node in nodes:
        debug_node(node)
        print("")


def join_neighbors(node1, node2):
    # this only makes sense if both
    # nodes are type=-1 (waste)
    if node1 == node2:
        return False
    parent = node1.up
    assert parent == node2.up, \
        '{} and {} are not siblings'.format(node1.name, node2.name)
    assert is_waste(node1) and is_waste(node2), \
        '{} and {} are not waste'.format(node1.name, node2.name)

    # this is okay because they are siblings:
    axis, dim = get_orientation_from_cut(node1)
    node1pos = getattr(node1, axis)
    node2pos = getattr(node2, axis)
    if not (node1pos + getattr(node1, dim) == node2pos):
        draw(node1.up, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
        assert (node1pos + getattr(node1, dim) == node2pos), \
            '{} and {} are not neighbors'.format(node1.name, node2.name)
    new_size = getattr(node1, dim) + getattr(node2, dim)
    # we need to update the first node because is the one that comes first
    setattr(node1, dim, new_size)
    node2.detach()
    return True


if __name__ == "__main__":

    import package.params as pm
    import package.solution as sol

    case = 'A2'
    path = pm.PATHS['experiments'] + case + '/'
    solution = sol.Solution.from_io_files(path=path, case_name=case)
    defects = solution.get_defects_per_plate()
    defect = defects[0][0]
    node1 = solution.trees[0]

    # search_node_of_defect(node1, defect)