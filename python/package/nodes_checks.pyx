# cython: profile=False

import package.nodes as nd
import logging as log
# import pyximport; pyximport.install()
cimport package.geometry as geom
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import palettable as pal
import os


cdef object check_consistency(tree):
    func_list = {
        'ch_size': check_nodes_fit
        , 'inside': check_parent_of_children
        # , 'cuts': check_cuts_number
        , 'types': check_wrong_type
        , 'ch_order': check_children_order
        , 'node_size': check_sizes
        , 'only_child': check_only_child
    }
    result = {k: v(tree) for k, v in func_list.items()}
    return {k: v for k, v in result.items() if len(v) > 0}


def check_nodes_fit(node):
    nodes_poblems = []
    for n in node.traverse():
        if not nd.check_children_fit(n):
            nodes_poblems.append(n)
    return nodes_poblems

def check_parent_of_children(tree):
    """
    We want to check if each node is inside its parent.
    :return:
    """
    nodes_poblems = []
    for node in tree.traverse():
        children_with_problems = nd.check_children_inside(node)
        if children_with_problems:
            nodes_poblems.append((node, children_with_problems))
    return nodes_poblems


def check_cuts_number(tree):
    """
    This checks if the CUT property of each node really corresponds with
    the node's level.
    :return:
    """
    levels = nd.assign_cut_numbers(tree, update=False)
    bad_cut = []
    for node, level in levels.items():
        if node.CUT != level:
            bad_cut.append((node, level))
    return bad_cut


def check_wrong_type(tree):
    wrong_type = []
    for node in tree.traverse():
        if node.is_leaf():
            if node.TYPE == -2:
                wrong_type.append((node, node.TYPE))
        elif node.TYPE != -2:
            wrong_type.append((node, node.TYPE))
    return wrong_type

def check_children_order(tree):
    # This checks that the order of the children
    # follows the positions.
    # meaining: if children A is before B
    # it is lower or more to the left
    bad_order = []
    for node in tree.traverse():
        axis, dim = nd.get_orientation_from_cut(node, inv=True)
        axis_i, dim_i = nd.get_orientation_from_cut(node)
        children = node.get_children()
        if len(children) < 2:
            continue
        for ch1, ch2 in zip(children, children[1:]):
            # For now, if one of the two children has dist=0
            # I'll ignore it... but I should change this.
            if not getattr(ch1, dim) or not getattr(ch2, dim):
                continue
            correct = getattr(ch1, axis) + getattr(ch1, dim) ==\
                getattr(ch2, axis)
            correct &= getattr(ch1, axis_i) == getattr(ch2, axis_i)
            correct &= getattr(ch1, dim_i) == getattr(ch2, dim_i)
            if not correct:
                bad_order.append((ch1, ch2))
    return bad_order


def check_sizes(tree):
    bad_size = []
    for node in tree.traverse():
        if node.WIDTH < 0 or node.HEIGHT < 0:
            bad_size.append(node)
    return bad_size


def check_only_child(tree):
    only_1_child = []
    for n in tree.traverse():
        if len(n.get_children()) == 1 and n.CUT > 0:
            only_1_child.append(n)
    return only_1_child



cdef bint check_swap_size(dict nodes, int min_waste, bint insert=False, rotate=None):
    if rotate is None:
        rotate = []
    dims_i = {
        k: nd.get_dim_of_node(node, inv=True)
        for k, node in nodes.items()
    }
    dims = {
        k: nd.get_dim_of_node(node)
        for k, node in nodes.items()
    }
    # # TODO: delete this
    # if nodes[1].up is None or nodes[2].up is None:
    #     a=1

    wastes = {k: [
        w for w in nd.find_all_wastes(node.up)
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
            dims_i[k]: nd.get_size_without_waste(node, dims_i[k])
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

    result = geom.check_nodespace_in_space(node_space, space, insert, min_waste)
    # if result:
    #     log.debug('node with size {} will fit in space {}'.format(node_space, space))
    return result


cdef object check_swap_size_rotation(object node1, object node2, int min_waste, dict params, bint insert=False):
    # TODO: here we could return if needed reduction of parent to fit
    if node1.up == node2.up:
        log.debug('nodes are siblings, we do not check for space')
        return []
    cdef dict nodes
    cdef int rotation_tries
    cdef float rotation_probs[4]
    cdef bint try_rotation
    rotation_tries = params['rotation_tries']
    try_rotation = params['try_rotation']
    rotation_probs = params['rotation_probs']

#    cdef int rotations[4][1]
#    cdef int rotation[1]
    rotations = [[]]
    nodes = {1: node1, 2: node2}

    if try_rotation:
        rotations_av = [[], [1], [2], [1, 2]]
#        print(rotation_probs)
        rotations = np.random.choice(a=rotations_av, p=rotation_probs, size=rotation_tries, replace=False)
    for rotation in rotations:
        if check_swap_size(nodes, min_waste=min_waste, insert=insert, rotate=rotation):
            return rotation
    return None


def draw_leaf(ax1, leaf, stack, sequence, colors, fontsize):
    if leaf.TYPE in stack:
        color = colors[stack[leaf.TYPE] % len(colors)]
        edge_color = 'black'
    else:
        color = 'white'
        edge_color = 'black'
    ax1.add_patch(
        patches.Rectangle(
            (leaf.X, leaf.Y),  # (x,y)
            leaf.WIDTH,  # width
            leaf.HEIGHT,  # height
            facecolor=color,
            edgecolor=edge_color,
            linewidth=3
        )
    )
    more_info = ''
    if leaf.TYPE >= 0:
        parent = leaf.up
        if parent is None:
            parent = ""
        else:
            parent = parent.name
        more_info = "\nstack={}\npos={}\ntype={}\nparent={}".format(
            stack.get(leaf.TYPE, ''),
            sequence.get(leaf.TYPE, ''),
            leaf.TYPE,
            parent
        )
    ax1.text(leaf.X + leaf.WIDTH / 2, leaf.Y + leaf.HEIGHT / 2,
             '{} x {}{}\nnode={}'.
             format(leaf.WIDTH,
                    leaf.HEIGHT,
                    more_info,
                    leaf.name),
             horizontalalignment='center',
             verticalalignment='center',
             fontsize=fontsize)


def draw_defect(ax1, defect):
    ax1.axhline(y=defect['Y'], color="red", ls='dashed')
    ax1.axvline(x=defect['X'], color="red", ls='dashed')
    ax1.add_patch(
        patches.Circle(
            (defect['X'], defect['Y']),  # (x,y)
            radius=20,
            # defect['WIDTH'],  # width
            # defect['HEIGHT'],  # height
            color='red',
        )
    )

def graph_solution(node, path="", name="rect", show=False, dpi=50, fontsize=30, width=6000, height=3210):
    node = node.get_tree_root()
    # batch_data = self.get_batch()
    # stack = batch_data.get_property('STACK')
    # sequence = batch_data.get_property('SEQUENCE')
    sequence = {}
    stack = {k: 1 for k in range(1000)}
    colors = pal.colorbrewer.qualitative.Set3_5.hex_colors
    # pieces_by_type = self.get_pieces_by_type(by_plate=True, solution=solution)
    # if pos is not None:
    #     pieces_by_type = pieces_by_type.filter([pos])
    fig1 = plt.figure(figsize=(width / 100, height / 100))
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.set_xlim([0, width])
    ax1.set_ylim([0, height])
    ax1.tick_params(axis='both', which='major', labelsize=50)
    for leaf in node.iter_leaves():
        # graph items and wastes
        draw_leaf(ax1, leaf, stack, sequence, colors, fontsize)
    # graph defects
    for defect in nd.get_defects_from_plate(node):
        draw_defect(ax1, defect)
    fig_path = os.path.join(path, '{}_{}.png'.format(name, node.PLATE_ID))
    fig1.savefig(fig_path, dpi=dpi, bbox_inches='tight')
    if not show:
        plt.close(fig1)

cpdef bint check_assumptions_swap(object node1, object node2, bint insert):
    siblings = node1.up == node2.up
    if siblings and insert:
        return False
    if nd.is_waste(node1) and nd.is_waste(node2):
        return False
    if nd.is_waste(node1) and insert:
        return False
    if node1.up == node2 or node1 == node2.up:
        return False
    if node2.CUT >= 4 and insert:
        return False
    # for now, we do not allow swapping between different levels
    # or inserting a node from a higher to a lower level
    if node1.CUT == 0 or node2.CUT == 0:
        return False
    # nodes cannot be ancestors of themselves. Too weird
    # if node2 in node1.get_ancestors() or node1 in node2.get_ancestors():
    #     return False
    # if node1.CUT != node2.CUT and not insert:
    #     return False
    # if node1.CUT < node2.CUT:
    #     return False
    return True
