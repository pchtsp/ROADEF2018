import package.nodes as nd
import package.nodes_checks as nc
import package.geometry as geom
import logging as log
import numpy as np
import random as rn
import copy
import package.superdict as sd


def insert_node_inside_node_traverse(node1, node_start, min_waste, params):
    # we want to insert node1 at the first available space in node_start's tree
    # but never before node_start.
    # If i just want to traverse the whole tree, just need to put node1= root
    def is_leaf_fn(node2):
        return not \
            geom.plate_inside_plate(
                nd.node_to_plate(node1),
                nd.node_to_plate(node2),
                turn=True
            ) or node2.CUT >= 3

    for node2 in nd.post_traverse_from_node(node_start, is_leaf_fn=is_leaf_fn):
        if node2.CUT >= 4:
            continue
        node1.CUT = node2.CUT
        if nd.is_waste(node2):
            inserted_nodes = try_change_node_simple(node=node1, candidates=[node2],
                                                    min_waste=min_waste, params=params, insert=True)
            if inserted_nodes:
                return inserted_nodes
            continue
        # If I failed inserting in the children or if node2 is an item:
        # I try to insert next to the node2
        next_sibling = nd.get_next_sibling(node2)
        if next_sibling is None:
            continue
        inserted_nodes = try_change_node_simple(node=node1, candidates=[next_sibling],
                                                min_waste=min_waste, params=params, insert=True)
        if inserted_nodes:
            return inserted_nodes
    return False


def try_change_node_simple(node, candidates, insert, min_waste, params):
    good_candidates = {}
    rotation = {}
    weights = params.get('weights', None)
    debug = params.get('debug', False)
    assert weights is not None, 'weights argument cannot be empty or None!'
    for candidate in candidates:
        if not nc.check_assumptions_swap(node, candidate, insert):
            continue
        result = nc.check_swap_size_rotation(node, candidate, insert=insert,
                                          min_waste=min_waste, params=params)
        if result is None:
            continue
        rotation[candidate] = result
        good_candidates[candidate] = 0
    if len(good_candidates) == 0:
        return False
    candidates_prob = sd.SuperDict({k: v for k, v in good_candidates.items()}).to_weights()
    node2 = np.random.choice(a=candidates_prob.keys_l(), size=1, p=candidates_prob.values_l())[0]
    rot = rotation[node2]
    value, wastes_to_edit = check_swap_nodes_defect(node, node2, min_waste, insert=insert, rotation=rot)
    inserted_nodes = swap_nodes_same_level(node, node2, insert=insert, rotation=rot,
                                           debug=debug, min_waste=min_waste, wastes_to_edit=wastes_to_edit)
    if debug:
        checks = nc.check_consistency(node2.get_tree_root())
        if len(checks) > 0:
            a = 1
    return inserted_nodes


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
    ordered_nodes = [nd.item_to_node(v) for v in values]
    for node in ordered_nodes:
        if node.WIDTH > node.HEIGHT:
            nd.rotate_node(node)
        if node.HEIGHT > plate_H:
            nd.rotate_node(node)
    dummy_tree = nd.create_dummy_tree(ordered_nodes, id=-1)
    tree_id = 0
    tree = nd.create_plate(width=plate_W, height=plate_H, id=tree_id, defects=defects.get(tree_id, []))
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

    for node in ordered_nodes:
        item_id = node.TYPE
        inserted_nodes = False
        t_start = 1
        # We first search the tree where the previous node in the sequence was inserted
        # only starting from the position of the previous node
        if item_id in item_prec:
            node2 = item_node[item_prec[item_id]]
            inserted_nodes = insert_node_inside_node_traverse(node, node2, min_waste=min_waste, params=params)
            if inserted_nodes:
                item_node[item_id] = nd.get_node_by_type(inserted_nodes[1], item_id)
                continue
            t_start = node2.PLATE_ID + 2
        #     TODO: check this in partial solutions
        # The first tree is our dummy tree so we do not want to use it.
        for tree in trees[t_start:]:
            inserted_nodes = insert_node_inside_node_traverse(node, tree, min_waste=min_waste, params=params)
            if inserted_nodes:
                break
        if inserted_nodes:
            item_node[item_id] = nd.get_node_by_type(inserted_nodes[1], item_id)
            continue
        tree_id = len(trees) - 1
        # If we arrive to the limit, it means we lost.
        # because we are about to create another tree.
        if limit_trees and tree_id == limit_trees:
            return None
        tree = nd.create_plate(width=plate_W, height=plate_H, id=tree_id, defects=defects.get(tree_id, []))
        trees.append(tree)
        inserted_nodes = insert_node_inside_node_traverse(node, tree, min_waste=min_waste, params=params)
        # TODO: warning, in the future this could be possible due to defects checking
        assert inserted_nodes, "node {} apparently doesn't fit in a blank new tree".format(node.name)
        item_node[item_id] = nd.get_node_by_type(inserted_nodes[1], item_id)

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
        node1, node2 = nd.order_nodes(node1, node2)
        nodes = nd.get_nodes_between_nodes_in_tree(node1=node1, node2=node2)
        return node1, node2, nodes

    if plate1 > plate2:
        node1, node2 = node2, node1
        plate1, plate2 = plate2, plate1
    # if not in the same plate: i have three parts:
    # the rest of node1's plate:
    nodes1 = nd.get_nodes_between_nodes_in_tree(node1=node1)
    # the beginning of node2's plate:
    nodes2 = nd.get_nodes_between_nodes_in_tree(node2=node2)
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
    :param insert: type of swap can be insert (true) or swap (false)
    :return: balance of violations. Bigger is better.
    """
    # get all leaves in node1 and node2
    nodes = {1: node1, 2: node2}
    moved_items = {k: set(n.TYPE for n in nd.get_node_leaves(v)) for k, v in nodes.items()}
    # get all leaves between the two nodes
    first_node, second_node, neighbors = get_nodes_between_nodes(node1, node2, solution=solution)
    first_i, second_i = 1, 2
    if first_node != node1:
        first_i, second_i = 2, 1
    # changed: before we used nodes, now we use the item codes
    neighbor_items = set(leaf.TYPE for node in neighbors for leaf in nd.get_node_leaves(node))
    crossings = {k: {'items_after': set(), 'items_before': set()} for k in nodes}
    # neighbors between nodes are almost the same.
    # The sole difference is that the second node arrives *before* the first node
    neighbor_items_k = {1: neighbor_items.copy(), 2: neighbor_items}
    neighbor_items_k[second_i] |= set(moved_items[first_i])
    nodes_iter = [1]
    if not insert:
        nodes_iter.append(2)
    # items_after means items that are 'later' in the sequence.
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


def get_swap_node_changes(nodes, min_waste, insert, rotation, **waste_find_params):
    to_swap = [1]
    if not insert:
        to_swap.append(2)
    parents = {k: v.up for k, v in nodes.items()}
    inv_k = {1: 2, 2: 1}
    positions = {k: nd.get_node_pos(v) for k, v in nodes.items()}
    dims = {}
    axiss = {}
    dims_i = {}
    axiss_i = {}
    for k, node in nodes.items():
        axiss[k], dims[k] = nd.get_orientation_from_cut(node)
        axiss_i[k], dims_i[k] = nd.get_orientation_from_cut(node, inv=True)
    siblings = parents[1] == parents[2]
    if siblings:
        # if they're siblings: I just get the nodes between them
        first_node, second_node = 1, 2
        if positions[1] > positions[2]:
            first_node, second_node = 2, 1
        neighbors = {
            first_node: range(positions[first_node] + 1, positions[second_node]),
            second_node: []
        }
    else:
        neighbors = {k: range(positions[k] + 1, len(parents[k].children)) for k, v in nodes.items()}

    _dims = dict(dims)
    # if rotation then in node the dimensions are different:
    if 1 in rotation:
        _dims[2] = dims_i[2]

    move_neighbors = {k: getattr(nodes[1], _dims[k]) for k in nodes}
    dif_nodes = {k: {d: getattr(nodes[ik], d) - getattr(nodes[k], d) for d in ['X', 'Y']}
                 for k, ik in inv_k.items()}

    if insert:  # we do not swap node 2, it moves to let node1 in.
        dif_nodes[2] = {axiss[2]: move_neighbors[2], axiss_i[2]: 0}

    else: # complete swap
        # to get the movement for neighbors we need to compare the diff among
        # each node's dimension.
        _dims = dict(dims)
        if 2 in rotation:
            _dims[1] = dims_i[1]
        move_neighbors = {k: v - getattr(nodes[2], _dims[k]) for k, v in move_neighbors.items()}

    # I fix the move_neighbors sense problem for node1:
    move_neighbors[1] *= - 1

    # the first node is going to be moved at slightly different place than where the second node was.
    # because of the difference in dimensions.
    if not insert and siblings:
        dif_nodes[first_node][axiss[first_node]] += move_neighbors[first_node]

    # this I need to make it taking into account that I remove auxiliary wastes
    # when extracting nodes from their environment
    # need to be sure of the diff on levels to use the good dim apparently.
    dif_remove_waste = {k: 0 for k in nodes}
    if (nodes[1].CUT - nodes[2].CUT) % 2:

        valid_v = [v for v in nodes if v not in rotation and (v == 1 or not insert)]

        wastes_node = {
            k: nd.find_waste(nodes[v], child=True) if v in valid_v else None
            for k, v in inv_k.items()
        }

        dif_remove_waste = {
            k: getattr(w, dims[k]) if w is not None else 0
            # k: 0
            for k, w in wastes_node.items()
        }
    else:
        for v in rotation:
            waste = nd.find_waste(nodes[v], child=True)
            if waste is not None:
                k = inv_k[v]
                dif_remove_waste[k] = getattr(waste, dims_i[k])

    # now i need to make a decision about individual nodes' dif_neighbors.
    # because I need to reduce and create some waste(s) in between.
    # only if they are not siblings.
    if siblings:
        change_parent = {1: - sum(move_neighbors.values()), 2: 0}
    else:
        change_parent = {k: -v + dif_remove_waste[k] for k, v in move_neighbors.items()}

    dif_per_sibling = {k: {ch: 0 for ch, _ in enumerate(v.children)} for k, v in parents.items()}
    wastes_mods = {}
    resize_node = {k: {} for k in nodes}
    for k, v in nodes.items():
        if not change_parent[k]:
            continue
        # if the other node is moving and is a waste, I can count it:
        ik = inv_k[k]
        other_nodes = None
        if ik in to_swap and nd.is_waste(nodes[ik]):
            other_nodes = [(positions[k], nodes[ik])]
        filt_nodes = None
        if k in to_swap and nd.is_waste(nodes[k]):
            filt_nodes = [(positions[k], nodes[k])]
        wastes_mods[k] = \
            search_wastes_to_repair_node(
                node=parents[k], min_waste=min_waste, change=change_parent[k],
                add_pos_wastes=other_nodes, ignore_wastes=filt_nodes,
                **waste_find_params
            )
        if wastes_mods[k] is None:  # infeasible waste removal!
            return {}, {k: None}
        # iterate over wastes and edit the pos_dif value one way or the other.
        # if I added an optional inserting waste... I need to count it!
        for waste_pos, change in wastes_mods[k]:
            # if waste is what we're inserting,
            # we're inserting it in the current position of the other node
            # this is already taken into account in the search_wastes method.

            # if the waste is not at the end:
            if waste_pos <= len(nodes[k].get_sisters()):
                resize_node[k][waste_pos] = change
            for num in range(waste_pos + 1, len(parents[k].children)):
                dif_per_sibling[k][num] += change

    # iterate over nodes to pass the move_neighbors parameter based on the actual swap
    for k, v in nodes.items():
        for num in neighbors[k]:
            dif_per_sibling[k][num] += move_neighbors[k]

    # this is the amount we need to move each affected neighbor.
    nodes_changes = {k: {ch: [{axiss[k]: change, axiss_i[k]: 0}, {axiss[k]: change, axiss_i[k]: 0}]
                         for ch, change in v.items() if change and ch != positions[k]}
                     for k, v in dif_per_sibling.items()}

    # we resize the second dimension of the wastes we have edited.
    for k, resizes in resize_node.items():
        for ch, change in resizes.items():
            if ch == positions[k] and k in to_swap:
                continue
            if ch not in nodes_changes[k]:
                nodes_changes[k][ch] = [{'X': 0, 'Y': 0}, {'X': 0, 'Y': 0}]
            nodes_changes[k][ch][1][axiss[k]] += change

    # Before using dif_nodes, we need to update it
    # with a possible modification based on dif_per_sibling
    # (modified wastes, for example)
    for k in to_swap:
        ik = inv_k[k]
        _pos = positions[ik]
        _axis = axiss[ik]
        dif_nodes[k][_axis] += dif_per_sibling[ik][_pos]

    # I add the modifications to the swapped nodes
    for k, v in dif_nodes.items():
        ch = positions[k]
        if ch not in nodes_changes[k]:
            nodes_changes[k][ch] = [{'X': 0, 'Y': 0}, {'X': 0, 'Y': 0}]
        for pos in range(2):
            for a, change in v.items():
                nodes_changes[k][ch][pos][a] += change

    return nodes_changes, wastes_mods


def get_swap_squares(nodes, nodes_changes, insert, rotation):

    parents = {k: v.up for k, v in nodes.items()}
    plates = {k: v.PLATE_ID for k, v in nodes.items()}
    inv_k = {1: 2, 2: 1}
    positions = {k: nd.get_node_pos(v) for k, v in nodes.items()}
    to_swap = [1]
    if not insert:
        to_swap.append(2)

    nodes_sq = {k: [] for k in nodes}
    # we get the squares plus the modifications of all the items
    for k, change_neighbors in nodes_changes.items():
        # nodes_sq is a list of a tree tuple.
        # the tuple means: square, modifications, 1 if part of the two swapped nodes
        children = parents[k].children
        nodes_sq[k] += [(nd.node_to_square(item), v, pos == positions[k])
                        for pos, v in change_neighbors.items()
                        for item in nd.get_node_leaves(children[pos])]

    before, after = 0, 1
    # here we edit the squares we created in (1) and (2)
    # squares is a list of two dictionaries.
    # We have for 'before' and 'after' the nodes affected indexed by the plate they belong to.
    # TODO: change this to numpy by summing quantities as arrays
    squares = [{plates[k]: [] for k in nodes} for pos in [before, after]]
    for k, squares_changes in nodes_sq.items():
        ref_pos = {d: getattr(nodes[k], d) for d in ['X', 'Y']}
        # here, depending on the node, I should store the before and after plate.
        before_plate = plates[k]
        after_plate = before_plate
        for (sq, change, swaped_nodes) in squares_changes:
            squares[before][before_plate].append(sq)  # we do not edit it in before
            _sq = copy.deepcopy(sq)
            if swaped_nodes and k in to_swap:
                after_plate = plates[inv_k[k]]
                # if rotation, we rotate the node before moving it.
                if k in rotation:
                    _sq = geom.rotate_square(_sq, ref_pos)
            for n in range(2):  # for each corner of the square
                for a in ['X', 'Y']:
                    _sq[n][a] += change[n][a]
            squares[after][after_plate].append(_sq)  # we do edit it in after

    return squares


def check_swap_nodes_defect(node1, node2, min_waste, insert=False, rotation=None, order_wastes=None):
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

    if rotation is None:
        rotation = []
    if node1 == node2:
        return 0, None
    if node1.up is None or node2.up is None:
        return -10000, None

    nodes = {1: node1, 2: node2}
    to_swap = [1]
    if not insert:
        to_swap.append(2)

    defects = {k: nd.get_defects_from_plate(v) for k, v in nodes.items()}
    # if there's no defects to check: why bother??
    if not np.any(len(r) for r in defects.values()):
        return 0, None

    # order_wastes = None
    if order_wastes is None and rn.random() > 0.5:
        order_wastes = lambda x: rn.random()

    # candidates_prob = sd.SuperDict({k: v[0] for k, v in candidates_eval.items()}).to_weights()
    # node2 = np.random.choice(a=candidates_prob.keys_l(), size=1, p=candidates_prob.values_l())[0]

    nodes_changes, wastes_mods = get_swap_node_changes(nodes, min_waste, insert, rotation, order_wastes=order_wastes)
    if np.any([n is None for n in wastes_mods.values()]):  # infeasible waste removal!
        return - 10000, None
    squares = get_swap_squares(nodes, nodes_changes, insert, rotation)

    # finally: the defects to check
    defects_sq = {v.PLATE_ID: [geom.defect_to_square(d) for d in defects[k]] for k, v in nodes.items()}

    # here I count the number of defects that collide with squares. Before and now.
    # we want the following structure:

    # TODO: test if sum is faster than using for with break.
    defects_found = [0 for pos in squares]
    for pos, sqs in enumerate(squares):  # for (before) => after
        defects_found[pos] = \
        np.sum(geom.square_intersects_square(d, sq)
               for plate, sq_list in sqs.items()
               for d in defects_sq[plate]
               for sq in sq_list)

        # for plate, sq_list in squares[pos].items():  # for each node
        #     for d in defects_sq[plate]:  # for each defect
        #         for sq in sq_list:  # for each neighbor
        #             if geom.square_intersects_square(d, sq):
        #                 defects_found[pos].append((d, sq))
        #                 # if it's inside some node, it's not in the rest:
        #                 break

    # as the rest of checks: the bigger the better.
    return defects_found[0] - defects_found[1], wastes_mods


def check_swap_space(node1, node2, global_params, insert=False):
    # (dif density) * (dif position)
    nodes = {1: node1, 2: node2}
    node_density = {
        n: nd.get_item_density_node(node) for n, node in nodes.items()
    }

    if insert:
        node_density[2] = 0

    cost = {
        n: nd.get_node_position_cost_unit(node, global_params['widthPlates'])
        for n, node in nodes.items()
    }
    # we divide it over cost[1] to scale it.
    # the bigger the better
    gains = (node_density[1] - node_density[2]) * (cost[1] - cost[2]) / cost[1]
    return gains


def evaluate_swap(weights, solution, precedence, precedence_inv, global_params, rotation, **kwargs):
    # the bigger the better
    components = {
        'space': check_swap_space(global_params=global_params, **kwargs)
        ,'seq': check_swap_nodes_seq(solution=solution, precedence=precedence,
                                     precedence_inv=precedence_inv, **kwargs)
    }
    defects = check_swap_nodes_defect(**kwargs, min_waste=global_params['minWaste'], rotation=rotation)
    components['defects'] = defects[0]
    gains = {k: v * weights[k] for k, v in components.items()}
    return sum(gains.values()), defects[1]


def check_swap_two_nodes(nodes, args_check_size, args_evaluate, evaluate, params):
    """

    :param nodes:
    :param args_check_size:
    :param args_evaluate:
    :param evaluate: evaluate or not
    :param params:
    :return: a tuple of size 3: (economical balance, defects to resize, rotations to make)
    """
    tolerance = params['tolerance']
    args_check_size = {**args_check_size, **nodes}
    rotation = nc.check_swap_size_rotation(**args_check_size)
    if rotation is None:
        return None
    args_evaluate = {**args_evaluate, **nodes}
    if not evaluate:
        return 0
    balance = evaluate_swap(**args_evaluate, rotation=rotation)
    if tolerance is not None and balance[0] <= tolerance:
        return None
    return balance[0], balance[1], rotation


def swap_nodes_same_level(node1, node2, min_waste, wastes_to_edit, insert=False, rotation=None, debug=False):
    if rotation is None:
        rotation = []
    nodes = {1: node1, 2: node2}
    other_node = {1: 2, 2: 1}
    parents = {k: node.up for k, node in nodes.items()}
    parent1 = parents[1]
    parent2 = parents[2]
    positions = {k: nd.get_node_pos(node) for k, node in nodes.items()}
    siblings = parents[1] == parents[2]

    recalculate = False
    nodes_to_move = []
    if debug:
        pass
        # self.draw(node1.PLATE_ID, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
        # self.draw(node2.PLATE_ID, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
    if node1.up != parent2 or positions[1] != positions[2]:
        nodes_to_move.append(1)
    if not insert and (node2.up != parent1 or positions[1] + 1 != positions[2]):
        nodes_to_move.append(2)

    n_inserted = {k: 0 for k in nodes}
    for k in nodes_to_move:
        node = nodes[k]
        other_k = other_node[k]
        destination = parents[other_k]
        ch_pos_dest = positions[other_k]
        _, dest_dim = nd.get_orientation_from_cut(destination)
        # 1: take out children waste
        node = nd.extract_node_from_position(node)
        # 1.5: collapse if only child
        node = nd.delete_only_child(node, check_parent=False)
        # 2: rotate
        if k in rotation:
            nd.rotate_node(node)
        # 3, 4: insert+child
        node = insert_node_at_position(node, destination, ch_pos_dest)

        # 5: if necessary, we open the node to its children
        inserted_nodes = nd.collapse_node(node)
        n_inserted[k] = len(inserted_nodes)
        for _node in inserted_nodes:
            for ch in _node.children:
                nd.collapse_node(ch)

        # 6: now we check if we need to create a waste children on the node:
        for _node in inserted_nodes:
            dest_size = getattr(destination, dest_dim)
            node_size = getattr(_node, dest_dim)
            status, recalculate = nd.add_child_waste(node=_node, child_size=dest_size-node_size)


        nodes[k] = node

    # 7: we need to update the waste at both sides
    # but only if they are not siblings
    if siblings:
        return nodes

    # actually, waste_mods need to change their position
    # if they are to the right of the inserted node
    # the change depends on the number of inserted nodes
    # we put a max cap on the number of nodes to take out
    _e_pos = {k: n_inserted[v] - min(n_inserted[k], 1) for k, v in other_node.items()}

    # if insert:
    #     _e_pos[2]
    for k, v in wastes_to_edit.items():
        if v is not None and _e_pos[k]:
            wastes_to_edit[k] = [(tup[0] + _e_pos[k], tup[1]) if tup[0] >= positions[k] else tup for tup in v]
    for k, parent in parents.items():
        wastes = None
        if k in wastes_to_edit:
            wastes = wastes_to_edit[k]
        repair_dim_node(parent, min_waste, wastes)

    return nodes
    # return recalculate


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
    axis_i, dim_i = nd.get_orientation_from_cut(node, inv=True)
    axis, dim = nd.get_orientation_from_cut(node)
    ref = {axis: dim, axis_i: dim_i}
    # sib_axis, sib_dim = nd.get_orientation_from_cut(destination.children[position])
    # dest_axis, dest_dim = nd.get_orientation_from_cut(destination)
    dest_axis_i, dest_dim_i = nd.get_orientation_from_cut(destination, inv=True)

    if destination.children:
        if position < len(destination.children):
            # we get the destination position and then make space:
            axis_dest = {a: getattr(destination.children[position], a) for a in ref}
            # because siblings could have different level than my node, the movement
            # needs to be according to the siblings dimensions
            # which are the opposite of the destination (parent)
            for sib in destination.children[position:]:
                nd.mod_feature_node(node=sib,
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
        node = nd.duplicate_node_as_its_parent(node)

    # we make the move:
    nd.change_feature(node, 'PLATE_ID', destination.PLATE_ID)
    dist = {a: axis_dest[a] - getattr(node, a) for a in ref}
    for k, v in dist.items():
        if v:
           nd.mod_feature_node(node=node, quantity=v, feature=k)

    # In case we're moving nodes from different levels, we need to update the CUT:
    cut_change = destination.CUT + 1 - node.CUT
    if cut_change:
        nd.mod_feature_node(node, feature='CUT', quantity=cut_change)

    log.debug('We insert node {} into destination {} in position {}'.
              format(node.name, destination.name, position))
    destination.add_child(node)

    # 3. update parents order:
    nd.order_children(destination)
    return node


def search_wastes_to_repair_node(node, min_waste, change, add_pos_wastes=None, ignore_wastes=None,
                                 add_at_end=False, order_wastes=None):
    # It has to do with inserting two nodes without notice.
    """
    if the position we get is a waste we edit it.
    if the position is a non waste, we create one there.
    if the position is equal to the length, we create one at the end.
    :param node:
    :param min_waste:
    :additional_wastes: optional list of wastes (usually the node we're also inserting when swapping...)
        the format for this list is tuples of [(pos, waste)]
    :return:  [(position, increase), ...]
    """
    axis_i, dim_i = nd.get_orientation_from_cut(node, inv=True)
    if order_wastes is None:
        order_wastes = lambda x: getattr(x, dim_i)
    # if surplus is negative: we need to add a waste, it's easier
    # if surplus is positive: we start deleting wastes.
    # before creating one, will try to increase any waste.
    wastes = nd.find_all_wastes(node)
    if ignore_wastes is not None:
        _, less_wastes = zip(*ignore_wastes)
        wastes = [w for w in wastes if w not in less_wastes]
    positions = additional_wastes = []
    if add_pos_wastes is not None and len(add_pos_wastes):
        positions, additional_wastes = zip(*add_pos_wastes)
    wastes.extend(additional_wastes)
    # we reverse it because we take them from the end first.
    wastes.sort(key=order_wastes, reverse=True)
    if change > 0:
        if len(wastes) and not add_at_end:
            waste = wastes.pop()
            return [(nd.get_node_pos(waste), change)]
        else:
            return [(len(node.children), change)]

    # we want the farthest at the end:
    # but we want to eliminate really small wastes before
    # wastes.sort(key=lambda x: (getattr(x, dim_i) < 20, getattr(x, axis_i)))

    # we want the smallest at the end:
    result = []
    remaining = -change
    comply_min_waste = True
    while wastes and remaining:
        waste = wastes.pop()
        size = getattr(waste, dim_i)
        quantity = size
        if remaining < size:
            waste_rem = size - remaining
            if 0 < waste_rem < min_waste and comply_min_waste:
                quantity = size - min_waste
            else:
                quantity = remaining
        if waste in additional_wastes:
            # if we're inserting the waste, we put the position it will be inserted in
            _pos = additional_wastes.index(waste)
            waste_pos = positions[_pos]
        else:
            waste_pos = nd.get_node_pos(waste)
        log.debug("waste {} with position {} has been selected for changing by {}".
                  format(waste.name, waste_pos, -quantity))
        # nd.draw(waste.up)
        result.append((waste_pos, -quantity))
        remaining -= quantity
    if remaining > 0:
        return None
        # assert remaining == 0, "repair_dim_node did not eliminate all waste. Left={}".format(remaining)
    return result


def repair_dim_node(node, min_waste, wastes_mods=None):
    axis_i, dim_i = nd.get_orientation_from_cut(node, inv=True)
    node_size = getattr(node, dim_i)
    change = nd.get_surplus_dim(node)
    children = node.get_children()
    if wastes_mods is None:
        wastes_mods = search_wastes_to_repair_node(node, min_waste, change)
    if wastes_mods is None:
        assert False, "repair_dim_node did not eliminate all waste."
    wastes_mods_node = [(children[node_pos], change) if node_pos < len(children) else (None, change)
                        for node_pos, change in wastes_mods]
    for ref_child, change in wastes_mods_node:
        if change < 0:
            # TODO: still needed
            if not nd.is_waste(ref_child):
                nd.draw(node)
            nd.resize_waste(ref_child, dim_i, change)
        elif ref_child is None:
            # this means add waste at the end:
            nd.add_child_waste(node, child_size=change, waste_pos=node_size-change, increase_node=False)
        elif nd.is_waste(ref_child):
            # this means increase waste:
            nd.resize_waste(ref_child, dim_i, change)
        else:
            # this means create waste at specific position :
            # we need the relative position of this children with respect to the parent
            node_axis = getattr(ref_child, axis_i) - getattr(node, axis_i)
            node_pos = nd.get_node_pos(ref_child)

            # we need to make room for the node:
            for sib in nd.get_next_children_iter(node, pos_start= node_pos):
                nd.mod_feature_node(node=sib, quantity=change, feature=axis_i)

            nd.add_child_waste(node, child_size=change, waste_pos=node_axis, increase_node=False)
    # After all the waste editions, it could be possible that we need to collapse the node:
    nd.delete_only_child(node, collapse_child=True)
    return True


def join_blanks_seq(node):
    """
    :param node:
    :return: nothing.
    """
    """
    1. search for a waste.
    2. search for waste neighbors to the left.
    3. search for waste neighbors to the right.
    4. join everything.
    """
    children = node.get_children()
    if not children:
        return False

    # This gets a list of lists of consecutive wastes:
    new_waste = True
    wastes = []
    new_wastes = []
    for ch in children:
        if not nd.is_waste(ch):
            new_waste = True
            if len(new_wastes) > 1:
                wastes.append(new_wastes)
            new_wastes = []
            continue
        if new_waste:
            new_wastes = [ch]
        else:
            new_wastes.append(ch)
        new_waste = False

    if len(new_wastes) > 1:
        wastes.append(new_wastes)
    # This iterates over each sequence of wastes and joins them
    for w_seq in wastes:
        while len(w_seq) >= 2:
            w_2 = w_seq.pop()
            w_1 = w_seq[-1]
            # for w_1, w_2 in zip(wastes, wastes[1:]):
            nd.join_neighbors(w_1, w_2)

    return True
