import package.solution as sol
import package.solution as sol
import package.params as pm
import copy
import package.superdict as sd
import package.tuplist as tl
import package.nodes as nd
import numpy as np
import random as rn
import math

# we could do something like...
# 1. find a candidate node to edit
# (good alternatives: sequence, defects)
# 2. find a node to exchange
# a node in the same level and with the same size (even if it's rotated)
# check: sequence and defects at both places.
#


class ImproveHeuristic(sol.Solution):

    def __init__(self, solution):
        self.trees = copy.deepcopy(solution.trees)
        self.input_data = copy.deepcopy(solution.input_data)
        self.order_all_children()
        self.clean_empty_cuts()
        self.join_blanks()
        # we store the best solution:
        self.update_best_solution()
        return

    def update_best_solution(self):
        self.best_solution = copy.deepcopy(self)

    def move_item_inside_node(self):
        # This should keep the cuts the same but
        # maybe put the "waste" outside the node
        defects = self.check_defects()
        for node, defect in defects:
            for sibling in node.get_sisters():
                if self.defects_in_node(sibling):
                    continue
                print('Found! Swapping {} and {}'.format(node.name, sibling.name))
                result = self.swap_nodes_same_level(node, sibling)
                break
        return

    def fill_defect_with_waste(self):
        # given a defect in a plate, try to put a waste
        # instead of an item
        pass

    def extract_node_from_position(self, node):
        # take node out from where it is (detach and everything)
        # update all the positions of siblings accordingly
        parent = node.up
        plate, ch_pos = nd.get_node_pos(node)
        axis, dim = nd.get_orientation_from_cut(node)
        for sib in parent.children[ch_pos+1:]:
            nd.move_node(node=sib, movement=-getattr(node, dim), axis=axis)
        node.detach()
        # In case there's any waste at the end: I want to trim it.
        nd.del_child_waste(node)
        return node

    def insert_node_at_position(self, node, destination, position):
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
        if position < len(destination.children):
            # we get the destination position and then make space:
            axis_dest = {a: getattr(destination.children[position], a) for a in ref}
            for sib in destination.children[position:]:
                nd.move_node(node=sib, movement=getattr(node, dim), axis=axis)
        else:
            # we're puting the node in a new, last place:
            # so we do not move anything.
            last_node = destination.children[-1]
            axis_dest = {a: getattr(last_node, a) for a in ref}
            axis_dest[axis] += getattr(last_node, dim)


        # we make the move:
        nd.change_feature(node, 'PLATE_ID', destination.PLATE_ID)
        dist = {a: axis_dest[a] - getattr(node, a) for a in ref}
        for k, v in dist.items():
            if not v:
                continue
            nd.move_node(node, v, k)

        # now we check if we need to create a waste children on the node:
        nd.add_child_waste(node, fill=getattr(destination, dim_i))
        destination.add_child(node)

        # 3. update parents order:
        nd.order_children(destination)

        return True

    def check_node_in_space(self, node_space, free_space, insert):
        """
        :param node_space: {1: {WIDTH: XX, HEIGHT: XX}, 2: {WIDTH: XX, HEIGHT: XX}}
        :param free_space: {1: {WIDTH: XX, HEIGHT: XX}, 2: {WIDTH: XX, HEIGHT: XX}}
        :return:
        """
        # if dimensions are too small, we can't do the change
        # in insert=True we only check node1 movement
        for d in ['HEIGHT', 'WIDTH']:
            if node_space[1][d] > free_space[2][d]:
                return False
            if not insert and node_space[2][d] > free_space[1][d]:
                return False
        return True

    def check_swap_size(self, node1, node2, insert=False, rotate=None):
        # ideally, there should be not reference to the tree here
            # so we can test nodes that are not part of a tree
        _, dim = nd.get_orientation_from_cut(node1)
        # TODO: if cut=4, there can be no waste
        axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)

        wastes = {1: nd.find_waste(node1), 2: nd.find_waste(node2)}
        for w in wastes.values():
            if w is None:
                # no waste in node2 or waste1. cannot do the change
                return False

        node_space = {
            1: {dim: getattr(node1, dim),
                  dim_i: nd.get_size_without_waste(node1, dim_i)},
            2: {dim: getattr(node2, dim),
                  dim_i: nd.get_size_without_waste(node2, dim_i)},
        }

        space = {}
        for pos, node in enumerate([node1, node2], start=1):
            space[pos] = {
                dim_i: getattr(node, dim_i),
                dim: getattr(wastes[pos], dim) + node_space[pos][dim]
            }
        # if not swapping, we have less space in node2
        if insert:
            space[2][dim] -= node_space[2][dim]

        if rotate is None:
            return self.check_node_in_space(node_space, space, insert)

        # rotate can be a list with the index of nodes to reverse.
        # this way, we can check different combinations of rotation
        for pos in rotate:
            node_space[pos][dim], node_space[pos][dim_i] = \
                node_space[pos][dim_i], node_space[pos][dim]

        return self.check_node_in_space(node_space, space, insert)

    def check_assumptions_swap(self, node1, node2):
        # for now, the only requisite is to have the same level.
        # Maybe this will be relaxed in the future.
        assert node1.CUT == node2.CUT, \
            'nodes {} and {} need to have the same level'.format(node1.name, node2.name)

    def check_swap_size_rotation(self, node1, node2, insert=False, try_rotation=False):
        rotation = []
        if node1.up != node2.up:
            if not self.check_swap_size(node1, node2, insert):
                if not try_rotation:
                    return None
                result = False
                rotations = [[1], [2], [1, 2]]
                # rotations = [[1]]
                for rotation in rotations:
                    result = self.check_swap_size(node1, node2, insert, rotate=rotation)
                    if result:
                        break
                if not result:
                    return None
        return rotation

    def swap_nodes_same_level(self, node1, node2, insert=False, rotation=None):
        if rotation is None:
            rotation = []
        axis, dim = nd.get_orientation_from_cut(node1)
        parent1 = node1.up
        parent2 = node2.up
        plate1, ch_pos1 = nd.get_node_pos(node1)
        plate2, ch_pos2 = nd.get_node_pos(node2)
        # since we're modifying the order while swapping
        # we need to take it into account before the swap
        # I think this should be included inside the
        # insert_nod_at_position function.
        if node1.up == node2.up:
            if ch_pos1 > ch_pos2:
                ch_pos1 += 1
            elif ch_pos1 < ch_pos2:
                ch_pos2 -= 1

        if node1.up != parent2 or ch_pos1 != ch_pos2:
            node1_trimmed = self.extract_node_from_position(node1)
            if 1 in rotation:
                nd.rotate_node(node1_trimmed)
            self.insert_node_at_position(node1_trimmed, parent2, ch_pos2)
        if not insert:
            if node2.up != parent1 or ch_pos1 + 1 != ch_pos2:
                node2_trimmed = self.extract_node_from_position(node2)
                if 2 in rotation:
                    nd.rotate_node(node2_trimmed)
                self.insert_node_at_position(node2_trimmed, parent1, ch_pos1)

        # we need to update the waste at both sides
        if parent1 == parent2:
            return True

        change = self.calculate_change_of_linear_waste(node1, node2, rotation, insert)
        nd.resize_node(parent1, dim, change[1])
        nd.resize_node(parent2, dim, change[2])
        return True

    def calculate_change_of_linear_waste(self, node1, node2, rotation, insert):
        axis, dim = nd.get_orientation_from_cut(node1)
        axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)
        nodes = {1: node1, 2: node2}
        present_size = {n: getattr(node, dim) for n, node in nodes.items()}
        previous_size = dict(present_size)

        for n in rotation:
            previous_size[n] = getattr(nodes[n], dim_i)

        if insert:
            previous_size[2] = present_size[2] = 0

        return {
            1: previous_size[1] - present_size[2],
            2: -present_size[1] + previous_size[2],
        }

    def calculate_change_of_area_waste(self, node1, node2, insert):
        # axis, dim = nd.get_orientation_from_cut(node1)
        axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)
        nodes = {1: node1, 2: node2}
        change_linear = {1: 0, 2: 0}
        if node1.up != node2.up:
            change_linear = self.calculate_change_of_linear_waste(node1, node2, [], insert)
        change_area = {n: v * getattr(nodes[n], dim_i) for n, v in change_linear.items()}

        # I have to add the smaller parts of waste that are also moved:
        waste = \
            {n:
                sum(n.HEIGHT * n.WIDTH
                    for n in nd.get_node_leaves(nodes[n], type_options=[-1, -3]))
             for n in nodes}
        if insert:
            waste[2] = 0

        add_waste = {
            1: - waste[1] + waste[2],
            2: - waste[2] + waste[1]
        }

        return {n: change_area[n] + add_waste[n] for n in nodes}

    def clean_empty_cuts(self):
        """
        An empty cut is a cut with a 0 distance in a dimension
        :return:
        """
        for tree in self.trees:
            for v in tree.traverse():
                children = v.get_children()
                if len(children) != 2:
                    continue
                if children[0].TYPE < 0 or \
                        (children[1].WIDTH > 0 and
                                 children[1].HEIGHT > 0):
                    continue
                v.TYPE = children[0].TYPE
                v.remove_child(children[0])
                v.remove_child(children[1])

    def join_neighbors(self, node1, node2):
        # this only makes sense if both
        # nodes are type=-1 (waste)
        parent = node1.up
        assert parent == node2.up, \
            '{} and {} are not siblings'.format(node1.name, node2.name)
        assert node1.TYPE == -1 and node2.TYPE == -1, \
            '{} and {} are not waste'.format(node1.name, node2.name)

        axis, dim = nd.get_orientation_from_cut(node1)
        if getattr(node1, axis) > getattr(node2, axis):
            node1, node2 = node2, node1
        node1pos = getattr(node1, axis)
        node2pos = getattr(node2, axis)
        assert node1pos + getattr(node1, dim) == node2pos, \
            '{} and {} are not neighbors'.format(node1, node2)
        new_size = getattr(node1, dim) + getattr(node2, dim)
        setattr(node1, dim, new_size)
        node2.detach()
        return True

    def join_blanks_tail(self, node):
        pos = len(node.children) - 1
        wastes = []
        while pos >= 0 and node.children[pos].TYPE in [-1, -3]:
            wastes.append(node.children[pos])
            pos -= 1
        if len(wastes) < 2:
            return False
        for w_1, w_2 in zip(wastes, wastes[1:]):
            self.join_neighbors(w_1, w_2)
        return True

    def join_blanks(self):
        for tree in self.trees:
            for v in tree.traverse('postorder'):
                children = v.children
                if not len(children):
                    continue
                candidates = [n for n in children if n.TYPE == -1]
                if len(candidates) < 2:
                    continue
                pos = len(children) - 1
                candidates_s = candidates[:]
                min_pos = len(children) - len(candidates)
                while len(candidates_s) and pos >= min_pos:
                    if children[pos] in candidates_s:
                        candidates_s.remove(children[pos])
                        pos -= 1
                        continue
                    c = candidates_s.pop(0)
                    self.swap_nodes_same_level(children[pos], c)
                candidates = children[min_pos:]
                while len(candidates) > 1:
                    self.join_neighbors(candidates[0], candidates[1])
                    candidates.pop(1)
                last_waste = candidates[0]
                if not last_waste.get_sisters():
                    last_waste.up.TYPE = -1
                    last_waste.detach()
        return True

    def check_swap_nodes_defect(self, node1, node2, insert=False):
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
        nodes = {1: node1, 2: node2}
        plate1, ch_pos1 = nd.get_node_pos(node1)
        plate2, ch_pos2 = nd.get_node_pos(node2)
        positions = {1: ch_pos1, 2: ch_pos2}
        plates = {1: plate1, 2: plate2}
        axis, dim = nd.get_orientation_from_cut(node1)
        axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)
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
                first_node: self.get_defects_per_plate(plates[1]).values(),
                second_node: []
            }
            # If a defect is to the left of the first node
            # or to the right of the second node: take it out.
            # TODO: This filtering could be done by searching the node where the defect is
            right = nd.filter_defects(nodes[first_node], defects[first_node])
            left = nd.filter_defects(nodes[second_node], defects[first_node], previous=False)
            defects[first_node] = [d for d in right if d in left]
            # defects[second_node] = nd.filter_defects(nodes[second_node], defects[second_node], previous=False)
        else:
            neighbors = {k: nodes[k].up.children[positions[k]+1:] for k in nodes}

            # If a defect is to the left / down of the node: take it out.
            defects = {k: self.get_defects_per_plate(plates[k]).values() for k in nodes}
            defects = {k: nd.filter_defects(nodes[k], defects[k]) for k in nodes}

        # if there's no defects to check: why bother??
        if not defects[1] and not defects[2]:
            return 0

        if insert:
            move_neighbors = getattr(nodes[1], dim)
            pos_dif = {
                1: {'X': nodes[2].X - nodes[1].X, 'Y': nodes[2].Y - nodes[1].Y},
                2: {axis: move_neighbors, axis_i: 0}
            }
        else:  # complete swap
            move_neighbors = getattr(nodes[1], dim) - getattr(nodes[2], dim)
            pos_dif = {
                1: {'X': nodes[2].X - nodes[1].X, 'Y': nodes[2].Y - nodes[1].Y},
                2: {'X': - nodes[2].X + nodes[1].X, 'Y': - nodes[2].Y + nodes[1].Y}
            }
            if siblings:
                    pos_dif[first_node][axis] -= move_neighbors

        # get squares
        # first of the nodes involved
        nodes_sq = {k: [(nd.node_to_square(p), pos_dif[k]) for p in nd.get_node_leaves(nodes[k])]
                    for k in nodes
                    }
        # the destination node needs to be included in the first one in case of siblings:
        if siblings:
            nodes_sq[first_node] += nodes_sq[second_node]

        # second of the nodes in between
        dif = {1: {axis: -move_neighbors, axis_i: 0}, 2: {axis: move_neighbors, axis_i: 0}}
        for k, _neighbors in neighbors.items():
            nodes_sq[k] += [(nd.node_to_square(p), dif[k]) for n in _neighbors for p in nd.get_node_leaves(n)]

        # finally of the defects to check
        defects_sq = {k: [self.defect_to_square(d) for d in defects[k]] for k in nodes}

        # here we edit the squares we created in (1) and (2)
        # squares is a list of two dictionaries.
        # We have for 'before' and 'after' the nodes affected indexed by 1 and 2.
        squares = [{k: [] for k in nodes_sq} for r in range(2)]
        for k, squares_changes in nodes_sq.items():
            for (sq, change) in squares_changes:
                _sq = copy.deepcopy(sq)
                squares[0][k].append(sq)
                for n in range(2):  # we change the position in the two nodes of the square
                    for a in [axis, axis_i]:
                        _sq[n][a] += change[a]
                squares[1][k].append(_sq)

        # here I count the number of defects that collide with squares. Before and now.
        defects_found = [[] for r in range(2)]
        for pos in range(2):  # for (before) => after
            for k, sq_list in squares[pos].items():  # for each node
                for d in defects_sq[k]:  # for each defect
                    for sq in sq_list:  # for each neighbor
                        if self.square_intersects_square(d, sq):
                            defects_found[pos].append(d)
                            # if it's inside some node, it's not in the rest:
                            break

        # as the rest of checks: the bigger the better.
        return len(defects_found[0]) - len(defects_found[1])

    def check_swap_space(self, node1, node2, insert=False):
        # 1. get waste in node1
        # 2. get waste in node2.
        # 3. get waste in between?? this is not very representative and costly to calculate.
        # 4. I'm not taking into account the actual positions of the wastes inside the node.
        nodes = {1: node1, 2: node2}
        w_change_area = self.calculate_change_of_area_waste(node1, node2, insert)

        values = {n: nd.get_node_position_cost_unit(nodes[n], self.get_param('widthPlates'))
                  for n in nodes
                  }
        gains = {n: values[n] * w_change_area[n] for n in nodes}
        return sum(gains.values())

    def check_swap_nodes_seq(self, node1, node2, insert=False):
        """
        checks if a change is beneficial in terms of sequence violations
        :param node1:
        :param node2:
        :param insert: type of swap can be insert or swap
        :return: balance of violations. Bigger is better.
        """
        precedence = self.get_previous_nodes()
        precedence_inv = precedence.list_reverse()
        # get all leaves in node1 and node2
        items1 = nd.get_node_leaves(node1)
        items2 = nd.get_node_leaves(node2)
        # get all leaves between the two nodes
        first_node, second_node, nodes = self.get_nodes_between_nodes(node1, node2)
        items = set(leaf for node in nodes for leaf in nd.get_node_leaves(node))
        negative = set()
        positive = set()
        # We'll assume node1 is the first node.
        # It's that not the case, we'll switch the math at the end
        # for each leaf in node1:
            # because node1 is "going forward":
            # if I find nodes that precede it: good
            # if I find nodes that follow it: bad
        for item in items1:
            negative |= items & set(precedence_inv.get(item, set()))
            positive |= items & set(precedence.get(item, set()))
        if insert:
            if node1 == second_node:
                negative, positive = positive, negative
            # we only care about the first node
            # because is the only one we move.
            return len(positive) - len(negative)
        # for each leaf in node2:
            # because node2 is "going back":
            # if I find nodes that precede it: bad
            # if I find nodes that follow it: good
        items |= set(items1)
        for item in items2:
            negative |= items & set(precedence.get(item, set()))
            positive |= items & set(precedence_inv.get(item, set()))
        # lastly, we need to take into account the items in each node.
        if node1 == second_node:
            negative, positive = positive, negative
        return len(positive) - len(negative)

    def evaluate_swap(self, weights=None, **kwargs):
        if weights is None:
            # weights = {'space': 1/10000, 'seq': 100000, 'defects': 1000}
            weights = {'space': 1 / 100000, 'seq': 100000, 'defects': 10000}
        components = {
            'space': self.check_swap_space(**kwargs)
            ,'seq': self.check_swap_nodes_seq(**kwargs)
            ,'defects': self.check_swap_nodes_defect(**kwargs)
        }
        gains = {k: v * weights[k] for k, v in components.items()}
        return sum(gains.values())

    def evaluate_solution(self, weights):
        components = {
            'space': - self.check_space_usage()
            ,'seq': len(self.check_sequence())
            ,'defects': len(self.check_defects())
        }
        gains = {k: v * weights[k] for k, v in components.items()}
        return sum(gains.values())

    @staticmethod
    def acceptance_probability(change_benefit, temperature):
        if change_benefit > 0:
            return 1.0
        else:
            return math.exp(change_benefit / temperature)

    def get_nodes_between_nodes(self, node1, node2):
        """
        :param node1:
        :param node2:
        :return: (n1, n2, list): the order of the nodes and a list of nodes in between.
        """
        plate1 = node1.PLATE_ID
        plate2 = node2.PLATE_ID

        if plate1 == plate2:
            # if they're in the same plate: I just get the nodes between them
            return self.get_nodes_between_nodes_in_tree(node1=node1, node2=node2)

        if plate1 > plate2:
            node1, node2 = node2, node1
            plate1, plate2 = plate2, plate1
        # if not in the same plate: i have three parts:
        # the rest of node1's plate:
        node1, _, nodes1 = self.get_nodes_between_nodes_in_tree(node1=node1)
        # the beginning of node2's plate:
        _, node2, nodes2 = self.get_nodes_between_nodes_in_tree(node2=node2)
        nodes = nodes1 + nodes2
        # nodes in intermediate plates:
        i = plate1 + 1
        while i < plate2:
            nodes += self.trees[i]
            i += 1
        return node1, node2, nodes

    def get_nodes_between_nodes_in_tree(self, node1=None, node2=None, only_order=False):
        """
        This procedure searches a tree for all nodes between node1 and node2.
        In case node1 is None: it should start in the first node
        If node2 is None: it should end in the last node
        :param node1:
        :param node2:
        :param only_order: we only care about which nodes comes before.
            Not the nodes in between.
        :return: (n1, n2, list): the order of the nodes and a list of nodes in between.
        """
        if node1 is None and node2 is None:
            raise ValueError("node1 and node2 cannot be None at the same time")
        if node1 is None:
            node1 = nd.get_descendant(node2.get_tree_root(), which='first')
        elif node2 is None:
            node2 = nd.get_descendant(node1.get_tree_root(), which='last')
        else:
            assert node1.get_tree_root() == node2.get_tree_root(), \
                "node {} does not share root with node {}".format(node1.name, node2.name)
        ancestor = node1.get_common_ancestor(node2)
        n1ancestors = node1.get_ancestors()
        n2ancestors = node2.get_ancestors()
        all_ancestors = set(n1ancestors) | set(n2ancestors)
        nodes = []
        first_node = None
        second_node = None

        def is_not_ancestor(node):
            return node not in all_ancestors

        for node in ancestor.iter_descendants(strategy='postorder', is_leaf_fn=is_not_ancestor):
            if first_node is None:
                if node not in [node1, node2]:
                    continue
                first_node, second_node = node1, node2
                if node == node2:
                    first_node, second_node = node2, node1
                if only_order:
                    break
                continue
            if node == second_node:
                break
            if is_not_ancestor(node):
                nodes.append(node)

        return first_node, second_node, nodes

    def order_all_children(self):
        for tree in self.trees:
            nd.order_children(tree)

    def cut_waste_with_defects(self):

        return True

    def try_change_node(self, node1, candidates, insert, temperature, tolerance=None, change_first=False, **kwargs):
        good_candidates = {}
        rotation = {}
        weights = weights=kwargs.get('weights', None)
        assert weights is not None, 'weights argument cannot be empty or None!'

        for node2 in candidates:
            siblings = node1.up == node2.up
            if siblings and insert:
                continue
            # if i'm swapping with a waste, it has to be a sister.
            if node2.TYPE in [-1, -3] and not insert and not siblings:
                continue
            if node1.TYPE in [-1, -3] and not siblings:
                continue
            # if insert_only=True, we insert node1 before node2 but we do not move node2
            self.check_assumptions_swap(node1, node2)
            result = self.check_swap_size_rotation(node1, node2, insert=insert,
                                                   try_rotation=kwargs.get('try_rotation', False))
            if result is None:
                continue
            balance = self.evaluate_swap(node1=node1, node2=node2, insert=insert, weights=weights)
            if tolerance is not None and balance <= tolerance:
                continue
            good_candidates[node2] = balance
            rotation[node2] = result
            if change_first:
                break
        if len(good_candidates) == 0:
            return False
        node2, balance = max(good_candidates.items(), key=lambda x: x[1])
        if self.acceptance_probability(balance, temperature=temperature) < rn.random():
            return False
        rot = rotation[node2]
        # print("I want to swap nodes {} and {} for a gain of {}".format(node1.name, node2.name, balance))
        change = 'Insert'
        if not insert:
            change = 'Swap'
        print('Found! {} between nodes {} and {} for a gain of {}'.
              format(change, node1.name, node2.name, round(balance)))
        self.swap_nodes_same_level(node1, node2, insert=insert, rotation=rot)
        new = self.evaluate_solution(weights)
        old = self.best_solution.evaluate_solution(weights)
        if new < old - 5000:  # only save if really different.
            print('Best solution updated to {}!'.format(round(new)))
            self.update_best_solution()
        return True

    def debug_nodes(self, node1, node2):
        for node in [node1, node2]:
            print("name={}\nPLATE_ID={}\nX={}\nY={}\nchildren={}".format(
                node.name, node.PLATE_ID, node.X, node.Y, [ch.name for ch in node.children])
            )
            print("")

    def change_level_by_seq2(self, level, max_iter=100, **kwargs):
        prec = self.check_sequence().to_dict(result_col=1)
        i = count = 0
        tolerance = None
        while count < max_iter and i < len(prec):
            # tolerance = 0
            # if rn.random() <= (max_iter - count)/max_iter * 0.5:
            #     tolerance = -30000
            count += 1
            # node = sorted([*prec], key=lambda x: x.name)[i]
            node = [*prec][i]
            i += 1
            node_level = nd.find_ancestor_level(node, level)
            if node_level is None:
                continue
            other_nodes = [nd.find_ancestor_level(p, level) for p in prec[node]]
            # other_node_levels = set([s for p in prec[node] for s in nd.find_ancestor_level(p, level)])
            candidates = [s for p in other_nodes if p is not None and p.up is not None
                          for s in p.up.get_children() if s != node_level]
            for _insert in [True, False]:
                # when dealing with siblings... we do not insert!
                if _insert:
                    candidates = [c for c in candidates if c not in node_level.get_sisters()]
                candidates = [c for c in candidates if c in node_level.get_sisters()]
                change = self.try_change_node(node1=node_level, candidates=candidates,
                                              tolerance=tolerance, insert=_insert, try_rotation=True,
                                              **kwargs)
                if change:
                    break
            if not change:
                continue
            # made a swap: recalculate
            seq = self.check_sequence()
            prec = seq.to_dict(result_col=1)

    def change_level(self, node, level, **kwargs):
        """
        General function that searches nodes in the same level of the node and try to swap them
        :param node:
        :param level: cut level
        :param kwargs: tolerance, temperature, ...
        :return: True if swap was succesful
        """
        node_level = nd.find_ancestor_level(node, level)
        if node_level is None:
            return False
        candidates = [ch for tree in self.trees for ch in tree.traverse(is_leaf_fn=lambda x: x.CUT == level)
                      if ch != node_level and ch.CUT == level]
        # always try insert first
        for _insert in [True, False]:
            change = self.try_change_node(node1=node_level, candidates=candidates, insert=_insert, **kwargs)
            if change:
                return True
        return False

    def change_level_by_seq(self, level, max_iter=100, **kwargs):
        rem = [n for tup in self.check_sequence() for n in tup]
        i = count = 0
        while count < max_iter and i < len(rem):
            count += 1
            node = rem[i]
            i += 1
            change = self.change_level(node, level, **kwargs)
            if not change:
                continue
            # made a swap: recalculate
            seq = self.check_sequence()
            rem = [n for tup in seq for n in tup]

    def collapse_to_left(self, level, **kwargs):
        wastes = [n for tree in self.trees
                  for n in nd.get_node_leaves(tree, type_options=[-1, -3])
                  if n.CUT == level]
        candidates = [ch for tree in self.trees for ch in tree.traverse() if ch.CUT == level and ch.TYPE not in [-1, -3]]
        candidates.sort(reverse=True, key=lambda x: (x.PLATE_ID * self.get_param('widthPlates') + x.X + x.Y) * x.HEIGHT * x.WIDTH)
        for c in candidates:
            w_not_sblings = [w for w in wastes if w.up != c.up]
            # w_sblings = [w for w in wastes if w.up == c.up and w != c]
            # if not siblings: I should be able to only insert
            change = self.try_change_node(node1=c, candidates=w_not_sblings, insert=True, **kwargs)
            # if siblings: I should be able to swap or insert
            change = self.try_change_node(node1=c, candidates=wastes, insert=False, **kwargs)

    def merge_wastes_tail(self):
        for tree in self.trees:
            for node in tree.traverse():
                if node.children:
                    self.join_blanks_tail(node)
        return

    def change_level_by_defects(self, level, max_iter=100, **kwargs):
        defects = self.check_defects()
        i = count = 0
        while count < max_iter and i < len(defects):
            count += 1
            defect = defects[i]
            node, actual_defect = defect
            i += 1
            change = self.change_level(node, level, **kwargs)
            if not change:
                continue
            # made a swap: recalculate
            defects = self.check_defects()
        return True

    def create_waste(self, node1, cut, **kwargs):
        # first, we split one waste in two.
        # then we make a swap to one of the nodes.
        parent = node1.up
        axis, dim = nd.get_orientation_from_cut(node1)
        axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)
        attributes = [axis, axis_i, dim, dim_i]
        size = getattr(node1, dim)
        assert node1.TYPE == -1, "node {} needs to be waste!".format(node1.name)
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
        nd.order_children(parent)
        candidates = [n for n in node2.get_sisters() if n.TYPE not in [-1, -3]]
        for node in nodes.values():
            change = self.try_change_node(node, candidates, insert=False, **kwargs)
            if change:
                return True
        return False

    def search_waste_cuts(self, **kwargs):
        defects = self.check_defects()
        for node, defect in defects:
            axis, dim = nd.get_orientation_from_cut(node)
            waste = nd.find_waste(node)
            if waste is None:
                continue
            cut = defect[axis] + defect[dim] + 1
            cut2 = self.get_param('widthPlates') - cut
            if cut > cut2:
                cut = cut2
            if getattr(waste, dim) <= cut:
                return False
            self.create_waste(waste, cut, **kwargs)
        return

    # Missing: function to move a little bit some plate by slicing wastes.

        # for defect in defects:
        #     node, actual_defect = defect
        #     node_level1 = nd.find_ancestor_level(node, 1)
        #     dist_x = actual_defect['X'] + actual_defect['WIDTH'] - node_level1.X
        #     dist_y = min(actual_defect['Y'], node_level1.HEIGHT - actual_defect['Y'])
        #     for sibling in node_level1.get_sisters():
        #         if self.defects_in_node(sibling):
        #             continue
        #         candidates = [n for n in sibling.children if n.TYPE == -1 and n.HEIGHT >= dist_y]
        #         if len(candidates) == 0:
        #             continue
        #         print('Found! Swapping {} and {}'.format(node_level1.name, sibling.name))
        #         result = self.swap_nodes_same_level(node_level1, sibling)
        #         break


if __name__ == "__main__":
    import pprint as pp
    # e = '201804271903/'  # A6 base case
    e = '201805020012/'  # A6 this one was optimised for sequence.
    # e = '201805090409/'  # A20
    path = pm.PATHS['experiments'] + e
    # path = pm.PATHS['results'] + 'multi1/A1/'
    solution = sol.Solution.from_io_files(path=path)

    self = ImproveHeuristic(solution)
    weights = {'space': 1 / 1000000, 'seq': 10000, 'defects': 10000}
    iterations = 100
    temp = 1000
    params = {'weights': weights, 'max_iter': iterations, 'temperature': temp}
    coolingRate = 0.007
    for x in range(100):
        for level in [1, 2]:
            self.change_level_by_seq2(level, **params)
            self.change_level_by_seq(level, **params)
            self.change_level_by_defects(level, **params)
            self.search_waste_cuts(**params)
            self.collapse_to_left(level, **params)
            self.merge_wastes_tail()
        temp *= 1 - coolingRate
        if temp < 1:
            break
        seq = self.check_sequence()
        defects = self.check_defects()
        print("TEMP IS: {}".format(temp))
        print("There's {} incorrect sequences".format(len(seq)))
        print("There's {} remaining defects".format(len(defects)))
        pass
        # self.move_item_inside_node()
        # for level in [1, 2]:
        #     self.collapse_to_left(1, weights=weights, temperature=temp)
        # self.graph_solution(path, name="edited", dpi=50)

    print(self.check_sequence())
    self.graph_solution(path, name="edited", dpi=50)


    # self.graph_solution(path, show=True, pos=3, dpi=30, fontsize=10)
    # heur.get_pieces_by_type(by_plate=True)
    # prev_nodes = self.get_previous_nodes()
    pass