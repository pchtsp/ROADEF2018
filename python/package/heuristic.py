import package.params as pm
import package.solution as sol
import copy
import package.superdict as sd
import package.tuplist as tl
import package.nodes as nd
import numpy as np
import random as rn
import math
import package.geometry as geom
import time
import logging as log
import multiprocessing as multi


class ImproveHeuristic(sol.Solution):

    def __init__(self, input_data, solution_data=None):
        if solution_data is None:
            solution_data = {}
        super().__init__(input_data, solution_data)
        self.debug = False
        self.input_data = copy.deepcopy(input_data)
        # we store the best solution:
        self.best_solution = []
        self.best_objective = 99999999
        self.last_objective = 99999999
        self.hist_objective = []
        self.type_node_dict = None
        self.previous_nodes = None
        self.next_nodes = None
        self.improved = 0
        self.accepted = 0
        self.evaluated = 0

    def update_best_solution(self, solution):
        self.best_solution = copy.deepcopy(solution)

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

    def clean_empty_cuts_2(self):
        """
        This finds leaves with a 0 dimension and eliminates it.
        If this is the only child of the parent, we make it the leaf.
        :return:
        """
        for tree in self.trees:
            for node in tree.iter_leaves():
                if node.WIDTH and node.HEIGHT:
                    continue
                parent = node.up
                if parent is None:
                    continue
                node.detach()
                nd.delete_only_child(parent)
        return True

    def join_neighbors(self, node1, node2):
        # this only makes sense if both
        # nodes are type=-1 (waste)
        if node1 == node2:
            return False
        parent = node1.up
        assert parent == node2.up, \
            '{} and {} are not siblings'.format(node1.name, node2.name)
        assert node1.TYPE in [-1, -3] and node2.TYPE in [-1, -3], \
            '{} and {} are not waste'.format(node1.name, node2.name)

        # this is okay because they are siblings:
        axis, dim = nd.get_orientation_from_cut(node1)
        node1pos = getattr(node1, axis)
        node2pos = getattr(node2, axis)
        if not (node1pos + getattr(node1, dim) == node2pos):
            # self.draw(node1.PLATE_ID)
            self.draw(node1.PLATE_ID, 'name','X', 'Y', 'WIDTH', 'HEIGHT')
            assert (node1pos + getattr(node1, dim) == node2pos), \
                '{} and {} are not neighbors'.format(node1.name, node2.name)
        new_size = getattr(node1, dim) + getattr(node2, dim)
        # we need to update the first node because is the one that comes first
        setattr(node1, dim, new_size)
        node2.detach()
        return True

    def join_blanks_seq(self, node):
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
            if ch.TYPE not in [-1, -3]:
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
                    nd.swap_nodes_same_level(children[pos], c, debug=self.debug, min_waste=self.get_param('minWaste'))
                    pos -= 1
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
        if node1.up is None or node2.up is None:
            return -10000
        nodes = {1: node1, 2: node2}
        plate1, ch_pos1 = nd.get_node_plate_pos(node1)
        plate2, ch_pos2 = nd.get_node_plate_pos(node2)
        positions = {1: ch_pos1, 2: ch_pos2}
        plates = {1: plate1, 2: plate2}
        dims = {}
        axiss = {}
        dims_i = {}
        axiss_i = {}
        for k, node in nodes.items():
            axiss[k], dims[k] = nd.get_orientation_from_cut(node)
            axiss_i[k], dims_i[k] = nd.get_orientation_from_cut(node, inv=True)
        # axis, dim = nd.get_orientation_from_cut(node1)
        # axis_i, dim_i = nd.get_orientation_from_cut(node1, inv=True)
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
                first_node: nd.get_defects(nodes[1]),
                second_node: []
            }
            # If a defect is to the left of the first node
            # or to the right of the second node: take it out.
            right = nd.filter_defects(nodes[first_node], defects[first_node])
            left = nd.filter_defects(nodes[second_node], defects[first_node], previous=False)
            defects[first_node] = [d for d in right if d in left]
            # defects[second_node] = nd.filter_defects(nodes[second_node], defects[second_node], previous=False)
        else:
            neighbors = {k: nodes[k].up.children[positions[k]+1:] for k in nodes}

            # If a defect is to the left / down of the node: take it out.
            defects = {k: nd.get_defects(nodes[k]) for k in nodes}
            defects = {k: nd.filter_defects(nodes[k], defects[k]) for k in nodes}

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
        nodes_sq = {k: [(nd.node_to_square(p), pos_dif[k]) for p in nd.get_node_leaves(nodes[k])]
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
            nodes_sq[k] += [(nd.node_to_square(p), dif[k]) for n in _neighbors for p in nd.get_node_leaves(n)]

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

    def get_stacks_from_nodes(self, nodes):
        batch = self.get_batch()
        batch_filt = [batch.get(node.TYPE, None) for node in nodes]
        stacks = list(set(b['STACK'] for b in batch_filt if b is not None))
        return stacks

    def check_swap_space(self, node1, node2, insert=False):
        # (dif density) * (dif position)
        nodes = {1: node1, 2: node2}
        node_density = {
            n: nd.get_item_density_node(node) for n, node in nodes.items()
        }

        if insert:
            node_density[2] = 0

        cost = {
            n: nd.get_node_position_cost_unit(node, self.get_param('widthPlates'))
            for n, node in nodes.items()
        }
        # we divide it over cost[1] to scale it.
        # the bigger the better
        gains = (node_density[1] - node_density[2]) * (cost[1] - cost[2]) / cost[1]
        return gains

    def calc_previous_nodes(self, solution=None, type_node_dict=None):
        """
        :param solution: forest: a list of trees.
        :return:
        """
        if solution is None:
            solution = self.trees
        if type_node_dict is None or solution is not None:
            type_node_dict = self.get_pieces_by_type(solution=solution)
        prev_items = self.get_previous_items()
        prev_nodes = {}
        for k, v in prev_items.items():
            prev_nodes[type_node_dict[k]] = []
            for i in v:
                prev_nodes[type_node_dict[k]].append(type_node_dict[i])
        return sd.SuperDict(prev_nodes)

    def get_previous_nodes(self, solution=None, type_node_dict=None):
        if solution is None and self.previous_nodes is not None:
            return self.previous_nodes
        return self.calc_previous_nodes(solution=solution)

    def get_next_nodes(self, solution=None, type_node_dict=None):
        if solution is None and self.next_nodes is not None:
            return self.next_nodes
        previous = self.calc_previous_nodes(solution=solution)
        return previous.list_reverse()

    def check_swap_nodes_seq(self, node1, node2, insert=False):
        """
        checks if a change is beneficial in terms of sequence violations
        :param node1:
        :param node2:
        :param insert: type of swap can be insert or swap
        :return: balance of violations. Bigger is better.
        """
        # get all leaves in node1 and node2
        nodes = {1: node1, 2: node2}
        moved_items = {k: nd.get_node_leaves(v) for k, v in nodes.items()}
        precedence = self.get_previous_nodes(type_node_dict=self.type_node_dict)
        precedence_inv = self.get_next_nodes(type_node_dict=self.type_node_dict)
        # items1 = nd.get_node_leaves(node1)
        # items2 = nd.get_node_leaves(node2)
        # get all leaves between the two nodes
        first_node, second_node, neighbors = self.get_nodes_between_nodes(node1, node2)
        first_i, second_i = 1, 2
        if first_node != node1:
            first_i, second_i = 2, 1
        # print(neighbors)
        neighbor_items = set(leaf for node in neighbors for leaf in nd.get_node_leaves(node))
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

    def evaluate_swap(self, weights, **kwargs):
        # the bigger the better
        components = {
            'space': self.check_swap_space(**kwargs)
            ,'seq': self.check_swap_nodes_seq(**kwargs)
            ,'defects': self.check_swap_nodes_defect(**kwargs)
        }
        gains = {k: v * weights[k] for k, v in components.items()}
        return sum(gains.values())

    def evaluate_solution(self, weights, solution=None):
        # the smaller the better
        components = {
            'space': - self.check_space_usage(solution)
            ,'seq': len(self.check_sequence(solution, type_node_dict=self.type_node_dict))
            ,'defects': len(self.check_defects(solution))
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
            nodes += self.trees[i]
            i += 1
        return node1, node2, nodes

    def cut_waste_with_defects(self):

        return True

    def try_change_node(self, node, candidates, insert, temperature, tolerance=None, change_first=False,
                        reverse=False, evaluate=True, **kwargs):
        good_candidates = {}
        rotation = {}
        weights = kwargs.get('weights', None)
        assert weights is not None, 'weights argument cannot be empty or None!'
        for candidate in candidates:
            node1, node2 = node, candidate
            if reverse:
                node1, node2 = candidate, node
            if not nd.check_assumptions_swap(node1, node2, insert):
                continue
            result = nd.check_swap_size_rotation(node1, node2, insert=insert,
                                                   min_waste=self.get_param('minWaste'),
                                                   try_rotation=kwargs.get('try_rotation', False),
                                                   rotation_probs=kwargs.get('rotation_probs', None),
                                                   rotation_tries=kwargs.get('rotation_tries', None))
            if result is None:
                continue
            balance = 0
            if evaluate:
                balance = self.evaluate_swap(node1=node1, node2=node2, insert=insert, weights=weights)
            if tolerance is not None and balance <= tolerance:
                continue
            good_candidates[node2] = balance
            rotation[node2] = result
            if change_first:
                break
        if len(good_candidates) == 0:
            return False
        candidates_prob = sd.SuperDict({k: v for k, v in good_candidates.items()}).to_weights()
        node2 = np.random.choice(a=candidates_prob.keys_l(), size=1, p=candidates_prob.values_l())[0]
        balance = good_candidates[node2]
        # node2, balance = max(good_candidates.items(), key=lambda x: x[1])
        improve = self.acceptance_probability(balance, temperature=temperature)
        self.evaluated += 1
        if improve < rn.random():
            return False
        self.accepted += 1
        self.improved += improve == 1
        rot = rotation[node2]
        change = 'Insert'
        if not insert:
            change = 'Swap'
        if len(rot):
            change += ' (rot={})'.format(rot)
        log.debug('Started {} nodes=({}/{}, {}/{}) gain={}'.
                  format(change, node1.name, node1.PLATE_ID, node2.name, node2.PLATE_ID,
                      round(balance)))
        # seq_before = self.check_sequence()
        # balance_seq = self.check_swap_nodes_seq(node1, node2, insert=insert)
        # print('sequence before= {}\nbalance= {}'.format(len(seq_before), balance_seq))
        recalculate = nd.swap_nodes_same_level(node1, node2, insert=insert, rotation=rot,
                                                 debug=self.debug, min_waste=self.get_param('minWaste'))

        if self.debug:
            consist = self.check_consistency()
            if len(consist):
                pass

        # seq_after = self.check_sequence()
        # print('sequence after= {}'.format(len(seq_after)))
        # len([n.name for tree in self.trees for n in tree.get_descendants() if n.Y < 0]) > 0
        # update previous and next nodes:
        if recalculate:
            self.update_precedence_nodes(solution=self.trees)
        if not evaluate:
            return improve
        new = self.evaluate_solution(weights)
        old = self.best_objective
        log.debug('Finished {} nodes=({}/{}, {}/{}) gain={}, new={}, best={}, last={}'.
                  format(change, node1.name, node1.PLATE_ID, node2.name, node2.PLATE_ID,
                      round(balance), round(new), round(self.best_objective), round(self.last_objective)
                         ))
        self.last_objective = new
        if new < old:
            log.info('Best solution updated to {}!'.format(round(new)))
            self.update_best_solution(self.trees)
            self.best_objective = new

        # if self.debug:
        #     self.hist_objective.append(old)
        return improve

    def debug_nodes(self, node1, node2):
        for node in [node1, node2]:
            print("name={}\nPLATE_ID={}\nX={}\nY={}\nchildren={}\nCUT={}\nTYPE={}".format(
                node.name, node.PLATE_ID, node.X, node.Y, [ch.name for ch in node.children],
                node.CUT, node.TYPE)
            )
            print("")

    def get_node_by_name(self, name):
        nodes = []
        for tree in self.trees:
            found = tree.search_nodes(name=name)
            nodes.extend(found)
        if len(nodes) == 1:
            return nodes[0]
        else:
            return nodes

    def change_level_by_seq2(self, level, max_iter=100, max_candidates=50, **kwargs):

        prec = self.check_sequence(type_node_dict=self.type_node_dict).to_dict(result_col=1)
        fails, successes = 0, 0
        i = count = 0
        while count < max_iter and i < len(prec):
            count += 1
            node = [*prec][i]
            i += 1
            node_level = nd.find_ancestor_level(node, level)
            if node_level is None:
                continue
            other_nodes = set([nd.find_ancestor_level(p, level) for p in prec[node]])
            candidates = [s for p in other_nodes if p is not None and p.up is not None
                          for s in p.up.get_children()]
            if len(candidates) > max_candidates:
                candidates = rn.sample(candidates, max_candidates)
            for _insert in [True, False]:
                change = self.try_change_node(node=node_level, candidates=candidates,
                                              insert=_insert, **kwargs)
                if change:
                    successes += 1
                    break
                fails += 1
            if not change:
                continue
            # made a swap: recalculate
            seq = self.check_sequence(type_node_dict=self.type_node_dict)
            prec = seq.to_dict(result_col=1)
        return fails, successes

    def get_nodes_by_level(self, level, filter_fn=None):
        if filter_fn is None:
           def filter_fn(x): return True
        return [ch for tree in self.trees for ch in tree.traverse(is_leaf_fn=lambda x: x.CUT == level)
            if ch.CUT == level and filter_fn(ch)]

    def change_level(self, node, level, max_candidates=50, siblings_only=False, fn_weights=None, **kwargs):
        """
        General function that searches nodes in the same level of the node and try to swap them
        :param node:
        :param level: cut level
        :param max_candidates: maximum number of candidates to search for.
        :param siblings_only: wether to search for the node's siblings only or not.
        :param fn_weights: a function that applied to the candidate returns the probability to choose it.
        :param kwargs: tolerance, temperature, ...
        :return: True if swap was successful
        """
        fails = successes = 0
        if not self.node_in_solution(node):
            return fails, successes
        node_level = nd.find_ancestor_level(node, level)
        if node_level is None:
            return fails, successes
        if fn_weights is None:
            fn_weights = lambda x: 1
        if siblings_only:
            candidates = [c for c in node.get_sisters() if self.node_in_solution(c)]
        else:
            candidates = [c for c in self.get_nodes_by_level(level) if c != node_level and
                          self.node_in_solution(c)]
        if len(candidates) > max_candidates:
            candidates_prob = sd.SuperDict({k: fn_weights(k) for k in candidates}).to_weights()
            candidates = \
                np.random.choice(
                    a=candidates_prob.keys_l(), size=max_candidates,
                    p=candidates_prob.values_l(), replace=False
                )
            # candidates = rn.sample(candidates, max_candidates)
        # always try insert first
        for _insert in [True, False]:
            change = self.try_change_node(node=node_level, candidates=candidates, insert=_insert, **kwargs)
            fails += not change
            successes += change
            if change:
                return fails, successes
        return fails, successes

    def change_level_by_seq(self, level, max_iter=100, include_sisters=False, **kwargs):
        fails = successes = 0
        seq = self.check_sequence(type_node_dict=self.type_node_dict)
        # we search all the nodes that have a seq problem
        rem = [nd.find_ancestor_level(n, level) for tup in seq for n in tup]
        rem = [n for n in rem if n is not None]
        # we also search for all the neighbors of these nodes
        if include_sisters:
            rem_2 = [n2 for n in rem for n2 in n.get_sisters()]
            rem_2 = [n for n in rem_2 if n is not None]
            rem.extend(rem_2)
        rem = list(set(rem))
        i = count = 0
        while count < max_iter and i < len(rem):
            count += 1
            node = rem[i]
            i += 1
            _fails, _successes = self.change_level(node, level, **kwargs)
            fails += _fails
            successes += _successes
            if not _successes:
                continue
            # made a swap: recalculate
            seq = self.check_sequence(type_node_dict=self.type_node_dict)
            rem = [n for tup in seq for n in tup]
            if include_sisters:
                rem_2 = [n2 for n in rem for n2 in n.get_sisters()]
                rem_2 = [n for n in rem_2 if n is not None]
                rem.extend(rem_2)
            rem = list(set(rem))
        return fails, successes

    def change_level_all(self, level, max_iter=100, **kwargs):
        fails = successes = 0
        candidates = [c for c in self.get_nodes_by_level(level)]
        i = 0
        rn.shuffle(candidates)
        for c in candidates:
            i += 1
            if i >= max_iter:
                break
            if not self.node_in_solution(c):
                continue
            _fails, _successes = self.change_level(c, level, **kwargs)
            fails += _fails
            successes += _successes
        return fails, successes

    def node_in_solution(self, node):
        return node.get_tree_root() in self.trees

    def collapse_to_left(self, level, max_candidates=50, max_iter=100, max_wastes=None, **kwargs):
        if max_wastes is None:
            max_wastes = max_candidates
        fails = successes = 0
        # TODO: now, wastes can be any level...
        wastes = self.get_nodes_by_level(level, filter_fn=nd.is_waste)
        # wastes.sort(key=lambda x: nd.get_node_pos_tup(x))
        candidates = self.get_nodes_by_level(level, filter_fn=lambda x: not nd.is_waste(x)) + \
                     self.get_nodes_by_level(level+1, filter_fn=lambda x: not nd.is_waste(x))

        wastes_prob = sd.SuperDict(
            {c: nd.get_node_position_cost_unit(c, self.get_param('widthPlates')) for c in wastes}
        )
        # candidates.sort(reverse=True, key=lambda x: nd.get_node_pos_tup(x))
        rn.shuffle(candidates)
        i = 0
        for c in candidates:
            if not self.node_in_solution(c):
                continue
            i += 1
            if i >= max_iter:
                break
            wastes_prob = sd.SuperDict({k: v for k, v in wastes_prob.items() if self.node_in_solution(k)})
            if not len(wastes_prob):
                break
            wastes_prob = wastes_prob.to_weights()
            w_candidates = wastes_prob.keys_l()
            if len(w_candidates) > max_candidates:
                w_candidates = np.random.choice(a=wastes_prob.keys_l(), size=max_candidates,
                                                replace=False, p=wastes_prob.values_l())
            # w_before_node = [w for w in w_candidates if
            #                  nd.get_node_pos_tup(w) < nd.get_node_pos_tup(c)]
            # if len(w_before_node) > max_candidates:
            #     w_before_node = rn.sample(w_before_node, max_candidates)
            for _insert in [True, False]:
                change = self.try_change_node(node=c, candidates=w_candidates, insert=_insert, **kwargs)
                if change:
                    break
            fails += not change
            successes += change
            # if change:
            #     continue
            # # didn't work: search for any waste:
            # w_after_node = [w for w in set(w_candidates) - set(w_before_node)]
            # if len(w_after_node) > max_candidates:
            #     w_after_node = rn.sample(w_after_node, max_candidates)
            # for _insert in [True, False]:
            #     change = self.try_change_node(node=c, candidates=w_after_node, insert=_insert, **kwargs)
            #     if change:
            #         break
            # fails += not change
            # successes += change
        return fails, successes

    def merge_wastes_seq(self):
        for tree in self.trees:
            for node in tree.traverse('postorder'):
                if node.children:
                    self.join_blanks_seq(node)
                if len(node.children) == 1:
                    nd.delete_only_child(node)
        return

    def change_level_by_defects(self, level, max_iter=100, **kwargs):
        fails = successes = 0
        defects = self.check_defects()
        i = count = 0
        while count < max_iter and i < len(defects):
            count += 1
            defect = defects[i]
            node, actual_defect = defect
            i += 1
            # first, we try siblings:
            _fails, _successes = \
                self.change_level(node, level, siblings_only=True, fn_weights=nd.get_waste_density_node, **kwargs)
            fails += _fails
            successes += _successes
            if _successes:
                # made a swap: recalculate
                defects = self.check_defects()
                continue
            _fails, _successes = self.change_level(node, level, **kwargs)
            fails += _fails
            successes += _successes
            if not _successes:
                continue
            # I want to give it a try doing some local changes afterwards.
            _fails, _successes = \
                self.change_level(node, level, siblings_only=False, fn_weights=nd.get_waste_density_node, **kwargs)
            fails += _fails
            successes += _successes
            if not _successes:
                continue
            # made a swap: recalculate
            defects = self.check_defects()
        return fails, successes

    def search_waste_cuts(self, level, **kwargs):
        """
        Looks for defects that fall inside items.
        And tries to cut the sibling waste to include the defect
        on one of the extremes
        :param level:
        :param kwargs:
        :return:
        """
        fails = successes = 0
        defects = self.check_defects()
        for node, defect in defects:
            node_level = nd.find_ancestor_level(node, level)
            axis, dim = nd.get_orientation_from_cut(node_level, inv=True)
            # axis_i, dim_i = nd.get_orientation_from_cut(node_level, inv=True)
            waste = nd.find_waste(node_level, child=True)
            if waste is None:
                continue
            cut = defect[axis]
            max_size = getattr(node_level, dim)
            cut2 = max_size - cut
            if cut > cut2:
                cut = cut2
            cut += defect[dim] + 1
            if getattr(waste, dim) <= cut:
                return fails, successes
            nodes = nd.split_waste(waste, cut, min_size=self.get_param('minWaste'))
            if not len(nodes):
                return fails, successes
            candidates = [n for n in nodes[2].get_sisters() if not nd.is_waste(n)]
            for node_level in nodes.values():
                change = self.try_change_node(node_level, candidates, insert=False, **kwargs)
                fails += not change
                successes += change
                if change:
                    return fails, successes
        return fails, successes

    def search_waste_cuts_2(self, level, **kwargs):
        # this finds wastes that have defects and tries to cut them
        # so that they incircle the defect
        # We first make a cut so the waste is in the first of the pieces.
        # Then we *should* make another cut so it lands in the second!
        fails = successes = 0
        node_defect = [(n, d) for n, d in self.get_nodes_defects()
                        if n.TYPE in [-1, -3] and n.CUT == level]
        def_min_size = self.get_param('minWaste')
        for waste, defect in node_defect:
            wastes = []
            for i in range(2):
                axis, dim = nd.get_orientation_from_cut(waste)
                if i == 0:
                    cut = defect[axis] - getattr(waste, axis) + max(defect[dim] + 1, def_min_size - 1)
                else:
                    cut = defect[axis] - getattr(waste, axis) - 1
                nodes = {}
                if getattr(waste, dim) > cut:
                    nodes = nd.split_waste(waste, cut, min_size=def_min_size)
                wastes.extend(nodes.values())

            if not len(wastes):
                continue
            if len(wastes) == 4: #  I've added the node 1 two times
                wastes.pop(2)

            # Here we try to swap the nodes with their siblings:
            candidates = [n for n in wastes[0].get_sisters() if not nd.is_waste(n)]
            for waste in wastes:
                change = self.try_change_node(waste, candidates, insert=False, **kwargs)
                fails += not change
                successes += change
                if change:
                    break
        return fails, successes

    def get_max_space_usage(self, solution=None):
        if solution is None:
            solution = self.trees
        return sum(self.get_param('widthPlates') ** pos
                   for pos, tree in enumerate(solution))

    def insert_nodes_somewhere(self, level, max_iter=100, include_sisters=False, dif_level=1, max_candidates=50, **params):
        fails = successes = 0
        rem = [n for tup in self.check_sequence(type_node_dict=self.type_node_dict) for n in tup]
        defects = self.check_defects()
        items = [i for tree in self.trees[-2:] for i in nd.get_node_leaves(tree)]
        candidates = set(rem) | set([d[0] for d in defects]) | set(items)
        # candidates = set(rem)
        level_cand = [nd.find_ancestor_level(n, level) for n in candidates]
        if include_sisters:
            level_cand = [n for n in set(level_cand) if n is not None]
            level_cand_s = [s for n in level_cand for s in n.get_sisters()]
            level_cand.extend(level_cand_s)
        level_cand = [n for n in set(level_cand) if n is not None]
        rn.shuffle(level_cand)
        # candidates_all = self.get_nodes_by_level(level=level-dif_level, filter_fn=lambda x: x.TYPE in [-1, -3])
        candidates_all = self.get_nodes_by_level(level=level - dif_level)
        i = 0
        for c in level_cand:
            if i >= max_iter:
                break
            candidates = candidates_all
            if len(candidates) > max_candidates:
                candidates = rn.sample(candidates_all, max_candidates)
            change = self.try_change_node(c, candidates, insert=True, **params)
            fails += not change
            successes += change
            i += 1
        return fails, successes

    # def insert_node_somewhere(self, node1, **params):
    #     _params = dict(params)
    #     # I want as candidates one level=1 block from each try.
    #     return change

    def try_reduce_nodes(self, level):
        candidates = self.get_nodes_by_level(level)
        for c in candidates:
            nd.reduce_children(c)
        return True

    def insert_node_inside_node(self, node1, node2, kwargs):
        # we do not know anything of node2.
        # we don't care about the CUT.
        # we just want to fit the node in the first possible
        # descendant. So, we'll traverse the nodes.
        node1.CUT = node2.CUT

        # if node2 is -2... I'll try to insert it *inside* it.
        # I want to see if it's worth it to continue (if it fits still)
        if node2.TYPE == -2:
            if geom.plate_inside_plate(nd.node_to_plate(node1),
                                       nd.node_to_plate(node2),
                                       turn=True):
                # either way, I'll try to make it fit with together with the children
                # (new or old). If successful, done!
                for ch in node2.children:
                    change = self.insert_node_inside_node(node1, ch, kwargs)
                    if change:
                        return True

        # if node2 is a waste:
        # I try to create a child waste and use this to insert.
        if nd.is_waste(node2):
            return self.try_change_node(node1, [node2], **kwargs)

        # If I failed inserting in the children or if node2 is an item:
        # I try to insert next to the node2
        next_sibling = nd.get_next_sibling(node2)
        if next_sibling is None:
            return False
        return self.try_change_node(node1, [next_sibling], **kwargs)

    def add_jumbo(self, num=1):
        plate_W = self.get_param('widthPlates')
        plate_H = self.get_param('heightPlates')
        for n in range(num):
            tree_id = len(self.trees)
            tree = nd.create_plate(width=plate_W, height=plate_H, id=tree_id, defects=self.get_defects_plate(tree_id))
            self.trees.append(tree)
        return True

    def add_1cut(self):
        for tree in self.trees:
            if not tree.children:
                nd.duplicate_waste_into_children(tree)
        return

    def swap_jumbo(self, jumbo1, jumbo2):
        if jumbo1 == jumbo2:
            return False
        tree1 = self.trees[jumbo1]
        tree2 = self.trees[jumbo2]
        nd.change_feature(tree1, feature="PLATE_ID", value=jumbo2)
        nd.change_feature(tree2, feature="PLATE_ID", value=jumbo1)
        self.trees[jumbo2] = tree1
        self.trees[jumbo1] = tree2
        return True

    def try_swap_jumbo(self, jumbo1, jumbo2, **params):
        """
        I just need to declare crossings for all leaves in each plate.
        And evaluate the sequence
        For the space: I use the densisties of the jumbos and the diference in position
        is one jumbo width.
        For the defects... I intersect the leaves with the other plates defects.
        Actually it is easier to do the change and revert it if not correct
        """
        self.swap_jumbo(jumbo1, jumbo2)
        temperature = params['temperature']
        weights = params['weights']
        new = self.evaluate_solution(weights)
        old = self.best_objective
        balance = old - new
        if balance > 0:
        # if self.acceptance_probability(balance, temperature=temperature) < rn.random():
            self.swap_jumbo(jumbo1, jumbo2)
            return False
        log.debug('I just swapped jumbos {} and {}: gain={}, new={}, best={}, last={}'.format(
            jumbo1, jumbo2, round(balance),
            round(new), round(self.best_objective), round(self.last_objective)
        ))
        self.last_objective = new
        if new < old:
            log.info('Best solution updated to {}!'.format(round(new)))
            self.update_best_solution(self.trees)
            self.best_objective = new
        return True

    def jumbos_swapping(self, params, max_jumbos=None):
        change = False
        count = 0
        jumbo_pairs = [(pos1, pos2) for pos1, tree1 in enumerate(self.trees)
                       for pos2, tree2 in enumerate(self.trees[pos1+1:], start=pos1+1)]
        rn.shuffle(jumbo_pairs)
        for pos1, pos2 in jumbo_pairs:
            change |= self.try_swap_jumbo(pos1, pos2, **params)
            count += 1
            if max_jumbos is not None and count >= max_jumbos:
                return change
        return change

    def mirror_jumbo_x(self, jumbo):
        """
        This inverses the position of 1cuts in the jumbo.
        :param jumbo:
        :return:
        """
        tree = self.trees[jumbo]
        nodes = tree.get_children()
        total = self.get_param("widthPlates")
        # we change x_position
        for n in nodes:
            new_pos = total - (n.X + n.WIDTH)
            nd.mod_feature_node(node=n, quantity=new_pos - n.X, feature='X')
        # we change the children's position:
        tree.children = list(reversed(nodes))
        pass

    def try_mirror_jumbo_x(self, jumbo, **params):
        self.mirror_jumbo_x(jumbo)
        temperature = params['temperature']
        weights = params['weights']
        new = self.evaluate_solution(weights)
        old = self.best_objective
        balance = old - new
        if balance > 0:
        # if self.acceptance_probability(balance, temperature=temperature) < rn.random():
            self.mirror_jumbo_x(jumbo)
            return False
        log.debug('I just mirrored jumbo {}: gain={}, new={}, best={}'.format(
            jumbo, round(balance),
            round(new), round(self.best_objective))
        )
        if new < old:
            log.info('Best solution updated to {}!'.format(round(new)))
            self.update_best_solution(self.trees)
            self.best_objective = new
        return True

    def jumbos_mirroring(self, params, max_jumbos=None):
        change = False
        count = 0
        enumeration = [pos1 for pos1, tree1 in enumerate(self.trees)]
        rn.shuffle(enumeration)
        for pos1 in enumeration:
            change |= self.try_mirror_jumbo_x(pos1, **params)
            count += 1
            if max_jumbos is not None and count >= max_jumbos:
                return change
        return change

    def try_change_tree(self, params, num_iterations, tolerance=200):
        incumbent = None
        # For now, we only search for trees without defects...
        # because if not we're going to be breaking them.
        for x in range(10):
            num_tree = rn.choice(range(len(self.trees)))
            if not self.trees[num_tree].DEFECTS or rn.random() < 0.2:
                incumbent = self.trees[num_tree]
                break
        if not incumbent:
            return None
        trees = self.get_initial_solution(params=dict(params), num_tree=num_tree, num_iterations=num_iterations)
        if trees is None:
            return
        candidate = trees[0]
        if self.calculate_objective([candidate]) < \
                self.calculate_objective([incumbent]) - tolerance:
            nd.change_feature(candidate, 'PLATE_ID', num_tree)
            self.trees[num_tree] = candidate
            self.update_precedence_nodes(self.trees)

            new = self.evaluate_solution(params['weights'])
            old = self.best_objective
            balance = old - new
            log.debug('I just remade tree {}: gain={}, new={}, best={}, last={}'.format(
                num_tree, round(balance), round(new),
                round(self.best_objective), round(self.last_objective)
            ))
            self.last_objective = new
            if balance > 0:
                log.info('Best solution updated to {}!'.format(round(new)))
                self.update_best_solution(self.trees)
                self.best_objective = new
            return True

            # log.info('better tree found!')
        return

    def get_initial_solution(self, params, num_tree=None, num_iterations=1000, num_process=None):
        """
        :param params:
        :param num_tree: optional to only deal with 1 tree for modifying it
        :param num_iterations: number of iterations to do
        :param num_process: number of processors
        :return:
        """
        if num_process is None:
            num_process = multi.cpu_count() - 1
        params = dict(params)
        defaults = \
            {
            'evaluate': False
            ,'insert': True
            ,'rotation_probs': [0.50, 0.50, 0, 0]
            ,'try_rotation': True
            ,'rotation_tries': 2
            }
        params = {**params, **defaults}
        pool = multi.Pool(processes=num_process)
        if num_tree is not None:
            incumbent = self.trees[num_tree]
            nodes = nd.get_node_leaves(incumbent, min_type=0)
            items_i = [n.TYPE for n in nodes]
            if not items_i:
                return None
            all_items = self.get_batch()
            stacks = sd.SuperDict(all_items).filter(items_i).\
                index_by_property('STACK')
            stacks = {k: [*v.values() ]for k, v in stacks.items()}
            defects = {incumbent.PLATE_ID: incumbent.DEFECTS}
            limit_trees = 1
        else:
            stacks = self.get_items_per_stack()
            defects = self.get_defects_per_plate()
            limit_trees = None
        args = {
            'params': params,
            'global_params': self.get_param(),
            'items_by_stack': stacks,
            'defects': defects,
            'sorting_function': nd.sorting_items,
            'limit_trees': limit_trees
        }
        result_x = {}
        for x in range(num_iterations):
            # result = nd.place_items_on_trees(**args)
            seed = {'seed': int(rn.random()*1000)}
            result_x[x] = pool.apply_async(nd.place_items_on_trees, kwds={**args, **seed})

        for x, result in result_x.items():
            result_x[x] = result.get(timeout=100)

        pool.close()
        values = [v for v in result_x.values() if v is not None]
        if not values:
            return None
        candidate = min(values, key=self.calculate_objective)
        return candidate

    def update_precedence_nodes(self, solution):
        self.type_node_dict = self.get_pieces_by_type(solution=solution)
        self.previous_nodes = self.calc_previous_nodes(solution=solution)
        self.next_nodes = self.previous_nodes.list_reverse()
        return True

    def solve(self, options, warm_start=False):
        import pprint as pp

        now = time.time()
        end = options['timeLimit']
        self.debug = options.get('debug', False)

        params = options['heur_params']
        weights = params['weights']

        if not warm_start:
            self.trees = self.get_initial_solution(params, num_iterations=params.get('iterations_initial', 1000))
            self.update_precedence_nodes(self.trees)
            self.best_objective = self.evaluate_solution(params['weights'])
            self.add_jumbo(params['extra_jumbos'])
            # return
        self.order_all_children()
        self.clean_empty_cuts()
        self.join_blanks()
        self.clean_empty_cuts_2()
        self.correct_plate_node_ids()
        assert 'weights' in params
        temp = params['temperature']
        try_rotation = params['try_rotation']
        coolingRate = params['cooling_rate']
        fsc = {}
        cats = ['cuts', 'cuts2', 'seq', 'def',
                'all', 'interlevel', 'seq2', 'collapse']
        fail_success_acum_cat = {c: (0, 0) for c in cats}
        count = 0
        changed_flag = False
        b_accepted = b_improved = 0
        max_wastes = params['max_candidates']
        while True:
            # self.jumbos_swapping(params, 5)
            # self.jumbos_mirroring(params, 5)
            for x in range(params['main_iter']):
                # for i in range(len(self.trees)//4):
                self.try_reduce_nodes(1)
                level = np.random.choice(a=[1, 2, 3], p=params['level_probs'])
                # if rn.random() > 0.5:
                self.try_change_tree(params,
                                     num_iterations=params.get('iterations_remake', 10),
                                     tolerance=0)
                fsc['collapse'] = self.collapse_to_left(level, **params, max_wastes=max_wastes)
                params['try_rotation'] = level >= 2 and try_rotation
                if not changed_flag and self.best_objective < weights['defects']//2:
                    params = {**options['heur_params'], **options['heur_optim']}
                    try_rotation = params['try_rotation']
                    weights = params['weights']
                    # temp = params['temperature']
                    changed_flag = True
                    log.info('activate optimisation')
                log.debug('DO: collapse left')
                fsc['collapse'] = self.collapse_to_left(level, **params, max_wastes=max_wastes)
                log.debug('DO: merge_wastes')
                if rn.random() > 0.8:
                    self.merge_wastes_seq()
                fsc['cuts'] = 0, 0
                if level == 1:
                    if rn.random() > 0.5:
                        fsc['cuts'] = self.search_waste_cuts(1, **params)
                include_sisters = True
                if rn.random() > 0.5:
                    fsc['cuts2'] = self.search_waste_cuts_2(level, **params)
                log.debug('DO: collapse left')
                fsc['collapse'] = self.collapse_to_left(level, **params, max_wastes=max_wastes)
                log.debug('DO: search_waste_cuts')
                fsc['seq2'] = self.change_level_by_seq2(level, **params)
                fsc['seq'] = self.change_level_by_seq(level, include_sisters=False, **params)
                fsc['def'] = self.change_level_by_defects(level, **params)
                log.debug('DO: change_level_by_defects')
                fsc['all'] = self.change_level_all(level, **params)
                if rn.random() > 0.5:
                    self.clean_empty_cuts_2()
                self.add_1cut()
                fsc['interlevel'] = \
                    self.insert_nodes_somewhere(level + 1, include_sisters=include_sisters, **params)
                if level in [2, 3]:
                    fsc['interlevel'] = \
                        self.insert_nodes_somewhere(level + 1, include_sisters=include_sisters, dif_level=2, **params)
                count += 1

            new_imp_ratio = (self.improved - b_improved) * 100 / (self.accepted - b_accepted + 1)
            b_accepted = self.accepted
            b_improved = self.improved
            if new_imp_ratio < 60:
                temp *= (1 - coolingRate)
            clock = time.time() - now
            if temp < 1 or clock > end:
                break
            seq = self.check_sequence(type_node_dict=self.type_node_dict)
            defects = self.check_defects()

            # fails, successes = tuple(map(sum, zip(*fail_success_acum)))
            log.info("TEMP={}, seq={}, def={}, best={}, time={}, evald={}, accptd={}, imprd={}, ratio_imp={}".format(
                round(temp),
                len(seq),
                len(defects),
                round(self.best_objective),
                round(clock),
                self.evaluated,
                self.accepted,
                self.improved,
                round(new_imp_ratio))
            )
            if count % 100 == 0:
                log.debug(fail_success_acum_cat)
        self.trees = self.best_solution
        self.clean_empty_cuts_2()
        self.merge_wastes_seq()
        self.trees = [tree for tree in self.trees if nd.get_node_leaves(tree)]
        pass


if __name__ == "__main__":
    # TODO: do jumbo swaps. But *very* rarely and when nothing happens.
    # TODO: add more random movements.
    # TODO: go back to initial solution *very* rarely
    # TODO: dynamic weights.
    # TODO: implement the multilevel swap
    # TODO: for eating wastes after swap, I can eat wastes right of the defect
        # or
    # TODO: profiling
    # TODO: add other items from stack to sequence swaps.
    # cut.
    import pprint as pp
    case = pm.OPTIONS['case_name']
    # path = pm.PATHS['experiments'] + e
    path = pm.PATHS['experiments'] + case + '/'

    self = ImproveHeuristic.from_input_files(case_name=case, path=path)
    self.solve(pm.OPTIONS)
