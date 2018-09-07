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

    def get_stacks_from_nodes(self, nodes):
        batch = self.get_batch()
        batch_filt = [batch.get(node.TYPE, None) for node in nodes]
        stacks = list(set(b['STACK'] for b in batch_filt if b is not None))
        return stacks

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

    def cut_waste_with_defects(self):

        return True

    def try_change_node(self, node, candidates, insert, params, pool, reverse=False, evaluate=True):
        weights = params['weights']
        temperature = params['temperature']
        assert weights is not None, 'weights argument cannot be empty or None!'
        args_evaluate = {
            'insert': insert,
            'global_params': self.get_param(),
            'weights': weights,
            'solution': self.trees,
            'precedence': self.get_previous_nodes(type_node_dict=self.type_node_dict),
            'precedence_inv': self.get_next_nodes(type_node_dict=self.type_node_dict)
        }
        args_check_size = {
            'insert': insert,
            'min_waste': self.get_param('minWaste'),
            'params': params
        }
        candidates_eval = {}
        # num_processes = pool._processes
        # iterator_per_proc = math.ceil(num_iterations / num_process)
        for candidate in candidates:
            node1, node2 = node, candidate
            if reverse:
                node1, node2 = candidate, node
            if not nd.check_assumptions_swap(node1, node2, insert):
                continue
            nodes = {'node1': node1, 'node2': node2}
            candidates_eval[node2] = \
                nd.check_swap_two_nodes(nodes, args_check_size, args_evaluate, evaluate, params)
        # for x, result in candidates_eval.items():
        #     candidates_eval[x] = result.get(timeout=10)
        candidates_eval = {k: v for k, v in candidates_eval.items() if v is not None}
        if len(candidates_eval) == 0:
            return False
        candidates_prob = sd.SuperDict({k: v[0] for k, v in candidates_eval.items()}).to_weights()
        node2 = np.random.choice(a=candidates_prob.keys_l(), size=1, p=candidates_prob.values_l())[0]
        balance, rot = candidates_eval[node2]
        # node2, balance = max(good_candidates.items(), key=lambda x: x[1])
        improve = self.acceptance_probability(balance, temperature=temperature)
        self.evaluated += 1
        if improve < rn.random():
            return False
        self.accepted += 1
        self.improved += improve == 1
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

    def change_level_by_seq2(self, level, params, pool):
        max_candidates = params['max_candidates']
        max_iter = params['max_iter']
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
                change = self.try_change_node(node=node_level, candidates=candidates, insert=_insert, params=params, pool=pool)
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

    def change_level(self, node, level, params, pool, siblings_only=False, fn_weights=None):
        """
        General function that searches nodes in the same level of the node and try to swap them
        :param node:
        :param level: cut level
        :param siblings_only: wether to search for the node's siblings only or not.
        :param fn_weights: a function that applied to the candidate returns the probability to choose it.
        :param params: tolerance, temperature, ...
        :return: True if swap was successful
        """
        max_candidates = params['max_candidates']
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
            change = self.try_change_node(node=node_level, candidates=candidates, insert=_insert, params=params, pool=pool)
            fails += not change
            successes += change
            if change:
                return fails, successes
        return fails, successes

    def change_level_by_seq(self, level, params, pool, include_sisters=False):
        max_iter = params['max_iter']
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
            _fails, _successes = self.change_level(node, level, params, pool)
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

    def change_level_all(self, level, params, pool):
        max_iter = params['max_iter']
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
            _fails, _successes = self.change_level(c, level, params, pool)
            fails += _fails
            successes += _successes
        return fails, successes

    def node_in_solution(self, node):
        return node.get_tree_root() in self.trees

    def collapse_to_left(self, level, params, pool, max_wastes=None):
        # if max_wastes is None:
        #     max_wastes = max_candidates
        max_candidates = params['max_candidates']
        max_iter = params['max_iter']
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
                change = self.try_change_node(node=c, candidates=w_candidates, insert=_insert, params=params, pool=pool)
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

    def change_level_by_defects(self, level, params, pool):
        max_iter = params['max_iter']
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
                self.change_level(node, level, siblings_only=True, fn_weights=nd.get_waste_density_node, params=params, pool=pool)
            fails += _fails
            successes += _successes
            if _successes:
                # made a swap: recalculate
                defects = self.check_defects()
                continue
            _fails, _successes = self.change_level(node, level, params, pool)
            fails += _fails
            successes += _successes
            if not _successes:
                continue
            # I want to give it a try doing some local changes afterwards.
            _fails, _successes = \
                self.change_level(node, level, siblings_only=False, fn_weights=nd.get_waste_density_node, params=params, pool=pool)
            fails += _fails
            successes += _successes
            if not _successes:
                continue
            # made a swap: recalculate
            defects = self.check_defects()
        return fails, successes

    def search_waste_cuts(self, level, params, pool):
        """
        Looks for defects that fall inside items.
        And tries to cut the sibling waste to include the defect
        on one of the extremes
        :param level:
        :param params:
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
                change = self.try_change_node(node_level, candidates, insert=False, params=params, pool=pool)
                fails += not change
                successes += change
                if change:
                    return fails, successes
        return fails, successes

    def search_waste_cuts_2(self, level, params, pool):
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
                change = self.try_change_node(waste, candidates, insert=False, params=params, pool=pool)
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

    def insert_nodes_somewhere(self, level, params, pool, include_sisters=False, dif_level=1):
        max_iter = params['max_iter']
        max_candidates = params['max_candidates']
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
            change = self.try_change_node(c, candidates, insert=True, params=params, pool=pool)
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

    def insert_node_inside_node(self, node1, node2, params, pool):
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
                    change = self.insert_node_inside_node(node1, ch, params=params, pool=pool)
                    if change:
                        return True

        # if node2 is a waste:
        # I try to create a child waste and use this to insert.
        if nd.is_waste(node2):
            return self.try_change_node(node1, [node2], params=params, insert=True, pool=pool)

        # If I failed inserting in the children or if node2 is an item:
        # I try to insert next to the node2
        next_sibling = nd.get_next_sibling(node2)
        if next_sibling is None:
            return False
        return self.try_change_node(node1, [next_sibling], params=params, insert=True, pool=pool)

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

    def try_change_tree(self, options, num_iterations, pool, num_process, tolerance=200):
        params = options['heur_params']
        remake_opts = options['heur_remake']
        probs = remake_opts.get('num_trees', [0.7, 0.2, 0.1])
        tree_options = range(1, len(probs)+1)
        start = rn.choice(range(len(self.trees)))
        size = min(
            np.random.choice(a=tree_options, p=probs),
            len(self.trees) - start
        )
        num_trees = range(start, start + size)
        incumbent = [self.trees[n] for n in num_trees]
        candidate = self.get_initial_solution(pool=pool, options=options, num_trees=num_trees,
                                              num_iterations=num_iterations, num_process=num_process)
        if candidate is None:
            return None
        # candidate = trees[0]
        # if the candidate is worse, we do not even bother.
        if self.calculate_objective(candidate) > \
                self.calculate_objective(incumbent) - tolerance:
            return None
        # With 20% prob we accept worse solutions even if there are more defects.
        # This can later be improved if we try to make the solution feasible regarding defects.
        if len(self.check_defects(incumbent)) < len(self.check_defects(candidate)) and rn.random() > 0.2:
            return None
        for pos, plate_id in enumerate(num_trees):
            nd.change_feature(candidate[pos], 'PLATE_ID', plate_id)
            self.trees[plate_id] = candidate[pos]
        self.update_precedence_nodes(self.trees)

        new = self.evaluate_solution(params['weights'])
        old = self.best_objective
        balance = old - new
        log.debug('I just remade tree {}: gain={}, new={}, best={}, last={}'.format(
            num_trees, round(balance), round(new),
            round(self.best_objective), round(self.last_objective)
        ))
        self.last_objective = new
        if balance > 0:
            log.info('Best solution updated to {}!'.format(round(new)))
            self.update_best_solution(self.trees)
            self.best_objective = new
        return True

    def get_initial_solution(self, options, pool, num_process, num_trees=None, num_iterations=1000):
        """
        :param options:
        :param num_trees: optional to only deal with X trees for modifying it. A list of num trees!
        :param num_iterations: number of iterations to do
        :param num_process: number of processors
        :return: a list of trees with all the items (solution or partial solution)
        """
        params = options['heur_params']
        remake = options['heur_remake']
        params = dict(params)
        defaults = \
            {
            'evaluate': False
            ,'insert': True
            ,'rotation_probs': remake.get('rotation_remake', [0.5, 0.5, 0, 0])
            ,'try_rotation': True
            ,'rotation_tries': 2
            }
        params = {**params, **defaults}
        iterator_per_proc = math.ceil(num_iterations / num_process)
        if num_trees is not None:
            incumbent = [self.trees[n] for n in num_trees]
            # incumbent = self.trees[num_tree]
            items_i = [l.TYPE for tree in incumbent for l in nd.get_node_leaves(tree, min_type=0)]
            # items_i = [n. for n in nodes]
            if not items_i:
                return None
            all_items = self.get_batch()
            stacks = sd.SuperDict(all_items).filter(items_i).\
                index_by_property('STACK')
            stacks = {k: [*v.values()]for k, v in stacks.items()}
            limit_trees = len(incumbent)
        else:
            stacks = self.get_items_per_stack()
            limit_trees = None
        args = {
            'params': params,
            'global_params': self.get_param(),
            'items_by_stack': stacks,
            'defects': self.get_defects_per_plate(),
            'sorting_function': nd.sorting_items,
            'limit_trees': limit_trees,
            'num_iters': iterator_per_proc
        }
        result_x = {}
        for x in range(num_process):
            # result = nd.place_items_on_trees(**args)
            seed = {'seed': int(rn.random()*1000)}
            result_x[x] = nd.iter_insert_items_on_trees(**{**args, **seed})
            # result_x[x] = pool.apply_async(nd.iter_insert_items_on_trees, kwds={**args, **seed})

        # for x, result in result_x.items():
        #     result_x[x] = result.get(timeout=10000)


        candidates = [sol for result_proc in result_x.values() for sol in result_proc if sol is not None]
        if not candidates:
            return None
        # Here we have two hierarchical criteria:
        # 1. number of defects
        # 2. objective function
        # For this, I need to change plate's ids to get the correct defects.
        if num_trees is not None:
            plate_W = self.get_param('widthPlates')
            plate_H = self.get_param('heightPlates')
            for candidate in candidates:
                for iter, plate_id in enumerate(num_trees):
                    if iter >= len(candidate):
                        new_tree = nd.create_plate(width=plate_W, height=plate_H, id=plate_id,
                                                   defects=self.get_defects_plate(plate_id))
                        candidate.append(new_tree)
                    else:
                        candidate[iter].PLATE_ID = plate_id
        candidate = min(candidates, key=lambda x: (len(self.check_defects(x)),
                                                   self.calculate_objective(x, discard_empty_trees=True)))
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
        p_remake = options['heur_remake']
        weights = params['weights']

        num_process = multi.cpu_count() - 1
        pool = multi.Pool(processes=num_process)

        if not warm_start:
            self.trees = \
                self.get_initial_solution(options = options, pool=pool, num_process=num_process,
                                          num_iterations=p_remake['iterations_initial'])
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
                self.try_change_tree(options=options, pool=pool,
                                     num_iterations=p_remake.get('iterations_remake', 10),
                                     tolerance=0, num_process=num_process)
                fsc['collapse'] = self.collapse_to_left(level, params=params, max_wastes=max_wastes, pool=pool)
                params['try_rotation'] = level >= 2 and try_rotation
                if not changed_flag and self.best_objective < weights['defects']//2:
                    params = {**options['heur_params'], **options['heur_optim']}
                    try_rotation = params['try_rotation']
                    weights = params['weights']
                    # temp = params['temperature']
                    changed_flag = True
                    log.info('activate optimisation')
                log.debug('DO: collapse left')
                fsc['collapse'] = self.collapse_to_left(level, params=params, max_wastes=max_wastes, pool=pool)
                log.debug('DO: merge_wastes')
                if rn.random() > 0.8:
                    self.merge_wastes_seq()
                fsc['cuts'] = 0, 0
                if level == 1:
                    if rn.random() > 0.5:
                        fsc['cuts'] = self.search_waste_cuts(1, params=params, pool=pool)
                include_sisters = True
                if rn.random() > 0.5:
                    fsc['cuts2'] = self.search_waste_cuts_2(level, params=params, pool=pool)
                log.debug('DO: collapse left')
                fsc['collapse'] = self.collapse_to_left(level, params, max_wastes=max_wastes, pool=pool)
                log.debug('DO: search_waste_cuts')
                fsc['seq2'] = self.change_level_by_seq2(level, params=params, pool=pool)
                fsc['seq'] = self.change_level_by_seq(level, include_sisters=False, params=params, pool=pool)
                fsc['def'] = self.change_level_by_defects(level, params=params, pool=pool)
                log.debug('DO: change_level_by_defects')
                fsc['all'] = self.change_level_all(level, params=params, pool=pool)
                if rn.random() > 0.5:
                    self.clean_empty_cuts_2()
                self.add_1cut()
                fsc['interlevel'] = \
                    self.insert_nodes_somewhere(level + 1, include_sisters=include_sisters, params=params, pool=pool)
                if level in [2, 3]:
                    fsc['interlevel'] = \
                        self.insert_nodes_somewhere(level + 1, include_sisters=include_sisters, dif_level=2, params=params, pool=pool)
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
        pool.close()
        self.trees = self.best_solution
        self.clean_empty_cuts_2()
        self.merge_wastes_seq()
        self.trees = [tree for tree in self.trees if nd.get_node_leaves(tree)]
        pass


if __name__ == "__main__":
    # TODO: add more random movements.
    # TODO: dynamic weights.
    # TODO: implement the multilevel swap
    # TODO: for eating wastes after swap, I can eat wastes right of the defect
    # TODO: profiling
    # TODO: make tree recreation work with defects.
    # TODO: parallelize search for candidates moves.
    # cut.
    import pprint as pp
    case = pm.OPTIONS['case_name']
    # path = pm.PATHS['experiments'] + e
    path = pm.PATHS['experiments'] + case + '/'

    self = ImproveHeuristic.from_input_files(case_name=case, path=path)
    self.solve(pm.OPTIONS)
