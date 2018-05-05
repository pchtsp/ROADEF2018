import package.solution as sol
import package.params as pm
import copy
import package.superdict as sd
import package.tuplist as tl
import package.nodes as nd
import numpy as np

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
        return

    def move_item_inside_node(self):
        # This should keep the cuts the same but
        # maybe put the "waste"
        defects = self.check_defects()
        for defect in defects:
            node = defect[0]
            for sibling in node.get_sisters():
                if sibling.TYPE != -1:
                    continue
                if self.defects_in_node(sibling):
                    continue
                print('Found! Swapping {} and {}'.format(node.name, sibling.name))
                result = self.swap_nodes_same_level(node, sibling)
                break
        return

    def exchange_level1_nodes_defects(self):
        # TODO: search for the thinest slice!
        # TODO: deal with several defects
        # TODO: try to exchange plates from different bins
        defects = self.check_defects()
        for defect in defects:
            node, actual_defect = defect
            node_level1 = nd.find_ancestor_level(node, 1)
            dist_x = actual_defect['X'] + actual_defect['WIDTH'] - node_level1.X
            dist_y = min(actual_defect['Y'], node_level1.HEIGHT - actual_defect['Y'])
            for sibling in node_level1.get_sisters():
                if self.defects_in_node(sibling):
                    continue
                candidates = [n for n in sibling.children if n.TYPE == -1 and n.HEIGHT >= dist_y]
                if len(candidates) == 0:
                    continue
                print('Found! Swapping {} and {}'.format(node_level1.name, sibling.name))
                result = self.swap_nodes_same_level(node_level1, sibling)
                break

    def exchange_level1_nodes_seq(self):
        seq = self.check_sequence()
        pass
        # for n1, n2 in seq:
        #
        #     node_level1 = heur.find_ancestor_level(defect[0], 1)
        #     actual_defect = defect[1]
        #     dist_x = actual_defect['X'] + actual_defect['WIDTH'] - node_level1.X
        #     dist_y = min(actual_defect['Y'], node_level1.HEIGHT - actual_defect['Y'])
        #     for sibling in node_level1.get_sisters():
        #         if heur.defects_in_node(sibling):
        #             continue
        #         candidates = [n for n in sibling.children if n.TYPE == -1 and n.HEIGHT >= dist_y]
        #         if len(candidates) == 0:
        #             continue
        #         print('Found! Swapping {} and {}'.format(node_level1.name, sibling.name))
        #         result = heur.swap_siblings(node_level1, sibling)
        #         break


    def fill_defect_with_waste(self):
        # given a defect in a plate, try to put a waste
        # instead of an item
        pass

    def insert_node_at_position(self, node, destination, position):
        """
        :param node: node I'm going to insert.
        :param destination: parent node were I want to insert it.
        :param position: position of a children on the parent node (1 - num_children)
        :return:
        """
        # 1. take node out from where it is (detach and everything)
        # update all the positions of siblings accordingly
        # if the node was in the same destination and in a smaller position:
        #   we need to reduce the position by one.
        parent = node.up
        plate, ch_pos = nd.get_node_pos(node)
        if parent == destination and ch_pos < position:
            position -= 1
        axis, dim = nd.get_orientation_from_cut(node)
        for sib in parent.children[ch_pos+1:]:
            nd.move_node(node=sib, movement=-getattr(node, dim), axis=axis)
        node.detach()

        # 2. move all nodes at destination starting from the marked position
        # to make space and assign new plate, and new axis
        # add node to new parent

        # we make space:
        if position < len(destination.children):
            axis_dest = getattr(destination.children[position], axis)
            for sib in destination.children[position:]:
                nd.move_node(node=sib, movement=getattr(node, dim), axis=axis)
        else:
            # we're puting the node in a new, last place:
            # so we do not move anything.
            last_node = destination.children[-1]
            axis_dest = getattr(last_node, axis) + getattr(last_node, dim)
        # we make the move:
        nd.change_feature(node, 'PLATE_ID', destination.PLATE_ID)
        dist = axis_dest - getattr(node, axis)
        nd.move_node(node, dist, axis)
        destination.add_child(node)

        # 3. update parents order:
        nd.order_children(destination)

        return True

    def check_swap_size(self, node1, node2, insert=False, cut=False):
        # TODO: we need to guarantee there's always a waste for every cut. Even if it's 0.
        # if insert=True, we insert node1 before node2. So we count the whole size
        siblings = node1.up == node2.up
        if siblings:
            # siblings? no problem
            return True
        _, dim = nd.get_orientation_from_cut(node1)

        if insert:
            dif_length = getattr(node1, dim)
        else:
            dif_length = getattr(node1, dim) - getattr(node2, dim)
        if not dif_length:
            # same size? no problem
            return True
        # check there's space in smaller piece waste to eat.
        if dif_length < 0:
            # node1 is always bigger
            dif_length = -dif_length
            node1, node2 = node2, node1
        # we check the small one's waste:
        waste2 = nd.find_waste_sibling(node2)
        # we need also the first one's waste, to expand it.
        waste1 = nd.find_waste_sibling(node1)
        if waste1 is None or waste2 is None:
            # no waste in node2 or waste1. cannot do the change
            return False
        waste_length = getattr(waste2, dim)
        if waste_length < dif_length:
            # waste is not big enough
            return False
        if cut:
            setattr(waste2, dim, getattr(waste2, dim) - dif_length)
            setattr(waste1, dim, getattr(waste1, dim) + dif_length)
        return True

    def check_assumptions_swap(self, node1, node2):
        assert node1.CUT == node2.CUT, \
            'nodes {} and {} need to have the same level'.format(node1.name, node2.name)
        axis_inv, inv_dim = nd.get_orientation_from_cut(node1, inv=True)
        assert getattr(node1, inv_dim) == getattr(node2, inv_dim), \
            'nodes {} and {} need to have the same fixed dim'.format(node1.name, node2.name)
        assert getattr(node1, axis_inv) == getattr(node2, axis_inv), \
            'nodes {} and {} need to have the same fixed axis'.format(node1.name, node2.name)

    def swap_nodes_same_level(self, node1, node2, insert=False):
        # for now, they need to be level=1
        # or siblings.
        # meaning: sharing the dimension we are not changing and the axis.
        # we do not want to make cuts for the moment.
        # if insert_only=True, we insert node1 before node2 but we do not move node2
        self.check_assumptions_swap(node1, node2)

        if not self.check_swap_size(node1, node2, insert, cut=True):
            return False

        print('Found! Change between nodes {} and {}'.format(node1.name, node2.name))
        parent1 = node1.up
        parent2 = node2.up
        plate1, ch_pos1 = nd.get_node_pos(node1)
        plate2, ch_pos2 = nd.get_node_pos(node2)
        self.insert_node_at_position(node1, parent2, ch_pos2)
        if not insert:
            self.insert_node_at_position(node2, parent1, ch_pos1)
        # we need to update the waste at the smaller node
        # self.check_swap_size(node1, node2, insert_only, cut=True)
        # for now we're doing it before the swapping...
        return True

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
            '{} and {} are not siblings'.format(node1, node2)
        assert node1.TYPE == -1 and node2.TYPE == -1, \
            '{} and {} are not waste'.format(node1, node2)

        if node1.CUT % 2:  # cuts 1 and 3
            dim = 'WIDTH'
            axis = 'X'
        else:  # cut 2 and 4
            dim = 'HEIGHT'
            axis = 'Y'
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

    def join_blanks(self):
        for tree in self.trees:
            for v in tree.traverse():
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
                    if children[pos] in candidates:
                        pos -= 1
                        continue
                    c = candidates_s.pop(0)
                    self.swap_nodes_same_level(c, children[pos])
                candidates = children[min_pos:]
                while len(candidates) > 1:
                    self.join_neighbors(candidates[0], candidates[1])
                    candidates.pop(1)
        return True

    def check_swap_nodes_defect(self, node1, node2, insert=False):
        # we need to check if each node will be able to arrange itself
        # between the defects present in the jumbo.
        # before that: we'll only check if exchanging the nodes gets a
        # positive defect balance.
        # We need to:
        # 1. list the nodes to check (re-use the 'nodes between nodes'?)
            # here there's two sets of nodes, I guess?
        # 1b. list the relevant defects (for each the plate?)
            # here there's two sets of defects, I guess?
        # 1c. calculate the present defect violations.
        # 2. calculate the distance to move them.
        # 3. create squares with new positions.
        # 4. calculate new defect violations
        plate1 = node1.PLATE_ID
        plate2 = node2.PLATE_ID
        if plate1 == plate2:
            first_node, second_node, nodes = self.get_nodes_between_nodes(node1, node2)

        pass

    def check_swap_nodes_seq(self, node1, node2, insert=False):
        """
        checks if a changes is benefitial in terms of sequence violations
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
        if node2 is None:
            node2 = nd.get_descendant(node1.get_tree_root(), which='last')
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

    def try_change_node(self, node, candidates, tolerance=0):
        did_change = False
        for node2 in candidates:
            balance1 = self.check_swap_nodes_seq(node, node2, insert=True)
            balance2 = self.check_swap_nodes_seq(node, node2, insert=False)
            insert = True
            balance = balance1
            if balance2 > balance1:
                insert = False
                balance = balance2
            if balance <= tolerance:
                continue
            result = self.swap_nodes_same_level(node, node2, insert=insert)
            if result:
                did_change = True
        return did_change


if __name__ == "__main__":
    # TODO: if a parent has only one child: collapse
    # TODO: check defects always together with sequence
    import pprint as pp
    e = '201804271903/'
    # e = '201805020012/'
    path = pm.PATHS['experiments'] + e
    solution = sol.Solution.from_io_files(path=path)

    self = ImproveHeuristic(solution)
    node1 = self.trees[1].search_nodes(name=78)[0]
    self.get_nodes_between_nodes_in_tree(node1=node1)
    # self.draw(pos=0, attributes=['name', 'WIDTH', 'HEIGHT', 'TYPE'])
    # self.draw(pos=0, attributes=['name', 'X', 'Y', 'TYPE'])
    # self.draw(pos=2, attributes=['name', 'X', 'WIDTH', 'TYPE'])
    for i in range(4):
        self.move_item_inside_node()
        self.exchange_level1_nodes_defects()
    #
    # defects = self.check_defects()
    # previous = self.get_previous_nodes()

    # seq = tl.TupList(self.check_sequence())
    # prec = seq.to_dict(result_col=1)
    tolerance = 0
    prec = self.check_sequence().to_dict(result_col=1)
    i = count = 0
    while count < 1000 and i < len(prec):
        count += 1
        node = [*prec][i]
        change = False
        node_level1 = nd.find_ancestor_level(node, 1)
        candidates = [nd.find_ancestor_level(p, 1) for p in prec[node]]
        change |= self.try_change_node(node_level1, candidates)
        # we're desperated: why not try with all nodes level1?
        candidates = [ch for ch in node_level1.get_sisters()]
        change |= self.try_change_node(node_level1, candidates)
        # we're desperated: why not try with all nodes level1?
        candidates = [ch for tree in self.trees for ch in tree.get_children() if ch != node_level1]
        change |= self.try_change_node(node_level1, candidates)
        i += 1
        if change:
            # made a swap: recalculate
            prec = self.check_sequence().to_dict(result_col=1)
            i = 0


    # succ = seq.to_dict(result_col=1)

    # len(prec[node])
    # len(previous[node])
    #
    # node.PLATE_ID
    # max(succ.to_lendict().items(), key=lambda x: x[1])

    # node1, node2, node3 = heur.trees[0].children[:3]
    # result = heur.swap_siblings(node1, node3)
    self.graph_solution(path, name="edited", dpi=50)


    # self.graph_solution(path, show=True, pos=3, dpi=30, fontsize=10)
    # heur.get_pieces_by_type(by_plate=True)
    # prev_nodes = self.get_previous_nodes()
    pass