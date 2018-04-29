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

    def swap_nodes_same_level(self, node1, node2):
        # for now, they need to be level=1
        # or siblings.
        # meaning: sharing the dimention we are not changing.
        # we do not want to make cuts for the moment.
        assert node1.CUT == node2.CUT, \
            'nodes {} and {} need to have the same level'.format(node1.name, node2.name)

        siblings = node1.up == node2.up
        axis, dim = nd.get_orientation_from_cut(node1)
        _, inv_dim = nd.get_orientation_from_cut(node1, inv=True)
        assert getattr(node1, inv_dim) == getattr(node2, inv_dim), \
            'nodes {} and {} need to have the same fixed dim'.format(node1.name, node2.name)

        dif_length = getattr(node1, dim) - getattr(node2, dim)
        # check there's space in smaller piece waste to eat.
        if dif_length < 0:
            # we check node1's waste:
            # node1 is always bigger
            dif_length = -dif_length
            node1, node2 = node2, node1
            # else:
            # same width: no need to change:
        if not siblings:
            # if sibling: no need of extra space
            waste = nd.find_waste_sibling(node2)
            if waste is None and dif_length:
                # no waste in node2. cannot do the change
                return False
            if waste.WIDTH < dif_length:
                # waste is not big enough
                return False
        # 1. move everything at the right/ top of each node.
        print('Found! Change between nodes {} and {}'.format(node1.name, node2.name))
        # TODO: if siblings: maybe only move things between the two positions
        parent1 = node1.up
        parent2 = node2.up
        ch_pos1 = parent1.children.index(node1)
        ch_pos2 = parent2.children.index(node2)
        # TODO: if I want to make space: I need to see *where* is the waste
        # it is not necessary in one end
        for sib in parent1.children[ch_pos1+1:]:
            nd.move_node(node=sib, movement=-dif_length, axis=axis)
        for sib in parent2.children[ch_pos2+1:]:
            nd.move_node(node=sib, movement=dif_length, axis=axis)
        # 2. exchange positions and plates.
        pos1 = getattr(node1, axis)
        if not siblings:
            plate1 = node1.PLATE_ID
            nd.change_feature(node1, 'PLATE_ID', node2.PLATE_ID)
            nd.change_feature(node2, 'PLATE_ID', plate1)
        nd.change_feature(node1, axis, getattr(node2, axis))
        nd.change_feature(node2, axis, pos1)
        # 3. change parents:
        if not siblings:
            node1.detach()
            node2.detach()
            parent1.add_child(node2)
            parent2.add_child(node1)
        nd.order_children(parent1)
        if not siblings:
            nd.order_children(parent2)
        return True

    def clean_empty_cuts(self):
        for tree in self.trees:
            for v in tree.traverse():
                children = v.children
                if len(children) != 2:
                    continue
                if children[0].TYPE < 0 or \
                        (children[1].WIDTH > 0 and
                                 children[1].HEIGHT > 0):
                    continue
                v.TYPE = children[0].TYPE
                v.remove_child(children[0])
                v.remove_child(children[0])

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

    def check_swap_nodes_seq(self, node1, node2):
        # convention: node1 is always before node2:
        if nd.get_node_pos(node1) > nd.get_node_pos(node2):
            node1, node2 = node2, node1
        precedence = self.get_previous_nodes()
        precedence_inv = precedence.list_reverse()
        # get all leaves in node1 and node2
        items1 = nd.get_node_leaves(node1)
        items2 = nd.get_node_leaves(node2)
        # get all leaves between the two nodes
        nodes = self.get_nodes_between_nodes(node1, node2)
        items = set(leaf for node in nodes for leaf in nd.get_node_leaves(node))
        # for each leaf in node2:
            # because node2 is "going back":
            # if I find nodes that precede it: bad
            # if I find nodes that follow it: good
        negative = set()
        positive = set()
        for item in items2:
            negative |= items & set(precedence.get(item, set()))
            positive |= items & set(precedence_inv.get(item, set()))
        # for each leaf in node1:
            # because node1 is "going forward":
            # if I find nodes that precede it: good
            # if I find nodes that follow it: bad
        for item in items1:
            negative |= items & set(precedence_inv.get(item, set()))
            positive |= items & set(precedence.get(item, set()))
        # for now, I'll return the balance:
        return len(positive) - len(negative)

    def get_nodes_between_nodes(self, node1, node2):
        # the level of the returned nodes may vary.
        # it will always be the biggest possible
        plate1, pos1 = nd.get_node_pos(node1)
        plate2, pos2 = nd.get_node_pos(node2)
        if (plate1, pos1) > (plate2, pos2):
            node1, node2 = node2, node1
            plate1, pos1, plate2, pos2 = plate2, pos2, plate1, pos1
        parent1 = node1.up
        parent2 = node2.up
        siblings = parent1 == parent2
        if siblings:
            return node1.up.children[pos1+1:pos2]
        nodes = node1.up.children[pos1+1:]
        nodes += node2.up.children[:pos2]
        i = plate1 + 1
        while i < plate2:
            nodes += self.trees[i]
            i += 1
        return nodes

    def order_all_children(self):
        for tree in self.trees:
            nd.order_children(tree)

    def cut_waste_with_defects(self):

        return True


if __name__ == "__main__":
    import pprint as pp
    e = '201804271903/'
    path = pm.PATHS['experiments'] + e
    solution = sol.Solution.from_io_files(path=path)

    self = ImproveHeuristic(solution)
    # heur.draw(pos=0, attributes=['name', 'WIDTH', 'HEIGHT', 'TYPE'])
    # heur.draw(pos=0, attributes=['name', 'WIDTH', 'HEIGHT', 'TYPE'])
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
        # node, _ = max(prec.to_lendict().items(), key=lambda x: x[1])
        node_level1 = nd.find_ancestor_level(node, 1)
        prec_lv1 = [nd.find_ancestor_level(p, 1) for p in prec[node]]
        # prec_lv1.sort(key=lambda x: x.PLATE_ID)
        node1 = node_level1
        result = False
        for node2 in prec_lv1:
            # we need to be sure we want to make this change...
            balance = self.check_swap_nodes_seq(node1, node2)
            if balance <= tolerance:
                continue
            result = self.swap_nodes_same_level(node1, node2)
            if result:
                break
        if result:
            # made a swap: recalculate
            prec = self.check_sequence().to_dict(result_col=1)
            i = 0
            continue
        # we're desperated: why not try with all nodes level1?
        nodes_lv1 = [ch for tree in self.trees for ch in tree.get_children() if ch != node1]
        result = False
        for node2 in nodes_lv1:
            # we need to be sure we want to make this change...
            balance = self.check_swap_nodes_seq(node1, node2)
            if balance <= tolerance:
                continue
            result = self.swap_nodes_same_level(node1, node2)
            if result:
                break
        if result:
            # made a swap: recalculate
            prec = self.check_sequence().to_dict(result_col=1)
            i = 0
            continue
        i += 1

    # succ = seq.to_dict(result_col=1)

    # len(prec[node])
    # len(previous[node])
    #
    # node.PLATE_ID
    # max(succ.to_lendict().items(), key=lambda x: x[1])

    # node1, node2, node3 = heur.trees[0].children[:3]
    # result = heur.swap_siblings(node1, node3)
    self.graph_solution(path, name="edited")
    # heur.get_pieces_by_type(by_plate=True)
    # prev_nodes = self.get_previous_nodes()
    pass