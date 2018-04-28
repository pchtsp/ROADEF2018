import package.solution as sol
import package.params as pm
import copy

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
        return

    def move_item_inside_node(self):
        # This should keep the cuts the same but
        # maybe put the "waste"
        defects = heur.check_defects()
        for defect in defects:
            node = defect[0]
            for sibling in node.get_sisters():
                if sibling.TYPE != -1:
                    continue
                if self.defects_in_node(sibling):
                    continue
                print('Found! Swapping {} and {}'.format(node.name, sibling.name))
                result = self.swap_siblings(node, sibling)
                break
        return

    def exchange_level1_nodes_defects(self):
        # TODO: search for the thinest slice!
        # TODO: deal with several defects
        # TODO: try to exchange plates from different bins
        defects = heur.check_defects()
        for defect in defects:
            node_level1 = heur.find_ancestor_level(defect[0], 1)
            actual_defect = defect[1]
            dist_x = actual_defect['X'] + actual_defect['WIDTH'] - node_level1.X
            dist_y = min(actual_defect['Y'], node_level1.HEIGHT - actual_defect['Y'])
            for sibling in node_level1.get_sisters():
                if heur.defects_in_node(sibling):
                    continue
                candidates = [n for n in sibling.children if n.TYPE == -1 and n.HEIGHT >= dist_y]
                if len(candidates) == 0:
                    continue
                print('Found! Swapping {} and {}'.format(node_level1.name, sibling.name))
                result = heur.swap_siblings(node_level1, sibling)
                break

    def exchange_level1_nodes_seq(self):
        pass
        # seq = heur.check_sequence()
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

    def swap_neighbors(self, node1, node2):
        parent = node1.up
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
        node2_movement = node1pos - node2pos
        self.move_node(node2, node2_movement, axis=axis)
        node1_movement = getattr(node2, dim)
        self.move_node(node1, node1_movement, axis=axis)

        # finally, I need to change the order in the tree:
        node1_order = parent.children.index(node1)
        node2_order = parent.children.index(node2)
        parent.children[node1_order] = node2
        parent.children[node2_order] = node1
        return True

    def swap_siblings(self, node1, node2):
        # check if trying to change the same node!
        if node1 == node2:
            return True
        parent = node1.up
        assert parent == node2.up, \
            '{} and {} are not siblings'.format(node1, node2)
        node1_order = parent.children.index(node1)
        node2_order = parent.children.index(node2)
        if node1_order > node2_order:
            node1, node2 = node2, node1
            node1_order, node2_order = node2_order, node1_order
        if abs(node1_order - node2_order) == 1:
            self.swap_neighbors(node1, node2)
            return True
        all_siblings = node1.up.children
        # intermediates = [node1_order:node2_order+1]
        neighbor = node1_order + 1
        while neighbor <= node2_order:
            self.swap_neighbors(node1, all_siblings[neighbor])
            neighbor += 1
        node2_order = parent.children.index(node2)
        neighbor = node2_order - 1
        while neighbor >= 0:
            self.swap_neighbors(node2, all_siblings[neighbor])
            neighbor -= 1
        return True

    def clean_empty_cuts(self):
        for tree in self.trees:
            for v in tree.traverse():
                children = v.children
                if len(children) != 2:
                    continue
                if children[0].TYPE < 0 or\
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
                    self.swap_siblings(c, children[pos])
                # for c in candidates:
                #     if children.index(c) <
                #     self.swap_siblings(c, children[pos])
                #     pos -= 1
                candidates = children[min_pos:]
                while len(candidates) > 1:
                    self.join_neighbors(candidates[0], candidates[1])
                    candidates.pop(1)
        return True

    def order_children(self):
        for tree in self.trees:
            for v in tree.traverse():
                if not len(v.children):
                    continue
                # we inverse this condition because we're dealing
                # with the children
                if not v.CUT % 2:  # cuts 1 and 3
                    axis = 'X'
                else:  # cut 2 and 4
                    axis = 'Y'
                v.children.sort(key=lambda x: getattr(x, axis))

    def cut_waste_with_defects(self):

        return True


if __name__ == "__main__":
    import pprint as pp
    e = '201804271903/'
    path = pm.PATHS['experiments'] + e
    solution = sol.Solution.from_io_files(path=path)

    heur = ImproveHeuristic(solution)
    # heur.draw(pos=0, attributes=['name', 'WIDTH', 'HEIGHT', 'TYPE'])
    heur.order_children()
    heur.clean_empty_cuts()
    heur.join_blanks()
    # heur.draw(pos=0, attributes=['name', 'WIDTH', 'HEIGHT', 'TYPE'])
    for i in range(4):
        heur.move_item_inside_node()
        heur.exchange_level1_nodes_defects()

    defects = heur.check_defects()
    seq = heur.check_sequence()
    # node1, node2, node3 = heur.trees[0].children[:3]
    # result = heur.swap_siblings(node1, node3)
    heur.graph_solution(path, name="edited")
    # heur.get_pieces_by_type(by_plate=True)
    prev_nodes = heur.get_previous_nodes()
    pass