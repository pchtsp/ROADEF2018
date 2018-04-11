import package.solution as sol
import package.tuplist as tl
import package.superdict as sd
import package.params as pm
import numpy as np
import ete3
import pprint as pp
import copy
import package.model_1 as mdl

# TODO: increase plate size
# TODO: add defects
# TODO: add sequence
# TODO: 4th cut
# TODO: add rest of first cut to O.F.
# TODO: add tolerances to cutting options
#
# TODO: add pricing model

class Model(sol.Solution):

    def flatten_stacks(self):
        # self
        items = {k: (v['WIDTH_ITEM'], v['LENGTH_ITEM'])
                 for k, v in self.input_data['batch'].items()}
        return sd.SuperDict.from_dict(items)

    def plate_generation(self):
        """
        calculates all possible cuts to plates and the resulting plates
        In the paper, this is called "Procedure 1"
        :return:
        """
        # items = self.flatten_stacks()
        # cutting_options_tup = tl.TupList()
        # cutting_options = sd.SuperDict()
        cut_level_next_o = {
            0: pm.ORIENT_V
            , 1: pm.ORIENT_H
            , 2: pm.ORIENT_V
            , 3: pm.ORIENT_H
        }
        cutting_production = tl.TupList()  # (j, o, q, k)
        plate0 = self.get_plate0(get_dict=False)
        cut_level = 0
        plates = set()
        plates.add((plate0, cut_level))
        non_processed = [(plate0, cut_level, list(self.flatten_stacks().values()))]
        max_iter = 0
        while len(non_processed) > 0 and max_iter < 100000:
            max_iter += 1
            print('Iteration={}; Nonprocessed= {}; plates={}'.format(max_iter, len(non_processed), len(plates)))
            j, cut_level, original_items = non_processed.pop()
            next_o = cut_level_next_o[cut_level]
            for o in pm.ORIENTATIONS:
                # the first cut needs to be vertical always:
                if cut_level == 0 and next_o != o:
                    continue
                # if we have next level orientation: we add 1. If not: we add 0.
                next_level = cut_level + (next_o == o)
                cutting_options, new_original_items = self.get_cut_positions(j, o, original_items)
                for q in cutting_options[1:]:
                    # cut j
                    j1, j2 = self.cut_plate(j, o, q)
                    print('from plate {}: {} and {} at cut_level= {}'.format(j, j1, j2, next_level))
                    for k in [j1, j2]:
                        if not self.check_plate_can_fit_some_item(k):
                            # if the piece is not useful,
                            # we do not consider it a plate
                            continue
                        # here we register the tuple
                        # of the production of plates
                        cutting_production.add(j, o, q, next_level, k)
                        if next_level >= 3:
                            continue
                        if (k, next_level) in plates:
                            # if we already registered it
                            # we will not re-add it
                            continue
                        plates.add((k, next_level))
                        non_processed.append((k, next_level, new_original_items))
        return cutting_production

    def get_cut_positions(self, plate, orientation, original_items=None):
        """
        :param plate: (width, height)
        :param orientation:
        :param original_items:
        :return:
        """
        dim = 1
        dim2 = 0
        if orientation == pm.ORIENT_V:
            dim, dim2 = dim2, dim
        max_size = plate[dim]
        max_size2 = plate[dim2]

        if original_items is None:
            original_items = list(self.flatten_stacks().values())
        items_rotated = [self.rotate_plate(v) for v in original_items]
        # items_rotated = []

        total_items = list(set(original_items + items_rotated))
        total_items_dim = [item[dim] for item in total_items if item[dim2] <= max_size2]
        total_items_dim = np.unique(total_items_dim)
        cuts = [0]
        for item in total_items_dim:
            new_cuts = []
            for cut in cuts:
                size = item + cut
                if size < max_size and size not in cuts and \
                    (size <= max_size // 2 or (max_size - size) not in cuts):
                    new_cuts.append(size)
            cuts.extend(np.unique(new_cuts))
        return cuts, total_items

    @staticmethod
    def cut_plate(plate, orientation, cut):
        """
        :param plate: (width, height), i.e. (500, 100)
        :param orientation:
        :param cut:
        :return:
        """
        # cut plate at position cut with orientation
        width, height = plate
        if orientation == pm.ORIENT_V:
            if cut > width:
                raise IndexError('plate with width {} is smaller than cut {}',
                                 width, cut)
            part1 = cut, height
            part2 = width - cut, height
        else:
            if cut > height:
                raise IndexError('plate with height {} is smaller than cut {}',
                                 width, cut)
            part1 = width, cut
            part2 = width, height - cut
        return part1, part2

    @staticmethod
    def rotate_plate(plate):
        return plate[1], plate[0]

    @staticmethod
    def vars_to_tups(var, binary=True):
        # because of rounding approximations; we need to check if its bigger than half:
        if binary:
            return tl.TupList([tup for tup in var if var[tup].value() > 0.5])
        return sd.SuperDict({tup: var[tup].value() for tup in var
                             if var[tup].value() is not None and var[tup].value() > 0.5})

    def plate_inside_plate(self, plate1, plate2, turn=True):
        origin = {'X': 0, 'Y': 0}
        result = self.square_inside_square(
            [origin, {'X': plate1[0], 'Y': plate1[1]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=False
        )
        if result or not turn:
            return result
        return self.square_inside_square(
            [origin, {'X': plate1[1], 'Y': plate1[0]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=False
    )


    def check_plate_can_fit_some_item(self, plate):
        items = self.flatten_stacks()
        for key, value in items.items():
            result = self.plate_inside_plate(value, plate)
            if result == 1:
                return True
            # if it doesn't fit
            if not result:
                if self.plate_inside_plate(self.rotate_plate(value), plate) == 1:
                    return True
        return False

    @staticmethod
    def search_cut_by_plate_in_solution(cuts, plate):
        for tup in cuts:
            #  or self.rotate_plate(tup[0]) == plate
            if tup[0] == plate:
                if cuts[tup] == 1:
                    cuts.pop(tup)
                else:
                    cuts[tup] -= 1
                return tup

        return None

    @staticmethod
    def get_plates_and_position_from_cut(cut, ref_pos):
        p, o, q, l = cut
        if o == pm.ORIENT_V:
            pos1 = ref_pos
            p1 = (q, p[1])
            pos2 = ref_pos[0] + q, ref_pos[1]
            p2 = (p[0] - q, p[1])
        else:
            pos1 = ref_pos
            p1 = (p[0], q)
            pos2 = ref_pos[0], ref_pos[1] + q,
            p2 = (p[0], p[1] - q)
        return zip([p1, p2], [pos1, pos2])

    def get_tree_from_solution(self, tree, cut_by_level, plate, ref_pos, is_sibling, cut_level):
        # if the cut has a different orientation as the parent:
            # add a child to the tree. If there is no tree: create a tree.
            # visit its children with:
            # cut_level+1, orientation, plate= this y ref_pos = this
        # if the cut has the same orientation as the parent:
            # I do not add it as a child.
            # I visit the children (which are actually siblings)
            # cut_level, orientation, plate= this y ref_pos = this

        # I search cuts in my same level and the next one if available:
        next_is_sibling = True
        r_cut = self.search_cut_by_plate_in_solution(cut_by_level.get(cut_level, []), plate)
        if r_cut is None and cut_level+1 in cut_by_level:
            next_is_sibling = False
            r_cut = self.search_cut_by_plate_in_solution(cut_by_level[cut_level+1], plate)

        if tree is None:
            child = ete3.Tree(name=plate)
            parent_id = None
            node_id = 0
        elif is_sibling:
            # this means this was a subsecuent cut in the same level.
            # the parent should go at the end
            child = tree.up.add_child(name=plate)
            parent_id = tree.up.NODE_ID
            if tree.NODE_ID is None:
                # we already signaled for deletion.
                node_id = len(child.get_tree_root().get_descendants()) - 1
            else:
                node_id = tree.NODE_ID
                tree.NODE_ID = None
        else:
            # this means this was a cut in the lower level or it's a leaf.
            child = tree.add_child(name=plate)
            parent_id = tree.NODE_ID
            node_id = len(child.get_tree_root().get_descendants()) - 1

        # we fill some of the nodes information (still missing TYPE, PLATE_ID)
        # it's possible to make it more efficient to assign the NODE_ID
        # by exploiting the fact that the node's ID needs to be bigger
        # than neighbors and parent.only.

        info = {
            'X': ref_pos[0]
            , 'Y': ref_pos[1]
            , 'WIDTH': plate[0]
            , 'HEIGHT': plate[1]
            , 'NODE_ID': node_id
            , 'PARENT': parent_id
            , 'CUT': cut_level
        }
        child.add_features(**info)

        # if there is not subsecuent cut: we have arrived to a leaf. We return
        if r_cut is None:
            return

        # child = tree
        new_orientation = r_cut[1]
        new_cut_level = r_cut[3]

        children_plates = self.get_plates_and_position_from_cut(r_cut, ref_pos)
        # print(child.get_tree_root().get_ascii(show_internal=True))
        for p, pos in children_plates:
            self.get_tree_from_solution(
                tree=child
                , cut_by_level=cut_by_level
                , plate=p
                , ref_pos=pos
                , is_sibling=next_is_sibling
                , cut_level=new_cut_level
            )
        return child.get_tree_root()

    def solve(self, options):
        return mdl.solve_model(self, options)


    def load_solution(self, cut_by_level):
        # pp.pprint(cut_by_level)
        cut_by_level_backup = copy.deepcopy(cut_by_level)
        num_trees = len([tup for tup in cut_by_level[1] if tup[0] == self.get_plate0()])
        self.trees = []

        for i in range(num_trees):
            tree = self.get_tree_from_solution(
                tree=None
                , cut_by_level=cut_by_level
                , ref_pos=(0, 0)
                , plate=self.get_plate0()
                , cut_level=0
                , is_sibling=False
            )
            self.trees.append(tree)

            # we delete intermediate nodes and assign plate:
            for n in tree.traverse():
                if n.NODE_ID is None:
                    n.detach()
                    continue
                n.add_features(PLATE_ID=i, TYPE=None)

        demand = self.search_items_in_tree(strict=False)
        if len(demand) > 0:
            print('Not all items were found in solution: {}'.format(demand))
        self.fill_node_types()


    def search_items_in_tree(self, strict=True):
        trees = self.trees
        demand = self.flatten_stacks()
        for tree in trees:
            for leaf in tree.iter_leaves():
                leaf_p = leaf.WIDTH, leaf.HEIGHT
                # print(demand)
                for k, v in demand.items():
                    found = False
                    if v == leaf_p or\
                        (not strict and self.plate_inside_plate(v, leaf_p)):
                        found = True
                    elif self.rotate_plate(v) == leaf_p or\
                        (not strict and self.plate_inside_plate(self.rotate_plate(v), leaf_p)):
                        found = True
                    if found:
                        print('Match: {} and leaf: {} from tree: {}'.format(k, leaf, leaf.PLATE_ID))
                        leaf.add_features(TYPE=k)
                        demand.pop(k)
                        break

        return demand

    def fill_node_types(self):

        for tree in self.trees:
            residual_nodes = [node for node in tree.children if
                            node.is_leaf() and node.TYPE is None]
            branch_nodes = [node for node in tree.traverse() if
                            not node.is_leaf()]
            waste_nodes = [node for node in tree.iter_leaves() if
                           node.TYPE is None and node not in residual_nodes]
            for node in residual_nodes:
                node.add_features(TYPE=-3)
            for node in branch_nodes:
                node.add_features(TYPE=-2)
            for node in waste_nodes:
                node.add_features(TYPE=-1)
        return


if __name__ == "__main__":
    pass