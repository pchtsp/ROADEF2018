import package.solution as sol
import package.tuplist as tl
import package.superdict as sd
import package.params as pm
import numpy as np
import pandas as pd
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

    def flatten_stacks(self, in_list=False):
        """
        :param in_list: return an size-ordered list of plates instead of dictionary?
        :return: dictionary indexed by piece and with a tuple
        of two dimentions. The first one is always smaller.
        """
        pieces = {k: (v['WIDTH_ITEM'], v['LENGTH_ITEM'])
                 for k, v in self.input_data['batch'].items()}
        for k, v in pieces.items():
            if v[0] > v[1]:
                pieces[k] = v[1], v[0]
        if in_list:
            pieces = sorted(pieces.values())
            return tl.TupList(pieces)
        return sd.SuperDict.from_dict(pieces)

    def get_smallest_items(self):
        pieces = self.flatten_stacks(in_list=True)
        min_length = pieces[1][1] + 1
        filter_out_pos = set()
        for pos, i in enumerate(pieces):
            if i[1] >= min_length:
                filter_out_pos.add(pos)
            else:
                min_length = i[1]
        return [p for pos, p in enumerate(pieces) if pos not in filter_out_pos]

    def plate_generation(self, max_iterations=None):
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
        non_processed = [(plate0, cut_level, self.flatten_stacks(in_list=True))]
        smallest_items = self.get_smallest_items()
        max_iter = 0
        while len(non_processed) > 0 and (max_iterations is None or max_iter < max_iterations):
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
                    # print('from plate {}: {} and {} at cut_level= {}'.format(j, j1, j2, next_level))
                    for k in [j1, j2]:
                        if not self.check_plate_can_fit_some_item(k, smallest_items):
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
        max_size_cut = plate[dim]
        max_size_fixed = plate[dim2]

        if original_items is None:
            original_items = self.flatten_stacks(in_list=True)
            items_rotated = [self.rotate_plate(v) for v in original_items]
            original_items = [item for item in list(set(original_items + items_rotated))]
        # items_rotated = []

        candidates = [item for item in original_items
                      if item[dim2] <= max_size_fixed
                      and item[dim] <= max_size_cut]
        all_lengths = [item[dim] for item in candidates]
        all_lengths = np.unique(all_lengths)

        # cuts = self.get_combination_cuts(all_lengths, max_size_cut)
        cuts = self.get_combination_cuts_strict(all_lengths, max_size_cut)
        return cuts, candidates

    @staticmethod
    def get_combination_cuts(all_lengths, max_size_cut):
        # TODO: this is not efficient.
        cuts = [0]
        all_lengths.sort()
        for item in all_lengths:
            new_cuts = []
            for cut in cuts:
                size = item + cut
                if size < max_size_cut and size not in cuts and \
                    (size <= max_size_cut // 2 or (max_size_cut - size) not in cuts):
                    new_cuts.append(size)
            cuts.extend(np.unique(new_cuts))
        return cuts

    @staticmethod
    def get_combination_cuts_strict(all_lengths, max_size_cut):
        return all_lengths

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

    def plate_inside_plate(self, plate1, plate2, turn=True, both_sides=False):
        origin = {'X': 0, 'Y': 0}
        result = self.square_inside_square(
            [origin, {'X': plate1[0], 'Y': plate1[1]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=both_sides
        )
        if result or not turn:
            return result
        return self.square_inside_square(
            [origin, {'X': plate1[1], 'Y': plate1[0]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=both_sides
    )

    def check_plate_can_fit_some_item(self, plate, items):
        for item in items:
            result = self.plate_inside_plate(item, plate)
            if result == 1:
                return True
            # if it doesn't fit, try rotating it
            if not result:
                if self.plate_inside_plate(self.rotate_plate(item), plate) == 1:
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
            node_id = 1
        elif is_sibling:
            # this means this was a subsecuent cut in the same level.
            # the parent should go at the end
            child = tree.up.add_child(name=plate)
            if tree.NODE_ID is None:
                # we already signaled for deletion.
                # we delete it??
                tree.detach()
                node_id = 1
            else:
                node_id = 1
                tree.NODE_ID = None
        else:
            # this means this was a cut in the lower level or it's a leaf.
            child = tree.add_child(name=plate)
            node_id = len(child.get_tree_root().get_descendants())

        # we fill some of the nodes information (still missing TYPE, PLATE_ID, NODE_ID, PARENT)
        # the node_id is only used here to determine if we should delete a sibling or not.
        # so the number is 1 or None

        info = {
            'X': ref_pos[0]
            , 'Y': ref_pos[1]
            , 'WIDTH': plate[0]
            , 'HEIGHT': plate[1]
            , 'NODE_ID': node_id
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

    def load_solution(self, solution):
        # pp.pprint(cut_by_level)
        # cut_by_level_backup = copy.deepcopy(cut_by_level)
        cut_by_level = solution.index_by_part_of_tuple(position=3, get_list=False)
        num_trees = len([tup for tup in cut_by_level[1] if tup[0] == self.get_plate0()])
        self.trees = []
        next_node_id = 0

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
            # we also make node_ids absolute across trees
            # and parents accordingly
            for n in tree.traverse():
                n.add_features(PLATE_ID=i, TYPE=None, NODE_ID=next_node_id)
                if n.up is not None:
                    n.add_features(PARENT=n.up.NODE_ID)
                else:
                    n.add_features(PARENT=None)
                next_node_id += 1

        # here, we assign the correct TYPE:
        demand = self.search_items_in_tree(strict=False)
        if len(demand) > 0:
            print('Not all items were found in solution: {}'.format(demand))
        self.fill_node_types()
        return

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

    @staticmethod
    def export_cuts(cuts, path=pm.PATHS['results']):
        code = 0
        result = {}
        for k, v in cuts.items():
            d =\
                {'width': k[0][0]
                 ,'height': k[0][1]
                 ,'orientation': k[1]
                 ,'cut': k[2]
                 ,'level': k[3]
                 ,'quantity': v
                 }
            code += 1
            result[code] = d
        table = pd.DataFrame.from_dict(result, orient='index')
        table.to_csv(path + 'cuts.csv', index=False, sep=';')

    @staticmethod
    def import_cuts(path):
        table_dict = \
            pd.read_csv(path + 'cuts.csv', sep=';').\
            set_index(['width', 'height', 'orientation', 'cut', 'level']).\
            to_dict(orient='index')

        cuts = {}
        for k, v in table_dict.items():
            cuts[(k[0], k[1]), k[2], k[3], k[4]] = v['quantity']
        return cuts


if __name__ == "__main__":
    pass