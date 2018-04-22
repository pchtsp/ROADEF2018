import package.data_input as di
import ete3
import math
import package.instance as inst
import package.params as pm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import palettable as pal
import pprint as pp
import pandas as pd
import package.auxiliar as aux
import os


class Solution(inst.Instance):

    def __init__(self, input_data, solution_data):
        super().__init__(input_data)
        # self.sol_data = solution_data
        if len(solution_data) == 0:
            self.trees = []
            return

        self.trees = []
        data_byPlate = solution_data.index_by_property('PLATE_ID', get_list=True)
        for plate in data_byPlate:
            tree = self.get_tree_from_solution_data(plate)
            self.trees.append(tree)

    @staticmethod
    def get_tree_from_solution_data(solution_data):
        parent_child = [(int(v['PARENT']), int(k), 1) for k, v in solution_data.items()
                        if not math.isnan(v['PARENT'])]
        for p, c, d in parent_child:
            if p == c:
                raise ValueError('parent cannot be the same node!')
        if len(parent_child) == 0:
            return ete3.Tree(name=0)

        tree = ete3.Tree.from_parent_child_table(parent_child)
        # add info about each node
        for node in tree.traverse():
            for k, v in solution_data[node.name].items():
                if math.isnan(v):
                    node.add_feature(k, v)
                else:
                    node.add_feature(k, int(v))
        return tree

    def draw(self, attributes=None, pos=0):
        if attributes is None:
            attributes = ['NODE_ID']
        print(self.trees[pos].get_ascii(show_internal=True, attributes=attributes))
        return

    def draw_interactive(self, pos=0):
        return self.trees[pos].show()

    @staticmethod
    def search_case_in_options(path):
        try:
            options = di.load_data(path + 'options.json')
        except FileNotFoundError:
            return None
        else:
            return options.get('case_name', None)

    @classmethod
    def from_io_files(cls, case_name=None, path=pm.PATHS['checker_data']):
        if case_name is None:
            case_name = cls.search_case_in_options(path)
        if case_name is None:
            raise ImportError('case_name is None and options.json is not available')
        input_data = di.get_model_data(case_name, path)
        solution = di.get_model_solution(case_name, path)
        return cls(input_data, solution)

    @classmethod
    def from_input_files(cls, case_name=None, path=pm.PATHS['data']):
        if case_name is None:
            case_name = cls.search_case_in_options(path)
        if case_name is None:
            raise ImportError('case_name is None and options.json is not available')
        return cls(di.get_model_data(case_name, path), {})

    @staticmethod
    def point_in_square(point, square, strict=True):
        """
        :param point: dict with X and Y values
        :param square: a list of two points that define a square.
        **important**: first point of square is bottom left.
        :param strict: does not count borders
        :return: True if point is inside square (with <)
        """
        if strict:
            return \
                square[0]['X'] < point['X'] < square[1]['X'] and\
                square[0]['Y'] < point['Y'] < square[1]['Y']
        else:
            return \
                square[0]['X'] <= point['X'] <= square[1]['X'] and\
                square[0]['Y'] <= point['Y'] <= square[1]['Y']

    def square_inside_square(self, square1, square2, both_sides=True):
        """
        Tests if square1 is inside square2.
        :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
        :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
        :param both_sides: if true, alse see if square2 is inside square1
        :return: number of square inside the other. or 0 if not
        """
        if self.point_in_square(square1[0], square2, strict=False):
            if self.point_in_square(square1[1], square2, strict=False):
                return 1
        if not both_sides:
            return 0
        if self.point_in_square(square2[0], square1, strict=False):
            if self.point_in_square(square2[1], square1, strict=False):
                return 2
        return 0

    @staticmethod
    def piece_to_square(piece):
        """
        Reformats a piece to a list of two points
        :param piece: a leaf from ete3 tree.
        :return: list of two points {'X': 1, 'Y': 1}
        """
        return [{'X': piece.X, 'Y': piece.Y},
                {'X': piece.X + piece.WIDTH, 'Y': piece.Y + piece.HEIGHT}]

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

    def get_cuts(self):
        # for each node, we get the cut that made it.
        # there's always one of the children that has no cut
        # each cut wll have as property the first and second piece

        pass

    def get_pieces_by_type(self, by_plate=False, pos=None, filter_type=True):
        """
        Gets the solution pieces indexed by the TYPE.
        :param by_plate: when active it returns a dictionary indexed
        by the plates. So it's {PLATE_0: {0: leaf0,  1: leaf1}, PLATE_1: {}}
        :return: {0: leaf0,  1: leaf1}
        """
        min_type = -5
        if filter_type:
            min_type = 0
        if pos is None:
            leaves = [leaf for tree in self.trees
                      for leaf in tree.get_leaves() if leaf.TYPE >= min_type]
        else:
            leaves = [leaf for leaf in self.trees[0].get_leaves() if leaf.TYPE >= min_type]

        if not by_plate:
            return {int(leaf.TYPE): leaf for leaf in leaves}

        leaves_by_plate = {tree.PLATE_ID: {} for tree in self.trees}
        for leaf in leaves:
            leaves_by_plate[leaf.PLATE_ID][int(leaf.TYPE)] = leaf
        if pos is None:
            return leaves_by_plate
        return leaves_by_plate[pos]

    def get_plate_production(self):
        return [(leaf.WIDTH, leaf.HEIGHT) for tree in self.trees for leaf in tree.get_leaves()]

    def check_all(self):
        func_list = {
            'overlapping': self.check_overlapping
            , 'sequence': self.check_sequence
            , 'defects': self.check_no_defects
            , 'demand': self.check_demand_satisfied
        }
        result = {k: v() for k, v in func_list.items()}
        return {k: v for k, v in result.items() if len(v) > 0}

    def check_overlapping(self):
        plate_leaves = self.get_pieces_by_type(by_plate=True)
        overlapped = []
        for plate, leaves in plate_leaves.items():
            for k1, leaf1 in leaves.items():
                point1 = {'X': leaf1.X, 'Y': leaf1.Y}
                point2 = {'X': leaf1.X + leaf1.WIDTH, 'Y': leaf1.Y + leaf1.HEIGHT}
                for k2, leaf2 in leaves.items():
                    square = self.piece_to_square(leaf2)
                    if self.point_in_square(point1, square) or \
                            self.point_in_square(point2, square):
                        overlapped.append((leaf1, leaf2))
        return overlapped

    def parent_of_children(self, parent):
        """
        We want to check if each node is inside its parent.
        Also: if the parent sums the area of all children.
        :return:
        """

        # TODO
        pass

    def check_sequence(self):
        code_leaf = self.get_pieces_by_type()
        wrong_order = []
        for stack, nodes_dict in self.get_items_per_stack().items():
            last_node = last_cut = last_plate = 0
            nodes = sorted([*nodes_dict.items()], key=lambda x: x[1]['SEQUENCE'])
            for k, v in nodes:
                if k not in code_leaf:
                    continue
                if code_leaf[k].PLATE_ID < last_plate:
                    wrong_order.append((last_node, k))
                if code_leaf[k].PLATE_ID == last_plate and \
                    code_leaf[k].CUT < last_cut:
                    wrong_order.append((last_node, k))
                last_node = k
                last_cut = code_leaf[k].CUT
                last_plate = code_leaf[k].PLATE_ID
        return wrong_order

    def check_no_defects(self):
        plate_cuts = self.get_pieces_by_type(by_plate=True)
        pieces_with_defects = []
        for plate, piece_dict in plate_cuts.items():
            defects_dict = self.get_defects_per_plate(plate)
            for k, piece in piece_dict.items():
                square = self.piece_to_square(piece)
                for k, defect in defects_dict.items():
                    if self.point_in_square(point=defect, square=square):
                        pieces_with_defects.append((piece, defect))
        return pieces_with_defects

    def check_siblings(self):
        # siblings must share a dimension of size and a point dimension
        # for example: W and X or H and Y
        pass

    def check_cuts_number(self):
        # TODO: this should check if there are no more than 3 cuts.
        return True

    def check_cuts_guillotine(self):
        # TODO: this should check that the cuts are guillotine-type
        return True

    def check_distance_limits(self):
        # TODO: check the min waste, min size of piece.
        pass

    def check_pieces_fit_in_plate(self):
        # TODO: depending on the size, pieces can enter or not in the plate
        pass

    def check_demand_satisfied(self):
        demand = self.input_data['batch']
        produced = []
        pieces = self.get_pieces_by_type()
        for k, leaf in pieces.items():
            item = demand.get(k, None)
            # there's no demand for this item code
            if item is None:
                continue
            plate1 = item['WIDTH_ITEM'], item['LENGTH_ITEM']
            plate2 = leaf.WIDTH, leaf.HEIGHT
            if self.plate_inside_plate(plate1, plate2):
                produced.append(k)
        return np.setdiff1d([*demand], produced)

    def graph_solution(self):
        colors = pal.colorbrewer.diverging.BrBG_5.hex_colors
        width, height = self.get_param('widthPlates'), self.get_param('heightPlates')
        for plate, leafs in self.get_pieces_by_type(by_plate=True).items():
            fig1 = plt.figure(figsize=(width/100, height/100))
            ax1 = fig1.add_subplot(111, aspect='equal')
            ax1.set_xlim([0, width])
            ax1.set_ylim([0, height])
            ax1.tick_params(axis='both', which='major', labelsize=50)
            for pos, leaf in enumerate(leafs.values()):
                ax1.add_patch(
                    patches.Rectangle(
                        (leaf.X, leaf.Y),  # (x,y)
                        leaf.WIDTH,  # width
                        leaf.HEIGHT,  # height
                        facecolor=colors[pos % len(colors)],
                        linewidth=3
                    )
                )
                ax1.text(leaf.X + leaf.WIDTH/2, leaf.Y + leaf.HEIGHT/2,
                         '{} x {}'.format(leaf.WIDTH, leaf.HEIGHT),
                         horizontalalignment='center', fontsize=30)
            for defect in self.get_defects_per_plate(plate).values():
                ax1.add_patch(
                    patches.Rectangle(
                        (defect['X'], defect['Y']),  # (x,y)
                        defect['WIDTH'],  # width
                        defect['HEIGHT'],  # height
                        facecolor='red',
                        linewidth=10
                    )
                )
            fig1.savefig('rect1_{}.png'.format(plate), dpi=90, bbox_inches='tight')
            plt.close(fig1)

    def export_solution(self, path=pm.PATHS['results'] + aux.get_timestamp(), prefix=''):
        if not os.path.exists(path):
            os.mkdir(path)
        result = {}
        k = 0
        for tree in self.trees:
            for v in tree.traverse():
                d = \
                    {'X': v.X,
                     'Y': v.Y,
                     'NODE_ID': v.NODE_ID,
                     'PLATE_ID': v.PLATE_ID,
                     'CUT': v.CUT,
                     'PARENT': v.PARENT,
                     'TYPE': v.TYPE,
                     'WIDTH': v.WIDTH,
                     'HEIGHT': v.HEIGHT,
                 }
                k += 1
                result[k] = d
        table = pd.DataFrame.from_dict(result, orient='index')
        table.to_csv(path + '{}solution.csv'.format(prefix), index=False, sep=';')
        return True


if __name__ == "__main__":
    input_data = di.get_model_data('A0', path=pm.PATHS['checker_data'])
    solution_data = di.get_model_solution('A0')
    solution_data_byPlate = solution_data.index_by_property('PLATE_ID', get_list=True)
    self = Solution(input_data, solution_data)
    self.graph_solution()
    self.draw()
    # pp.pprint(self.sol_data)
    # self.trees = []

    # Useful functions:

    # tree.search_nodes(name=0)[0]
    # tree & 0
    # help(tree)
    # print(tree)
    # tree.show()
