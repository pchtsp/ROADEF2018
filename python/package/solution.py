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


class Solution(inst.Instance):

    def __init__(self, input_data, solution_data):
        super().__init__(input_data)
        self.sol_data = solution_data
        parent_child = [(int(v['PARENT']), int(k), 1) for k, v in solution_data.items()
                        if not math.isnan(v['PARENT'])]
        if len(parent_child) > 0:
            self.tree = ete3.Tree.from_parent_child_table(parent_child)
            # add info about each node
            for node in self.tree.traverse():
                for k, v in solution_data[node.name].items():
                    node.add_feature(k, v)
        else:
            self.tree = ete3.Tree(name=0)

    def draw(self):
        print(self.tree)
        return

    def draw_interactive(self):
        return self.tree.show()

    @classmethod
    def from_files(cls, case_name, path=pm.PATHS['checker_data']):
        return cls(di.get_model_data(case_name, path),
                   di.get_model_solution(case_name, path))

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

    def square_inside_square(self, square1, square2):
        if self.point_in_square(square1[0], square2, strict=False):
            if self.point_in_square(square1[1], square2, strict=False):
                return 1
        if self.point_in_square(square2[0], square1, strict=False):
            if self.point_in_square(square2[1], square1, strict=False):
                return 2
        return False

    @staticmethod
    def piece_to_square(piece):
        """
        Reformats a piece to a list of two points
        :param piece: a leaf from ete3 tree.
        :return: list of two points {'X': 1, 'Y': 1}
        """
        return [{'X': piece.X, 'Y': piece.Y},
                {'X': piece.X + piece.WIDTH, 'Y': piece.Y + piece.HEIGHT}]

    def get_cuts(self):
        # for each node, we get the cut that made it.
        # there's always one of the children that has no cut
        # each cut wll have as property the first and second piece

        pass

    def get_pieces(self, by_plate=False):
        """
        Gets the solution pieces indexed by the NODE_ID.
        :param by_plate: when active it returns a dictionary indexed
        by the plates. So it's {PLATE_0: {0: leaf0,  1: leaf1}, PLATE_1: {}}
        :return: {0: leaf0,  1: leaf1}
        """
        leaves = self.tree.get_leaves()
        if not by_plate:
            return {int(leaf.TYPE): leaf for leaf in leaves if leaf.TYPE >= 0}
        leaves_by_plate = {leaf.PLATE_ID: {} for leaf in leaves}
        for leaf in leaves:
            leaves_by_plate[leaf.PLATE_ID][leaf.TYPE] = leaf
        return leaves_by_plate

    def check_all(self):
        return True

    def check_overlapping(self):
        plate_leaves = self.get_pieces(by_plate=True)
        overlapped = []
        for plate, leaves in plate_leaves:
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
        code_leaf = self.get_pieces()
        wrong_order = []
        for stack, nodes_dict in self.get_batch_per_stack().items():
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
        plate_cuts = self.get_pieces(by_plate=True)
        pieces_with_defects = []
        for plate, defects_dict in self.get_defects_per_plate():
            for k, piece in plate_cuts[plate].items():
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
        for leaf in self.tree.iter_leaves():
            if leaf.TYPE not in demand:
                continue
            if leaf.WIDTH == demand[leaf.TYPE]['WIDTH_ITEM'] and\
            leaf.HEIGHT == demand[leaf.TYPE]['LENGTH_ITEM']:
                produced.append(leaf.name)
        return np.setdiff1d([*demand], produced)

    def graph_solution(self):
        colors = pal.colorbrewer.diverging.BrBG_5.hex_colors
        width, height = self.get_param('plate_width'), self.get_param('plate_height')
        for plate, leafs in self    .get_pieces(True).items():
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
                        facecolor=colors[pos],
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
            fig1.savefig('rect1.png', dpi=90, bbox_inches='tight')


if __name__ == "__main__":
    input_data = di.get_model_data('A0', path=pm.PATHS['checker_data'])
    solution_data = di.get_model_solution('A0')
    self = Solution(input_data, solution_data)
    self.graph_solution()



    # Useful functions:

    # tree.search_nodes(name=0)[0]
    # tree & 0
    # help(tree)
    # print(tree)
    # tree.show()
