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
import package.superdict as sd
import package.tuplist as tl
import package.nodes as nd


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
            # this means we have a one-node tree:
            name = [*solution_data.keys()][0]
            tree = ete3.Tree(name=name)
        else:
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
    def point_in_square(point, square, strict=True, lag=None):
        """
        :param point: dict with X and Y values
        :param square: a list of two points that define a square.
        **important**: first point of square is bottom left.
        :param strict: does not count being in the borders as being inside
        :param lag: move the node in some distance
        :return: True if point is inside square (with <)
        """
        if lag is None:
            lag = {'X': 0, 'Y': 0}
        if strict:
            return \
                square[0]['X'] < point['X'] + lag['X'] < square[1]['X'] and\
                square[0]['Y'] < point['Y'] + lag['Y'] < square[1]['Y']
        else:
            return \
                square[0]['X'] <= point['X'] + lag['X'] <= square[1]['X'] and\
                square[0]['Y'] <= point['Y'] + lag['Y'] <= square[1]['Y']

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

    def square_intersects_square(self, square1, square2):
        """
        Tests if some point in square1 is inside square2 (or viceversa).
        :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
        :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
        :return: True if both squares share some (smaller) area
        """
        for point in square1:
            if self.point_in_square(point, square2, strict=True):
                return True
        for point in square2:
            if self.point_in_square(point, square1, strict=True):
                return True
        return False

    @staticmethod
    def defect_to_square(defect):
        """
        Reformats a defect to a list of two points
        :param defect: a dict.
        :return: list of two points {'X': 1, 'Y': 1}
        """
        return [{'X': defect['X'], 'Y': defect['Y']},
                {'X': defect['X'] + defect['WIDTH'], 'Y': defect['Y'] + defect['HEIGHT']}]

    def piece_inside_piece(self, piece1, piece2, **kwargs):
        square1 = nd.node_to_square(piece1)
        square2 = nd.node_to_square(piece2)
        return self.square_inside_square(square1, square2, **kwargs)

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

    def get_pieces_by_type(self, by_plate=False, pos=None, min_type=0):
        """
        Gets the solution pieces indexed by the TYPE.
        :param by_plate: when active it returns a dictionary indexed
        by the plates. So it's {PLATE_0: {0: leaf0,  1: leaf1}, PLATE_1: {}}
        :param pos: get an individual plate marked by pos
        :param filter_type: if True: gets only demanded items. If not: returns all plates
        :return: {0: leaf0,  1: leaf1}
        """
        if pos is None:
            leaves = [leaf for tree in self.trees
                      for leaf in nd.get_node_leaves(tree, min_type)]
        else:
            leaves = [leaf for leaf in nd.get_node_leaves(self.trees[pos], min_type)]

        if not by_plate:
            return {int(leaf.TYPE): leaf for leaf in leaves}

        leaves_by_plate = sd.SuperDict({tree.PLATE_ID: {} for tree in self.trees})
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
            , 'defects': self.check_defects
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
                    square = nd.node_to_square(leaf2)
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

    def get_previous_nodes(self):
        prev_items = self.get_previous_items()
        code_leaf = self.get_pieces_by_type()
        prev_nodes = {}
        for k, v in prev_items.items():
            prev_nodes[code_leaf[k]] = []
            for i in v:
                prev_nodes[code_leaf[k]].append(code_leaf[i])
        return sd.SuperDict(prev_nodes)

    def check_sequence(self):
        wrong_order = []
        for node, prec_nodes in self.get_previous_nodes().items():
            for prec_node in prec_nodes:
                # prec is in a previous plate: correct
                if node.PLATE_ID > prec_node.PLATE_ID:
                    continue
                # if prec is in the next plate: very incorrect
                if node.PLATE_ID < prec_node.PLATE_ID:
                    wrong_order.append((node, prec_node))
                    continue
                # if we're here, they are in the same plate/ tree.
                # we find the common ancestor and check which node's
                # ancestors appear first
                ancestor = node.get_common_ancestor(prec_node)
                n1ancestors = set([node] + node.get_ancestors())
                n2ancestors = set([prec_node] + prec_node.get_ancestors())
                for n in ancestor.iter_descendants():
                    if n in n1ancestors:
                        wrong_order.append((node, prec_node))
                        break
                    if n in n2ancestors:
                        break
        return tl.TupList(wrong_order)

    def check_defects(self):
        plate_cuts = self.get_pieces_by_type(by_plate=True)
        pieces_with_defects = []
        for plate, piece_dict in plate_cuts.items():
            defects_dict = self.get_defects_per_plate(plate)
            for k, piece in piece_dict.items():
                defects = self.defects_in_node(piece, defects=defects_dict)
                for defect in defects:
                    pieces_with_defects.append((piece, defect))
        return pieces_with_defects

    def defects_in_node(self, node, defects=None):
        square = nd.node_to_square(node)
        defects_in_node = []
        if defects is None:
            defects = self.get_defects_per_plate(plate=node.PLATE_ID)
        for k, defect in defects.items():
            square2 = self.defect_to_square(defect)
            if self.square_intersects_square(square2, square):
                defects_in_node.append(defect)
        return defects_in_node

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

    def graph_solution(self, path="", name="rect", show=False, pos=None, dpi=90, fontsize=30, min_type=0):
        batch_data = self.get_batch()
        stack = batch_data.get_property('STACK')
        sequence = batch_data.get_property('SEQUENCE')
        colors = pal.colorbrewer.qualitative.Set3_5.hex_colors
        width, height = self.get_param('widthPlates'), self.get_param('heightPlates')
        pieces_by_type = self.get_pieces_by_type(by_plate=True)
        if pos is not None:
            pieces_by_type = pieces_by_type.filter([pos])
        for plate, leafs in pieces_by_type.items():
            fig1 = plt.figure(figsize=(width/100, height/100))
            ax1 = fig1.add_subplot(111, aspect='equal')
            ax1.set_xlim([0, width])
            ax1.set_ylim([0, height])
            ax1.tick_params(axis='both', which='major', labelsize=50)
            # graph items
            for pos, leaf in enumerate(leafs.values()):
                self.draw_leaf(ax1, leaf, stack, sequence, colors, fontsize)
            # graph wastes:
            wastes = nd.get_node_leaves(self.trees[plate], min_type=-1, max_type=-1)
            for waste in wastes:
                self.draw_leaf(ax1, waste, stack, sequence, colors, fontsize)
            # graph defects
            for defect in self.get_defects_per_plate(plate).values():
                self.draw_defect(ax1, defect)
            fig_path = os.path.join(path, '{}_{}.png'.format(name, plate))
            fig1.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            if not show:
                plt.close(fig1)

    @staticmethod
    def draw_leaf(ax1, leaf, stack, sequence, colors, fontsize):
        if leaf.TYPE in stack:
            color = colors[stack[leaf.TYPE] % len(colors)]
            edge_color = 'black'
        else:
            color = 'white'
            edge_color = 'black'
        ax1.add_patch(
            patches.Rectangle(
                (leaf.X, leaf.Y),  # (x,y)
                leaf.WIDTH,  # width
                leaf.HEIGHT,  # height
                facecolor=color,
                edgecolor=edge_color,
                linewidth=3
            )
        )
        ax1.text(leaf.X + leaf.WIDTH / 2, leaf.Y + leaf.HEIGHT / 2,
                 '{} x {}\nstack={}\npos={}\nnode={}'.
                 format(leaf.WIDTH,
                        leaf.HEIGHT,
                        stack.get(leaf.TYPE, ''),
                        sequence.get(leaf.TYPE, ''),
                        leaf.name),
                 horizontalalignment='center',
                 verticalalignment='center',
                 fontsize=fontsize)

    @staticmethod
    def draw_defect(ax1, defect):
        ax1.axhline(y=defect['Y'], color="red", ls='dashed')
        ax1.axvline(x=defect['X'], color="red", ls='dashed')
        ax1.add_patch(
            patches.Circle(
                (defect['X'], defect['Y']),  # (x,y)
                radius=20,
                # defect['WIDTH'],  # width
                # defect['HEIGHT'],  # height
                color='red',
            )
        )

    def export_solution(self, path=pm.PATHS['results'] + aux.get_timestamp(), prefix='', name='solution'):
        # TODO:
        """
        When creating the output forest:
        – Always add the header as it is done in the example (see Figure 3).
        – The trees must be given in the correct sequence of plates, i.e. first
        the nodes of plate 0, then nodes of plate 1....
        – For each plate, the root (node with CUT=0) should be given first.
        – The children of a node should be given from left to right (as repre-
        sented on the cutting pattern).
        – The nodes of trees should be given in depth first search.
        – If the last 1-cut of the forest in the last cutting pattern is a waste, it
        must be declared as a residual.
        :param path:
        :param prefix:
        :param name:
        :return:
        """
        if not os.path.exists(path):
            os.mkdir(path)
        result = {}
        for tree in self.trees:
            for v in tree.traverse():
                parent = v.PARENT
                if parent is not None:
                    parent = int(parent)
                d = nd.get_features(v)
                d['PARENT'] = parent
                result[int(v.NODE_ID)] = d
        table = pd.DataFrame.from_dict(result, orient='index')
        table.to_csv(path + '{}{}.csv'.format(prefix, name), index=False, sep=';')
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
