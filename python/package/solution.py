import package.data_input as di
import ete3
import math
import package.instance as inst
import package.params as pm
import numpy as np
import matplotlib
# matplotlib.use('Qt5Agg', warn=False, force=True)
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
import package.geometry as geom


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

        self.order_all_children()

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

    def draw(self, pos=0, *attributes):
        node = self.trees[pos]
        nd.draw(node, *attributes)
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
    def from_io_files(cls, case_name=None, path=pm.PATHS['checker_data'], solutionfile="solution"):
        if case_name is None:
            case_name = cls.search_case_in_options(path)
        if case_name is None:
            raise ImportError('case_name is None and options.json is not available')
        input_data = di.get_model_data(case_name, path)
        solution = di.get_model_solution(case_name, path, filename=solutionfile)
        return cls(input_data, solution)

    @classmethod
    def from_input_files(cls, case_name=None, path=pm.PATHS['data'], **kwargs):
        if case_name is None:
            case_name = cls.search_case_in_options(path)
        if case_name is None:
            raise ImportError('case_name is None and options.json is not available')
        return cls(di.get_model_data(case_name, path), **kwargs)

    def get_cuts(self):
        # for each node, we get the cut that made it.
        # there's always one of the children that has no cut
        # each cut wll have as property the first and second piece

        pass

    def order_all_children(self):
        for tree in self.trees:
            nd.order_children(tree)

    def get_pieces_by_type(self, by_plate=False, pos=None, min_type=0, solution=None):
        """
        Gets the solution pieces indexed by the TYPE.
        :param by_plate: when active it returns a dictionary indexed
        by the plates. So it's {PLATE_0: {0: leaf0,  1: leaf1}, PLATE_1: {}}
        :param pos: get an individual plate marked by pos
        :param filter_type: if True: gets only demanded items. If not: returns all plates
        :param solution: if given it's evaluated instead of self.trees
        :return: {0: leaf0,  1: leaf1}
        """
        if solution is None:
            solution = self.trees
        if pos is None:
            leaves = [leaf for tree in solution
                      for leaf in nd.get_node_leaves(tree, min_type)]
        else:
            leaves = [leaf for leaf in nd.get_node_leaves(solution[pos], min_type)]

        if not by_plate:
            return {int(leaf.TYPE): leaf for leaf in leaves}

        leaves_by_plate = sd.SuperDict({tree.PLATE_ID: {} for tree in solution})
        for leaf in leaves:
            leaves_by_plate[leaf.PLATE_ID][int(leaf.TYPE)] = leaf
        if pos is None:
            return leaves_by_plate
        return leaves_by_plate[pos]

    def get_plate_production(self):
        return [(leaf.WIDTH, leaf.HEIGHT) for tree in self.trees for leaf in tree.get_leaves()]

    def count_errors(self):
        checks = self.check_all()
        return len([i for k, v in checks.items() for i in v])

    def check_all(self):
        func_list = {
            'overlapping': self.check_overlapping
            , 'sequence': self.check_sequence
            , 'defects': self.check_defects
            , 'demand': self.check_demand_satisfied
            , 'ch_size': self.check_nodes_fit
            , 'inside': self.check_parent_of_children
            , 'cuts': self.check_cuts_number
            , 'position': self.check_nodes_inside_jumbo
            , 'types': self.check_wrong_type
            , 'ch_order': self.check_children_order
            , 'node_size': self.check_sizes
        }
        result = {k: v() for k, v in func_list.items()}
        return {k: v for k, v in result.items() if len(v) > 0}

    def check_consistency(self):
        func_list = {
            'ch_size': self.check_nodes_fit
            , 'inside': self.check_parent_of_children
            , 'cuts': self.check_cuts_number
            , 'position': self.check_nodes_inside_jumbo
            , 'types': self.check_wrong_type
            , 'ch_order': self.check_children_order
            , 'node_size': self.check_sizes
            , 'only_child': self.check_only_child
        }
        result = {k: v() for k, v in func_list.items()}
        return {k: v for k, v in result.items() if len(v) > 0}

    def check_only_child(self):
        only_1_child = []
        for tree in self.trees:
            for n in tree.traverse():
                if len(n.get_children()) == 1 and n.CUT > 0:
                    only_1_child.append(n)
        return only_1_child

    def check_overlapping(self):
        plate_leaves = self.get_pieces_by_type(by_plate=True)
        overlapped = []
        for plate, leaves in plate_leaves.items():
            for k1, leaf1 in leaves.items():
                point1 = {'X': leaf1.X, 'Y': leaf1.Y}
                point2 = {'X': leaf1.X + leaf1.WIDTH, 'Y': leaf1.Y + leaf1.HEIGHT}
                for k2, leaf2 in leaves.items():
                    square = nd.node_to_square(leaf2)
                    if geom.point_in_square(point1, square) or \
                            geom.point_in_square(point2, square):
                        overlapped.append((leaf1, leaf2))
        return overlapped

    def get_previous_nodes(self, solution=None, type_node_dict=None):
        """
        :param solution: forest: a list of trees.
        :return:
        """
        if type_node_dict is None or solution is not None:
            type_node_dict = self.get_pieces_by_type(solution=solution)
        prev_items = self.get_previous_items()
        prev_nodes = {}
        for k, v in prev_items.items():
            prev_nodes[type_node_dict[k]] = []
            for i in v:
                prev_nodes[type_node_dict[k]].append(type_node_dict[i])
        return sd.SuperDict(prev_nodes)

    def check_sequence(self, solution=None, type_node_dict=None):
        wrong_order = []
        n_prec = self.get_previous_nodes(solution=solution, type_node_dict=type_node_dict)
        for node, prec_nodes in n_prec.items():
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
                if nd.check_node_order(node, prec_node):
                    wrong_order.append((node, prec_node))
        return tl.TupList(wrong_order)

    def check_defects(self, solution=None):
        """
        :return: [(node, defect), ()]
        """
        node_defect = self.get_nodes_defects(solution)
        return [(node, defect) for node, defect in node_defect if node.TYPE >= 0]

    def get_nodes_defects(self, solution=None):
        # TODO: I could give a level as argument to filter when searching.
        if solution is None:
            solution = self.trees
        defect_node = []
        defects_by_plate = self.get_defects_per_plate()
        for pos, tree in enumerate(solution):
            if pos not in defects_by_plate:
                continue
            for defect in defects_by_plate[pos]:
                node = nd.search_node_of_defect(tree, defect)
                assert node is not None, 'defect {} doesnt have node'.format(defect)
                defect_node.append((node, defect))
        return defect_node

    def check_space_usage(self, solution=None):
        if solution is None:
            solution = self.trees
        return sum(self.calculate_residual_plate(tree)*(pos+1)**4 for pos, tree in enumerate(solution)) / \
               (self.get_param('widthPlates') * len(solution)**4)
                # sum(nd.get_node_position_cost(n, self.get_param('widthPlates')) for tree in solution
                #     for n in nd.get_node_leaves(tree, type_options=[-1, -3])) / \
                # ((self.get_param('widthPlates') * len(solution))**2 *self.get_param('widthPlates')*self.get_param('heightPlates'))

    def calculate_residual_plate(self, node):
        waste = nd.find_waste(node, child=True)
        if waste is None:
            return 0
        return waste.WIDTH


    def check_siblings(self):
        # siblings must share a dimension of size and a point dimension
        # for example: W and X or H and Y
        pass

    def check_cuts_number(self):
        """
        This checks if the CUT property of each node really corresponds with
        the node's level.
        :return:
        """
        levels = {}
        for tree in self.trees:
            levels = {**levels, **nd.assign_cut_numbers(tree, update=False)}
        bad_cut = []
        for node, level in levels.items():
            if node.CUT != level:
                bad_cut.append((node, level))
        return bad_cut

    def check_max_cut(self):
        """
        check that the maximum achieved level is 4
        :return:
        """
        levels = {}
        for tree in self.trees:
            levels = {**levels, **nd.assign_cut_numbers(tree, update=False)}
        return [(node, level) for node, level in levels.items() if level > 4]

    def check_nodes_inside_jumbo(self):
        w, h = self.get_param('widthPlates'), self.get_param('heightPlates')
        plate_square = [{'X': 0, 'Y': 0}, {'X': w, 'Y': h}]
        bad_position = []
        for tree in self.trees:
            for node in tree.iter_leaves():
                square = nd.node_to_square(node)
                if geom.square_inside_square(square,
                                             plate_square,
                                             both_sides=False):
                    continue
                bad_position.append(node)
        return bad_position

    def check_cuts_guillotine(self):
        # TODO: this should check that the cuts are guillotine-type
        return True

    def check_distance_limits(self):
        # TODO: check the min waste, min size of piece.
        pass

    def check_wrong_type(self):
        wrong_type = []
        for tree in self.trees:
            for node in tree.traverse():
                if node.is_leaf():
                    if node.TYPE == -2:
                        wrong_type.append((node, node.TYPE))
                elif node.TYPE != -2:
                    wrong_type.append((node, node.TYPE))
        return wrong_type

    def check_nodes_fit(self):
        nodes_poblems = []
        for tree in self.trees:
            for node in tree.traverse():
                if not nd.check_children_fit(node):
                    nodes_poblems.append(node)
        return nodes_poblems

    def check_children_order(self):
        # This checks that the order of the children
        # follows the positions.
        # meaining: if children A is before B
        # it is lower or more to the left
        bad_order = []
        for tree in self.trees:
            for node in tree.traverse():
                axis, dim = nd.get_orientation_from_cut(node, inv=True)
                axis_i, dim_i = nd.get_orientation_from_cut(node)
                children = node.get_children()
                if len(children) < 2:
                    continue
                for ch1, ch2 in zip(children, children[1:]):
                    # For now, if one of the two children has dist=0
                    # I'll ignore it... but I should change this.
                    if not getattr(ch1, dim) or not getattr(ch2, dim):
                        continue
                    correct = getattr(ch1, axis) + getattr(ch1, dim) ==\
                        getattr(ch2, axis)
                    correct &= getattr(ch1, axis_i) == getattr(ch2, axis_i)
                    correct &= getattr(ch1, dim_i) == getattr(ch2, dim_i)
                    if not correct:
                        bad_order.append((ch1, ch2))
        return bad_order

    def check_sizes(self):
        bad_size = []
        for tree in self.trees:
            for node in tree.traverse():
                if node.WIDTH < 0 or node.HEIGHT < 0:
                    bad_size.append(node)
        return bad_size

    def check_parent_of_children(self):
        """
        We want to check if each node is inside its parent.
        :return:
        """
        nodes_poblems = []
        for tree in self.trees:
            for node in tree.traverse():
                children_with_problems = nd.check_children_inside(node)
                if children_with_problems:
                    nodes_poblems.append((node, children_with_problems))
        return nodes_poblems

    def check_demand_satisfied(self):
        demand = self.get_batch()
        produced = []
        pieces = self.get_pieces_by_type()
        for k, leaf in pieces.items():
            item = demand.get(k, None)
            # there's no demand for this item code
            if item is None:
                continue
            plate1 = item['WIDTH_ITEM'], item['LENGTH_ITEM']
            plate2 = leaf.WIDTH, leaf.HEIGHT
            if geom.plate_inside_plate(plate1, plate2):
                produced.append(k)
        return np.setdiff1d([*demand], produced)

    def calculate_objective(self):
        if not self.trees:
            return None
        heigth, width = self.get_param('heightPlates'), self.get_param('widthPlates')
        items_area = self.get_items_area()
        last_tree_children = self.trees[-1].get_children()
        last_waste_width = 0
        if last_tree_children and last_tree_children[-1].TYPE == -3:
            last_waste_width = last_tree_children[-1].WIDTH
        return len(self.trees) * heigth * width - last_waste_width * heigth - items_area

    def graph_solution(self, path="", name="rect", show=False, pos=None, dpi=50, fontsize=30, solution=None):
        if solution is None:
            solution = self.trees
        batch_data = self.get_batch()
        stack = batch_data.get_property('STACK')
        sequence = batch_data.get_property('SEQUENCE')
        colors = pal.colorbrewer.qualitative.Set3_5.hex_colors
        width, height = self.get_param('widthPlates'), self.get_param('heightPlates')
        pieces_by_type = self.get_pieces_by_type(by_plate=True, solution=solution)
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
            wastes = nd.get_node_leaves(solution[plate], type_options=[-1, -3])
            for waste in wastes:
                self.draw_leaf(ax1, waste, stack, sequence, colors, fontsize)
            # graph defects
            for defect in self.get_defects_plate(plate):
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
        more_info = ''
        if leaf.TYPE >= 0:
            parent = leaf.up
            if parent is None:
                parent = ""
            else:
                parent = parent.name
            more_info = "\nstack={}\npos={}\ntype={}\nparent={}".format(
                stack.get(leaf.TYPE, ''),
                sequence.get(leaf.TYPE, ''),
                leaf.TYPE,
                parent
            )
        ax1.text(leaf.X + leaf.WIDTH / 2, leaf.Y + leaf.HEIGHT / 2,
                 '{} x {}{}\nnode={}'.
                 format(leaf.WIDTH,
                        leaf.HEIGHT,
                        more_info,
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

    def correct_plate_node_ids(self, solution=None, features=None):
        if features is None:
            features = nd.default_features()
        if solution is None:
            solution = self.trees
        result = {}
        order = 0
        for pos, tree in enumerate(solution):
            for v in tree.traverse("preorder"):
                # rename the NODE_IDs and PLATE_ID
                v.add_features(PLATE_ID=pos, NODE_ID=order)
                v.name = v.NODE_ID
                # correct all wastes to -1 by default
                if 'TYPE' in features:
                    if v.TYPE == -3:
                        v.TYPE = -1
                d = nd.get_features(v, features)
                v.add_features(PARENT=d['PARENT'])
                result[int(v.NODE_ID)] = d
                order += 1
        # last waste is -3
        if 'TYPE' in features:
            if result[order-1]['TYPE'] == -1 and result[order-1]['CUT'] == 1:
                result[order - 1]['TYPE'] = -3
        return result

    def export_solution(self, path=pm.PATHS['results'] + aux.get_timestamp(), prefix='',
                        name='solution', solution=None):
        """
        When creating the output forest:
        – The trees must be given in the correct sequence of plates, i.e. first
        the nodes of plate 0, then nodes of plate 1....
        – For each plate, the root (node with CUT=0) should be given first.
        – The children of a node should be given from left to right (as repre-
        sented on the cutting pattern).
        – The nodes of trees should be given in depth first search.
        – If the last 1-cut of the forest in the last cutting pattern is a waste, it
        must be declared as a residual.
        :param path:
        :param prefix: usually the case.
        :param name: name after prefix. Without extension.
        :return:
        """
        if solution is None:
            solution = self.trees
        if not os.path.exists(path):
            os.mkdir(path)
        result = self.correct_plate_node_ids(solution)
        table = pd.DataFrame.from_dict(result, orient='index')
        col_order = ['PLATE_ID', 'NODE_ID', 'X', 'Y', 'WIDTH', 'HEIGHT', 'TYPE', 'CUT', 'PARENT']
        table = table[col_order]
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

