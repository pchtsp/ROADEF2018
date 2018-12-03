import unittest
import package.params as pm
import package.heuristic as heur
import package.data_input as di
import package.nodes as nd
import package.nodes_optim as no
import package.nodes_checks as nc

class TestHeuristic(unittest.TestCase):

    # test search_node_of_defect

    def check_defects(self, case, expected, plates=None):
        path = pm.PATHS['root'] + 'python/examples/{}/'.format(case)
        program = heur.ImproveHeuristic.from_io_files(path=path)
        if plates is None:
            plates = range(len(program.trees))
        solution = [program.trees[n] for n in plates]
        defects = program.check_defects(solution)
        defect_codes = [d[0].name for d in defects]
        self.assertEqual(defect_codes, expected)

    def check_swap(self, node1, node2, insert, rotation, expected, case='A6'):
        path = pm.PATHS['root'] + 'python/examples/{}/'.format(case)
        program = heur.ImproveHeuristic.from_io_files(path=path)
        options = di.load_data(path=path + 'options.json')
        min_waste = program.get_param('minWaste')
        global_params = program.get_param()
        params = kwargs = options['heur_params']
        # weights = options['heur_weights']

        # self.graph_solution()
        # swap_2_before
        _nodes = {1: node1, 2: node2}
        nodes = {k: program.get_node_by_name(v) for k, v in _nodes.items()}
        rot = nc.check_swap_size_rotation(nodes[1], nodes[2], insert=insert, min_waste=min_waste, params=params)
        self.assertIsNotNone(rot)
        # nd.check_swap_space(nodes[1], nodes[2], insert=insert, global_params=params)
        nodes_changes, wastes_to_edit = no.get_swap_node_changes(nodes, min_waste, insert, rotation)
        result = \
            no.swap_nodes_same_level(nodes[1], nodes[2], min_waste, wastes_to_edit, insert=insert, rotation=rotation)
        self.assertEqual(len(program.check_consistency()), 0)
        # self.assertEqual(expected, result)

    def check_swap_defects(self, node1, node2, insert, rotation, expected, case='A6'):
        path = pm.PATHS['root'] + 'python/examples/{}/'.format(case)
        program = heur.ImproveHeuristic.from_io_files(path=path)
        options = di.load_data(path=path + 'options.json')
        min_waste = program.get_param('minWaste')
        # params = kwargs = options['heur_params']
        # weights = options['heur_weights']

        # self.graph_solution()
        # swap_2_before
        node1 = program.get_node_by_name(node1)
        node2 = program.get_node_by_name(node2)

        def order_wastes(node):
            axis_i, dim_i = nd.get_orientation_from_cut(node, inv=True)
            return getattr(node, dim_i)

        result, waste_changes= \
            no.check_swap_nodes_defect(node1, node2, insert=insert, min_waste=min_waste,
                                       rotation=rotation, order_wastes=order_wastes)
        self.assertEqual(result, expected)
        # add_at_end = True


    def check_swap_defects_squares(self, node1, node2, insert, rotation, expected, case='A6', **swap_nodes_params):
        path = pm.PATHS['root'] + 'python/examples/{}/'.format(case)
        program = heur.ImproveHeuristic.from_io_files(path=path)
        _nodes = {1: node1, 2: node2}

        nodes = {k: program.get_node_by_name(v) for k, v in _nodes.items()}
        min_waste = program.get_param('minWaste')
        nodes_changes, wastes_mods = no.get_swap_node_changes(nodes, min_waste, insert, rotation, **swap_nodes_params)
        squares = no.get_swap_squares(nodes, nodes_changes, insert, rotation)
        self.assertEqual(expected, squares)
        # self.assertDictEqual(expected[1], squares[1])

    def test_check_swap_defects1(self):
        return self.check_swap_defects(node1=4, node2=33, insert=True, rotation=[], expected=-1)

    def test_check_swap_defects2(self):
        return self.check_swap_defects(node1=11, node2=8, insert=False, rotation=[], expected=-10000)

    def test_check_swap_defects3(self):
        return self.check_swap_defects(node1=10, node2=8, insert=False, rotation=[], expected=-1)

    def test_check_swap_defects4(self):
        return self.check_swap_defects(node1=44, node2=59, insert=True, rotation=[], expected=-2)

    def test_check_swap_defects5(self):
        return self.check_swap_defects(node1=67, node2=59, insert=True, rotation=[], expected=-1)

    def test_check_swap_defects6(self):
        return self.check_swap_defects(node1=67, node2=59, insert=False, rotation=[], expected=0)

    def test_check_swap_defects_sq1(self):
        expected = [
            {0: [
                [{'X': 289, 'Y': 0}, {'X': 289+321, 'Y': 691}],
                [{'X': 6000-382-382, 'Y': 1269}, {'X': 6000-382, 'Y': 3210-224}]
            ]},
            {0: [
                [{'X': 6000-382-382, 'Y': 0}, {'X': 6000-382-382+321, 'Y': 691}],
                [{'X': 6000-382-382, 'Y': 1269+224}, {'X': 6000-382, 'Y': 3210}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=4, node2=33, insert=True, rotation=[], expected=expected)

    def no_test_check_swap_defects_sq2(self):
        expected = [
            {0: [
                [{'X': 610, 'Y': 705+1684}, {'X': 610+843, 'Y': 705+1684+821}],
                [{'X': 610, 'Y': 705}, {'X': 610+864, 'Y': 705+1684}]
            ]},
            {0: [
                [{'X': 610, 'Y': 0}, {'X': 610+843, 'Y': 821}],
                [{'X': 610, 'Y': 821}, {'X': 610+864, 'Y': 821+1684}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=11, node2=8, insert=False, rotation=[], expected=expected)

    def test_check_swap_defects_sq3(self):
        expected = [
            {0: [
                [{'X': 610, 'Y': 705+1684}, {'X': 610+843, 'Y': 705+1684+821}],
                [{'X': 610, 'Y': 705}, {'X': 610+864, 'Y': 705+1684}]
            ]},
            {0: [
                [{'X': 610, 'Y': 0}, {'X': 610+843, 'Y': 821}],
                [{'X': 610, 'Y': 821}, {'X': 610+864, 'Y': 821+1684}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=10, node2=8, insert=False, rotation=[], expected=expected)

    def test_check_swap_defects_sq4(self):
        expected = [
            {1: [
                [{'X': 864, 'Y': 402}, {'X': 864+1259, 'Y': 402+2128}],
                [{'X': 864, 'Y': 402 + 2128}, {'X': 864 + 1009, 'Y': 3210 - 278}],
                [{'X': 864 + 416, 'Y': 0}, {'X': 864 + 416 + 843, 'Y': 402}],
                [{'X': 6000-824-31-1398, 'Y': 964}, {'X': 6000-824-31, 'Y': 3210-80}]
            ]},
            {1: [
                [{'X': 864, 'Y': 0}, {'X': 864+1259, 'Y': 2128}],
                [{'X': 864, 'Y': 2128}, {'X': 864 + 1009, 'Y': 2128+402}],
                [{'X': 6000-824-31-1398+416, 'Y': 0}, {'X': 6000-824-31-1398+416 + 843, 'Y': 402}],
                [{'X': 6000-824-31-1398, 'Y': 964+80}, {'X': 6000-824-31, 'Y': 3210}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=44, node2=59, insert=True, rotation=[], expected=expected)

    def test_check_swap_defects_sq6(self):
        expected = [
            {1: [
                [{'X': 6000-824, 'Y': 1009+1259}, {'X': 6000-528, 'Y': 1009+1259+584}],
                [{'X': 6000 - 824 - 31 - 1398, 'Y': 964}, {'X': 6000 - 824 - 31, 'Y': 3210 - 80}]
            ]},
            {1: [
                [{'X': 6000 - 824 - 31 - 1398, 'Y': 0}, {'X': 6000 - 824 - 31 - 1398+296, 'Y': 584}],
                [{'X': 6000 - 824 - 31 - 1398, 'Y': 784}, {'X': 6000 - 824 - 31, 'Y': 784+2166}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=67, node2=59, insert=False, rotation=[], expected=expected)

    def test_check_swap_defects_sq7(self):
        expected = [
            {2: [
                [{'X': 947+1398+821+602, 'Y': 0}, {'X': 947+1398+821+602+496, 'Y': 784}]
            ]},
            {2: [
                [{'X': 947, 'Y': 130+2166}, {'X': 947+784, 'Y': 130+2166+496}]
            ]}
        ]
        return self.check_swap_defects_squares(node1=86, node2=77, insert=False, rotation=[1], expected=expected,
                                               add_at_end = True)

    def test_swap1(self):
        self.check_swap(node1=18, node2=23, insert=False, rotation=[], expected=True)

    def test_swap2(self):
        self.check_swap(node1=4, node2=17, insert=True, rotation=[], expected=True)

    def test_swap3(self):
        self.check_swap(node1=4, node2=33, insert=True, rotation=[], expected=True)

    def test_swap4(self):
        self.check_swap(node1=67, node2=53, insert=True, rotation=[], expected=True)

    def test_swap5(self):
        self.check_swap(node1=84, node2=71, insert=True, rotation=[], expected=True)

    def test_swap6(self):
        self.check_swap(node1=4, node2=35, insert=False, rotation=[], expected=True)

    def test_swap7(self):
        self.check_swap(node1=94, node2=75, insert=True, rotation=[1], expected=True)

    def test_swap8(self):
        self.check_swap(node1=67, node2=57, insert=True, rotation=[1], expected=True)

    def test_swap9(self):
        # rotation
        self.check_swap(node1=2, node2=21, insert=True, rotation=[1], expected=True)

    def test_swap10(self):
        # interlevel 3 => 1
        self.check_swap(node1=86, node2=82, insert=True, rotation=[], expected=True)

    def test_swap11(self):
        # interlevel 4 => 2
        self.check_swap(node1=68, node2=57, insert=True, rotation=[], expected=True)

    def test_swap12(self):
        # interlevel 4 => 1
        self.check_swap(node1=68, node2=73, insert=True, rotation=[], expected=True)

    def test_swap13(self):
        # interlevel 3 => 2 with waste
        self.check_swap(node1=67, node2=57, insert=True, rotation=[], expected=True)

    def test_swap14(self):
        # interlevel 2 => 1
        self.check_swap(node1=101, node2=120, insert=True, rotation=[], expected=True)

    def test_swap15(self):
        # interlevel rotation + unnest
        self.check_swap(node1=84, node2=77, insert=True, rotation=[1], expected=True)

    def test_swap16(self):
        # rotation + swap
        self.check_swap(node1=10, node2=24, insert=False, rotation=[1], expected=True)

    def test_swap17_A15(self):
        # swap siblings
        self.check_swap(node1=179, node2=180, insert=False, rotation=[], expected=True, case='A15')

    def test_check_swap_defects_A2_1(self):
        return self.check_swap_defects(node1=130, node2=159, insert=False, rotation=[], expected=0, case='A2')

    # def test_swap1_A2(self):
    #     # swap siblings
    #     self.check_swap(node1=130, node2=159, insert=False, rotation=[], expected=True, case='A2')

    # test swap or insert where there is nodes that need to be collapsed.

    def test_def_A2(self):
        expected = [144]
        self.check_defects(case='A2', expected=expected, plates=[3])

    def test_def_A2_2(self):
        expected = [106, 97]
        self.check_defects(case='A2_2', expected=expected, plates=[2])

    def test_check_swap_defects_A2_3(self):
        return self.check_swap_defects(node1=173, node2=57, insert=False, rotation=[], expected=-1,
                                       case='A2_3')

    def test_check_swap_defects_A2_3_sq(self):
        dif_width = -1294+1077
        expected = [
            {4: [
                [{'X': 6000 - 1077, 'Y': 0}, {'X': 6000-621, 'Y': 469}],
                [{'X': 6000 - 1077, 'Y': 469+296}, {'X': 6000-139-469, 'Y': 3210-1989}],
                [{'X': 6000 - 139-469, 'Y': 469 + 296}, {'X': 6000 - 139, 'Y': 3210 - 1989}],
                [{'X': 6000 - 1077, 'Y': 3210-1989}, {'X': 6000, 'Y': 3210}]
            ],
            1: [
                [{'X': 6000 - 3371, 'Y': 0}, {'X': 6000, 'Y': 810}],
                [{'X': 6000 - 3371, 'Y': 810}, {'X': 6000 - 65, 'Y': 810 + 803}],
                [{'X': 6000 - 3371, 'Y': 810 + 803}, {'X': 6000, 'Y': 3210 - 423 - 364}],
                [{'X': 6000 - 3371, 'Y': 3210 - 423}, {'X': 6000 - 2462, 'Y': 3210}],
                [{'X': 1288+47, 'Y': 0}, {'X': 1288 +47+469, 'Y': 456}],
                [{'X': 1288 + 47, 'Y': 456+286}, {'X': 1288 + 47 + 746, 'Y': 456+286+426}],
                [{'X': 1288 + 47, 'Y': 456 + 286+426}, {'X': 1288 + 47 + 456, 'Y': 3210-1573}],
                [{'X': 1288 + 47, 'Y': 3210-1573}, {'X': 6000-3371, 'Y': 3210}],
            ]
            },
            {4: [
                [{'X': 6000 - 1294, 'Y': 0}, {'X': 6000 - 825, 'Y': 456}],
                [{'X': 6000 - 1294, 'Y': 456+286}, {'X': 6000-548, 'Y': 456+286+426}],
                [{'X': 6000 - 1294, 'Y': 456+286+426}, {'X': 6000 - 838, 'Y': 3210 - 1573}],
                [{'X': 6000 - 1294, 'Y': 3210-1573}, {'X': 6000, 'Y': 3210}]
            ],
            1: [
                [{'X': 1288+47, 'Y': 0}, {'X': 1288 +47+456, 'Y': 469}],
                [{'X': 1288 + 47, 'Y': 469+296}, {'X': 1288 + 47 + 469, 'Y': 3210-1989}],
                [{'X': 1288 + 47+ 469, 'Y': 469 + 296}, {'X': 1288 + 47 + 469+469, 'Y': 3210-1989}],
                [{'X': 1288 + 47, 'Y': 3210-1989}, {'X': 1288 + 47+1077, 'Y': 3210}],
                [{'X': 6000-3371+dif_width, 'Y': 0}, {'X': 6000+dif_width, 'Y': 810}],
                [{'X': 6000 - 3371+dif_width, 'Y': 810}, {'X': 6000-65+dif_width, 'Y': 810+803}],
                [{'X': 6000 - 3371+dif_width, 'Y': 810+803}, {'X': 6000+dif_width, 'Y': 3210-423-364}],
                [{'X': 6000 - 3371+dif_width, 'Y': 3210-423}, {'X': 6000-2462+dif_width, 'Y': 3210 }],
            ],
            }
        ]
        return self.check_swap_defects_squares(node1=173, node2=57, insert=False, rotation=[], expected=expected,
                                               add_at_end = True, case='A2_3')


if __name__ == "__main__":
    # t = TestHeuristic()
    # t.test_check_swap_defects_A2_3_sq()
    # # t.test_swap17_A15()
    # # t.test_check_swap_defects_sq7()
    unittest.main()



# debug defects check swap:
#
# node1.PLATE_ID
# node1, node2 = recalculate.values()
# recalculate = nd.swap_nodes_same_level(node1, node2, insert=insert, rotation=rot,
#                                               debug=self.debug, min_waste=self.get_param('minWaste'))
# node1, node2 = recalculate.values()
# self.graph_solution(pos=6)
# from package.nodes import *
# import pprint as pp
# evaluate the defects swap checks for the new nodes.