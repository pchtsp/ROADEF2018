import unittest
import package.params as pm
import package.heuristic as heur
import package.data_input as di
import package.nodes as nd

# class TddInPythonExample(unittest.TestCase):
#     def test_calculator_add_method_returns_correct_result(self):
#         calc = Calculator()
#         result = calc.add(2, 2)
#         self.assertEqual(4, result)

# test checking defects when swapping.
# test sequence when swapping.
# test correct swapping under several conditions
# test

"""
types of swapping cases:
* same plate siblings
* same plate, not siblings
* different plate same level
* different plate different level
*

"""

class TestHeuristic(unittest.TestCase):

    def check_swap_defects(self, node1, node2, insert, expected):
        path = pm.PATHS['root'] + 'python/examples/A6/'
        program = heur.ImproveHeuristic.from_io_files(path=path)
        options = di.load_data(path=path + 'options.json')
        # params = kwargs = options['heur_params']
        # weights = options['heur_weights']

        # self.graph_solution()
        # swap_2_before
        node1 = program.get_node_by_name(node1)
        node2 = program.get_node_by_name(node2)
        result = nd.check_swap_nodes_defect(node1, node2, insert=insert, min_waste=program.get_param('minWaste'))
        self.assertEqual(expected, result)

    def test_check_swap_defects1(self):
        return self.check_swap_defects(node1=4, node2=33, insert=True, expected=-1)

    def test_check_swap_defects2(self):
        return self.check_swap_defects(node1=11, node2=8, insert=False, expected=-1)

    def test_check_swap_defects3(self):
        return self.check_swap_defects(node1=10, node2=8, insert=False, expected=-1)

    def test_check_swap_defects4(self):
        return self.check_swap_defects(node1=44, node2=59, insert=True, expected=-2)

    def test_check_swap_defects5(self):
        return self.check_swap_defects(node1=67, node2=59, insert=True, expected=-1)

    def test_check_swap_defects6(self):
        return self.check_swap_defects(node1=67, node2=59, insert=False, expected=-1)

if __name__ == "__main__":
    t = TestHeuristic()
    t.test_check_swap_defects6()
    # unittest.main()



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