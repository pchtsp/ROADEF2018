# this script helps build the designs for the documentation

import package.params as pm
import package.solution as sol
from scripts.check import benchmarking
import package.heuristic as heur
import package.data_input as di

path = pm.PATHS['results'] + 'heuristic1800/'

case = 'A1'
path_case = path + case + '/'

case1 = sol.Solution.from_io_files(path=path_case)

# first graph
case1.graph_solution(pos=0)

# benchmark graph
benchmarking()

# for getting swaps I need to store some solutions.
# path =


path = pm.PATHS['root'] + '/python/examples/A6/'
self = heur.ImproveHeuristic.from_io_files(path=path)
options = di.load_data(path= path + 'options.json')
options['heur_params']['weights'] = options['heur_weights']
params = kwargs = options['heur_params']
weights = options['heur_weights']

# self.graph_solution()
# swap_2_before
node1 = self.get_node_by_name(110)
node2 = self.get_node_by_name(107)
try_rotation = False
insert = True
result = self.check_swap_size_rotation(node1, node2, insert=insert, try_rotation=try_rotation, rotation_probs=[0, 1, 0, 0])
self.try_change_node(node=node1, good_candidates=node2, insert=insert, **kwargs)

# swap_3_rot_before
node1 = self.get_node_by_name(88)
node2 = self.get_node_by_name(76)
try_rotation = True
insert = True
result = self.check_swap_size_rotation(node1, node2, insert=insert, try_rotation=try_rotation, rotation_probs=[0, 1, 0, 0])
recalculate = self.swap_nodes_same_level(node1, node2, insert=insert, rotation=result, min_waste=self.get_param('minWaste'))