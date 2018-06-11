import package.heuristic as heur
import package.data_input as di
path = '/home/pchtsp/Documents/projects/ROADEF2018/results/heuristic1800/A1/'
self = heur.ImproveHeuristic.from_io_files(path=path)
options = di.load_data(path= path + 'options.json')
options['heur_params']['weights'] = options['heur_weights']
params = kwargs = options['heur_params']
weights = options['heur_weights']

# seq = self.check_sequence()
# # self.graph_solution(name='', path=path)
# pos = 1
# node1 = seq[pos][0]
# node2 = seq[pos][1]
# self.debug_nodes(node1, node2)
node1 = self.get_node_by_name(8)
node2 = self.get_node_by_name(11)
#
# node1, node2 = node2, node1
insert = True
self.check_swap_space(node1=node1, node2=node2, insert=insert)
self.check_swap_nodes_seq(node1=node1, node2=node2, insert=insert)
self.check_swap_nodes_defect(node1=node1, node2=node2, insert=insert)
self.evaluate_swap(node1=node1, node2=node2, insert=insert, weights=weights)
#
# self.change_level_by_seq2(level=2, **options['heur_params'])
# self.graph_solution(name='', path=path)
#
#
# self.best_solution = self.trees
# self.best_objective = self.evaluate_solution(weights)
# self.evaluate_swap(node1=node1, node2=node2, insert=False, weights=weights)
#
# node1 = self.trees[1].get_children()[-1]
# node2 = self.trees[2].get_children()[0]
self.try_change_node(node=node1, candidates=node2, insert=insert, **kwargs)

self.search_waste_cuts_2(1, **params)
