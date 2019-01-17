import package.heuristic as heur
import package.data_input as di
import package.nodes as nd
import package.params as pm
import package.nodes_optim as no

path = pm.PATHS['results'] + '/clust1_20180708_py/A1/'
path = pm.PATHS['results'] + 'test/A2/'
path = pm.PATHS['root'] + 'python/examples/A2_6/'
self = heur.ImproveHeuristic.from_io_files(path=path)
options = di.load_data(path= path + 'options.json')
# options['heur_params']['weights'] = options['heur_weights']
params = kwargs = options['heur_params']
weights = options['heur_params']['weights']
self.graph_solution(path=r'C:\Users\pchtsp\Documents\projects\ROADEF2018\python\examples\A2_6/')
# self.export_solution(path=r'C:\Users\pchtsp\Documents\projects\ROADEF2018\python\examples\A2_6/')
# defects = self.trees[3].DEFECTS
# nd.search_nodes_of_defect(self.trees[3], defects[0])

node1 = self.get_node_by_name(57)
node2 = self.get_node_by_name(216)
def order_wastes(node):
    axis_i, dim_i = nd.get_orientation_from_cut(node, inv=True)
    return getattr(node, dim_i)
result, waste_changes = \
    no.check_swap_nodes_defect(node1, node2, insert=True, min_waste = self.get_param('minWaste'),
                               rotation=[1], order_wastes=order_wastes)

# params['rotation_probs'] = [0.01, 0.99, 0, 0]
recalculate = \
    no.swap_nodes_same_level(node1, node2, insert=True, rotation=[1],
                             wastes_to_edit={1: [(2, 456)], 2: [(2, -469)]},
                             min_waste=self.get_param('minWaste'), debug=True)

# result = self.try_change_node(node=node1, candidates=[node2], insert=True, params=params)
# n, defe = self.check_defects([self.trees[3]])[0]
# self.export_solution(path=)

consist = self.check_consistency()
consist['ch_size'][0].X
# seq = self.check_sequence()
# # self.graph_solution(name='', path=path)
# pos = 1
# node1 = seq[pos][0]
# node2 = seq[pos][1]
# self.debug_nodes(node1, node2)
node1 = self.get_node_by_name(7)
node2 = self.get_node_by_name(8)
#
# node1, node2 = node2, node1
insert = False
self.check_swap_space(node1=node1, node2=node2, global_params=self.get_param(),  insert=insert)
self.check_swap_nodes_seq(node1=node1, node2=node2, global_params=self.get_param(), insert=insert)
self.check_swap_nodes_defect(node1=node1, node2=node2, global_params=self.get_param(), insert=insert)
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
# self.try_change_node(node=node1, candidates=node2, insert=insert, **kwargs)
result = self.check_swap_size_rotation(node1, node2, insert=insert, try_rotation=False)
# self.search_waste_cuts_2(1, **params)

self.best_objective = self.calculate_objective()
self.best_solution= self.trees
for k in weights:
    weights[k] *= 1000000
self.collapse_to_left(1, params)
self.graph_solution()
