# this script helps build the designs for the documentation

import package.params_win as pm
import package.solution as sol

path = pm.PATHS['results'] + 'heuristic1800/'

case = 'A1'
path_case = path + case + '/'

case1 = sol.Solution.from_io_files(path=path_case)

case1.graph_solution(pos=0)