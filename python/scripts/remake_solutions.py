import package.params as pm
import package.heuristic as heur
import package.data_input as di
import package.nodes as nd
import logging as log
import os

path = pm.PATHS['root'] + 'python/examples/A6/'
program = heur.ImproveHeuristic.from_io_files(path=path)
options = pm.OPTIONS
options['warm_start'] = True
options['path'] = ''

level = log.INFO
if options.get('debug', False):
    level = log.DEBUG
logFile = os.path.join(options['path'], 'output.log')
logFormat = '%(asctime)s %(levelname)s:%(message)s'
open(logFile, 'w').close()
fileh = log.FileHandler(logFile, 'a')
formatter = log.Formatter(logFormat)
fileh.setFormatter(formatter)
_log = log.getLogger()
_log.handlers = [fileh]
_log.setLevel(level)

program.solve(options)
program.correct_plate_node_ids()
program.graph_solution(options['path'], name="plate", dpi=50)

