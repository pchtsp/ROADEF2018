import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import importlib
import package.data_input as di
import package.model as md
import package.heuristic as heur
import argparse
import logging as log
import copy


def solve_case(options):

    case = options['case_name']
    self = md.Model.from_input_files(case)
    new_width = options.get('max_width', None)

    # if specified, change max width:
    if new_width is not None:
        self.input_data['global_param']['widthPlates'] = new_width
    prefix = case + '_'

    if not os.path.exists(options['path']):
        os.mkdir(options['path'])

    output_path = options['path']

    self.export_input_data(path=output_path, prefix=prefix)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    try:
        solution = self.solve(options)
    except:
        solution = None

    if solution is None:
        return None

    # exporting part:
    self.export_cuts(solution, path=output_path)
    try:
        self.load_solution(solution)
    except:
        print('There was a problem loading the solution.')

    self.export_solution(path=output_path, prefix=prefix)


def solve_case_iter(options):
    case = options['case_name']
    self = md.Model.from_input_files(case, path=options['path'])

    # solving part:
    # try:
    self.solve_iterative(options,
                         export=True,
                         max_items=options.get('max_items', 20),
                         sort=True)
    # except:
    #     print('There was an error with the solving!')
    #     print(sys.exc_info()[0])
    #     return None

    return True


def solve_heuristic(options):
    path = options['path']
    case = options['case_name']
    self = heur.ImproveHeuristic.from_input_files(case_name=case, path=path)
    self.solve(options)
    # self.trees = self.best_solution
    self.correct_plate_node_ids()
    if options.get('graph', False):
        self.graph_solution(path, name="edited", dpi=50)
    # print(self.check_sequence(solution=self.best_solution))
    self.export_solution(path=path, prefix=case + '_', name="solution")


def solve(options, case=None):

    # case = args.case
    options = copy.deepcopy(options)
    if case is None:
        case = options['case_name']

    options['path'] += case + '/'
    options['case_name'] = case
    output_path = options['path']

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    di.export_data(output_path, options, name="options", file_type='json')

    level = log.INFO
    if options.get('debug', False):
        level = log.DEBUG
    logFile = os.path.join(output_path, 'output.log')
    logFormat = '%(asctime)s %(levelname)s:%(message)s'
    open(logFile, 'w').close()
    fileh = log.FileHandler(logFile, 'a')
    formatter = log.Formatter(logFormat)
    fileh.setFormatter(formatter)
    _log = log.getLogger()
    _log.handlers = [fileh]
    _log.setLevel(level)

    # log.basicConfig(format='%(asctime)s %(levelname)s:%(message)s',
    #                 filename=logFile,
    #                 level=level)

    try:
        if options['solver'] == 'HEUR':
            solve_heuristic(options)
        else:
            solve_case_iter(options)
    except:
        log.exception("error while executing solve")


if __name__ == "__main__":
    import pprint as pp
    parser = argparse.ArgumentParser(description='Solve an instance ROADEF.')
    parser.add_argument('-c', '--config-file', dest='file', default="package.params",
                        help='config file (default: package.params)')
    parser.add_argument('-a', '--case-name', dest='case', help='case name', nargs='*', default=[None])
    parser.add_argument('-all', '--all-cases', dest='all_cases', help='solve all cases', action='store_true')
    parser.add_argument('-pr', '--path-root', dest='root', help='absolute path to project root')
    parser.add_argument('-rr', '--path-results', dest='results', help='absolute path to results')

    args = parser.parse_args()
    if args.root is not None:
        os.environ['PYTHONPATH'] += ':' + args.root + 'python'
        print(os.environ['PYTHONPATH'])
    pm = importlib.import_module(args.file)

    cases = args.case
    if args.all_cases:
        cases = ['A{}'.format(case) for case in range(1, 21)]

    if args.root is not None:
        pm.PATHS = {**pm.PATHS, **pm.calculate_paths_root(args.root)}

    if args.results is not None:
        pm.PATHS = {**pm.PATHS, **pm.calculate_paths_results(args.results)}

    pm.OPTIONS['path'] = pm.PATHS['experiments']

    print('Using config file in {}'.format(args.file))

    for case in cases:
        solve(pm.OPTIONS, case)