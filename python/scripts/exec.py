# python3
import os, sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# import importlib
import package.data_input as di
import package.model as md
import package.heuristic as heur
import scripts.caseManager as cs
import argparse
import logging as log
import copy
import random as rn
import numpy as np
import package.params as pm


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
    input_path = options['input_path']
    output_path = options['output_path']
    case = options['case_name']
    filename = options['output_file_name']
    prefix = ''
    if filename is None:
        filename = 'solution.csv'
        prefix = case + '_'
    self = heur.ImproveHeuristic.from_input_files(case_name=case, path=input_path)
    try:
        self.solve(options)
    except AssertionError as e:
        self.export_solution(path=output_path, name=filename, prefix=prefix)
        raise e
    self.correct_plate_node_ids()
    if options.get('graph', False):
        self.graph_solution(output_path, name="plate", dpi=50)
    # print(self.check_sequence(solution=self.best_solution))
    self.export_solution(path=output_path, name=filename, prefix=prefix)
    return self


def solve(options):
    output_path = options['output_path']

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
    # print(logFile)
    formatter = log.Formatter(logFormat)
    fileh.setFormatter(formatter)
    _log = log.getLogger()
    _log.handlers = [fileh]
    _log.setLevel(level)
    # log.info(options)

    try:
        if options['solver'] == 'HEUR':
            return solve_heuristic(options)
        else:
            return solve_case_iter(options)
    except:
        log.exception("error while executing solve")
        return None

if __name__ == "__main__":
    import pprint as pp
    import json

    parser = argparse.ArgumentParser(description='Solve an instance ROADEF.')
    # parser.add_argument('-c', '--config-file', dest='file', default="package.params", help='config file (default: package.params)')
    parser.add_argument('-a', '--case-name', dest='case', help='case name', nargs='*', default=[None])
    parser.add_argument('-p', '--case-location', dest='case_loc', help='case location')
    parser.add_argument('-all', '--all-cases', dest='all_cases', help='solve all cases', action='store_true')
    parser.add_argument('-pr', '--path-root', dest='root', help='absolute path to project root')
    parser.add_argument('-rr', '--path-results', dest='results', help='absolute path to results')
    parser.add_argument('-rd', '--results-dir', dest='results_dir', help='directory to export experiments')
    parser.add_argument('-ng', '--no-graph', dest='no_graph', help='avoid graphing at the end', action='store_true')
    parser.add_argument('-ej', '--extra-jumbos', dest='extra_jumbos', help='number of extra jumbos to add', type=int)
    parser.add_argument('-tl', '-t', '--time-limit', dest='time_limit', help='max time to solve instance', type=int)
    parser.add_argument('-s', '--seed', dest='seed', help='seed', type=int)
    parser.add_argument('-temp', '--temperature', dest='temperature', help='initial temperature', type=int)
    parser.add_argument('-o', '--output-file', dest='output_file', help='file to write solution', default='solution.csv')
    parser.add_argument('-name', '--name-group', dest='name', help='name of group', action='store_true')
    parser.add_argument('-np', '--num-process', dest='num_process', help='num of processors', type=int)
    parser.add_argument('-ds', '--data-set', dest='data_set', help='dataset to solve', default='A')
    parser.add_argument('-hr', '--heur-remake', dest='heur_remake', type=json.loads)
    parser.add_argument('-hp', '--heur-params', dest='heur_params', type=json.loads)
    parser.add_argument('-ho', '--heur-optim', dest='heur_optim', type=json.loads)
    parser.add_argument('-mp', '--main-param', dest='main_param', type=json.loads)

    args = parser.parse_args()
    if getattr(sys, 'frozen', False):
        # we are running in a bundle
        root = sys._MEIPASS + '/'
    else:
        # we are running in a normal Python environment
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) + '/'
    # print('the root path: {}'.format(root))
    # print('the files: {}'.format(os.listdir(root)))
    if args.root is not None:
        root = args.root
        if 'PYTHONPATH' not in os.environ:
            os.environ['PYTHONPATH'] = ''
        os.environ['PYTHONPATH'] += ':' + root + 'python'
        # print(os.environ['PYTHONPATH'])
    # pm = importlib.import_module(args.file)

    if args.main_param:
        pm.OPTIONS.update(args.main_param)

    if args.heur_remake:
        pm.OPTIONS['heur_remake'].update(args.heur_remake)

    if args.heur_params:
        pm.OPTIONS['heur_params'].update(args.heur_params)

    if args.heur_optim:
        pm.OPTIONS['heur_optim'].update(args.heur_optim)

    # print(pm.OPTIONS['heur_remake'])

    pm.PATHS.update(pm.calculate_paths_root(root, data_set=args.data_set))

    cases = args.case
    data_set = args.data_set

    num_cases = {'A': 21, 'B': 16}

    if args.all_cases:
        cases = ['{}{}'.format(data_set, case) for case in range(1, num_cases[data_set])]

    # Two options of configuring output.
    # If a results path is given: a directory is used to generate the output.
    if args.results is not None:
        pm.PATHS.update(pm.calculate_paths_results(args.results))

        if args.results_dir is not None:
            pm.PATHS['experiments'] = pm.PATHS['results'] + args.results_dir + '/'
            if not os.path.exists(pm.PATHS['experiments']):
                cs.separate_cases(name=args.results_dir,
                                  data_dir=pm.PATHS['data'],
                                  results_dir=pm.PATHS['results'],
                                  cases=cases)

        pm.OPTIONS['input_path'] = pm.OPTIONS['output_path'] = pm.PATHS['experiments']
    # if not, the output will be written in a single file.
    else:
        if args.case_loc:
            location, case = os.path.split(args.case_loc)
            cases = [case]
            pm.OPTIONS['input_path'] = location + '/'
            pm.OPTIONS['output_file_name'] = None
        else:
            pm.OPTIONS['input_path'] = pm.PATHS['data']
        pm.OPTIONS['output_path'] = './'
        pm.OPTIONS['output_file_name'] = args.output_file

    if args.no_graph:
        pm.OPTIONS['graph'] = False

    if args.extra_jumbos is not None:
        pm.OPTIONS['heur_params']['extra_jumbos'] = args.extra_jumbos

    if args.time_limit is not None:
        pm.OPTIONS['timeLimit'] = args.time_limit

    # we'll take 5 seconds from the time limit to be sure to stop in time.
    pm.OPTIONS['timeLimit'] -= 15

    if args.temperature is not None:
        pm.OPTIONS['heur_params']['temperature'] = args.temperature

    if args.num_process:
        pm.OPTIONS['num_processors'] = args.num_process

    if args.seed is not None:
        pm.OPTIONS['seed'] = args.seed

    if pm.OPTIONS['seed'] is None:
        pm.OPTIONS['seed'] = rn.randint(1, 10000)

    # print('Using config file in {}'.format(args.file))
    if args.name:
        print('S22')
        exit()
    for case in cases:
        options = copy.deepcopy(pm.OPTIONS)
        if case:
            options['case_name'] = case
        if args.results is not None:
            options['output_path'] += options['case_name'] + '/'
            options['input_path'] += options['case_name'] + '/'
        solve(options)