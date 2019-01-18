import package.data_input as di
import package.solution as sol
import package.heuristic as heur
import package.model as mod
import package.params as pm

import numpy as np
import pprint as pp
import package.superdict as sd
import pandas as pd
import os
# import matplotlib
# matplotlib.use('Qt5Agg', warn=False, force=True)
import shutil
import re
import subprocess
# import tabulate

def get_all_cases(exp_path):
    return os.listdir(exp_path)


def get_experiments_paths(path, filter_exps=True):
    # cases = get_all_cases()
    if filter_exps:
        experiments = {f: path + f + '/' for f in os.listdir(path) if not re.match('^(old)|(test)|(template)|(README)', f)}
    else:
        experiments = {f: path + f + '/' for f in os.listdir(path) if not re.match('^(old)|(README)', f)}

    cases = {f: os.listdir(path + f + '/') for f in os.listdir(path) if not re.match('^(old)|(README)', f)}
    result = {exp: {c: path + c + '/' for c in cases[exp]} for exp, path in experiments.items()}
    return sd.SuperDict.from_dict(result)


def get_solutions(exp_paths):
    result = {
        exp: {
            c: sol.Solution.from_io_files(path=p, case_name=c)
            for c, p in cases.items() if di.dir_has_solution(p)
        }
        for exp, cases in exp_paths.items()
    }
    return sd.SuperDict.from_dict(result)


def is_pareto_dominated(costs):
    """
    :param costs: An (n_points, n_costs) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient
    """
    is_dominated = np.ones(costs.shape[0], dtype=bool)
    for i, c in enumerate(costs):
        is_dominated[i] = sum(np.all(costs<=c, axis=1)) > 1
    return is_dominated


def dominant_experiments():
    # TODO: last for loop is not working.
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.read_csv(others_path, sep=';')
    table1.columns = ['INSTANCE', 'others', 'TEAM']
    exp_paths = get_experiments_paths(pm.PATHS['results'])
    # experiments_names = list(set(i for k in exp_paths.values() for i in k))
    experiments_names = exp_paths.keys_l()

    solutions = get_solutions(exp_paths)

    feasibility = {
        exp: {
            c: s.count_errors()
            for c, s in cases.items()
        }
        for exp, cases in solutions.items()
    }

    objectives = np.zeros((len(experiments_names), len(solutions)))
        # exp_pos = 0
    for exp_pos, exp in enumerate(experiments_names):
        case_pos = 0

        if exp in solutions and feasibility[c][exp] == 0:
            objectives[exp_pos][case_pos] = experiments[exp].calculate_objective()
        else:
            objectives[exp_pos][case_pos] = 9000000000
        case_pos += 1

    return np.array(experiments_names)[is_pareto_dominated(objectives)]


def benchmarking(value='dif_jumbo', experiments_filter=None):
    assert value in ['obj', 'dif_jumbo', 'dif', 'dif_perc']
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.read_csv(others_path, sep=';')
    table1.columns = ['INSTANCE', 'others', 'TEAM']
    exp_paths = get_experiments_paths(pm.PATHS['results'])
    if experiments_filter is not None:
        experiments_filter.append('heuristic1800')
        exp_paths = exp_paths.filter(experiments_filter)

    solutions = get_solutions(exp_paths)

    feasibility = {
        exp: {
            c: s.count_errors()
            for c, s in cases.items()
        }
        for exp, cases in solutions.items()
    }

    objectives = {
        exp: {
            c: s.calculate_objective()
            for c, s in cases.items()
            if feasibility[exp][c] == 0
        }
        for exp, cases in solutions.items()
    }


    f_experiment = 'heuristic1800'
    items_area = {c: s.get_items_area() for c, s in solutions[f_experiment].items()}
    instance_case1 = solutions[f_experiment]['A1']
    jumbo_area = instance_case1.get_param('widthPlates') * instance_case1.get_param('heightPlates')

    table_items = pd.DataFrame.from_dict(items_area, orient='index').reset_index().\
        rename(columns={0: 'items', 'index': 'case'})
    renames = {'index': 'experiment', 'variable': 'case'}
    table_obj = pd.DataFrame.from_dict(objectives, orient='index').reset_index().\
        melt(id_vars='index', value_name='obj').rename(columns=renames)
    table_feas = pd.DataFrame.from_dict(feasibility, orient='index').reset_index().\
        melt(id_vars='index', value_name='feas').rename(columns=renames)
    others = table1[['INSTANCE', 'others']].rename(columns={'INSTANCE': 'case'})
    others_ed = others.copy()
    others_ed['experiment'] = 'others'
    table_obj = table_obj.append(others_ed.rename(columns={'others': 'obj'}), sort=False)

    # df_final = reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)
    params = {'how': 'outer'}
    summary = \
        table_obj.\
            merge(others, **params).\
            merge(table_items, **params).\
            merge(table_feas, **params)

    summary['obj'] = summary.obj
    summary['dif'] = (summary.obj - summary.others)
    summary['dif_perc'] = (summary.obj - summary.others) / summary.others * 100
    summary['dif_jumbo'] = (summary.obj - summary.others) / jumbo_area
    # value = 'dif_perc'
    indeces = ['case', 'experiment'] + [value]
    summary[indeces].pivot(index='case', columns='experiment', values=value).plot.bar()


def graph(experiment, case=None, dpi=25):
    # TODO: make experiment optional to graph all
    # experiment = 'clust1_20180918_venv'
    exp_paths = get_experiments_paths(pm.PATHS['results'], filter_exps=False)
    if case is not None:
        exp_paths = exp_paths.filter([case])
    solutions = get_solutions(exp_paths)
    solutions_n = solutions[experiment]
    destination = pm.PATHS['root'] + 'graphs/'
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    for k, v in solutions_n.items():
        path = destination + k
        os.makedirs(path)
        v.graph_solution(path, dpi=dpi)


def execute_checker(experiment, path_checker):
    # experiment = 'prise_20180917_venv'
    exp_path = pm.PATHS['results'] + experiment + '/'
    cases = get_all_cases(exp_path)
    executable = "bin/Release/Checker"
    complete_path = os.path.join(path_checker, executable)
    if not os.path.exists(complete_path):
        executable = './checker'
        complete_path = os.path.join(path_checker, executable)
    assert os.path.exists(complete_path), 'Checker not found, please build it first'
    # path_checker = '/home/pchtsp/Documents/projects/ROADEF2018/resources/checker/'
    destination = path_checker + 'instances_checker/'
    files_cases = {c: exp_path + c + '/solution.csv' for c in cases}
    for case, _f in files_cases.items():
        location, filename = os.path.split(_f)
        dest_filename = str(case) + '_' + filename
        _f_alt = os.path.join(location, dest_filename)
        dest_name = os.path.join(destination, dest_filename)
        if not os.path.exists(_f):
            _f = _f_alt
        if not os.path.exists(_f):
            if os.path.exists(dest_name):
                os.remove(dest_name)
            continue
        shutil.copy(_f, dest_name)
    results = {}
    for case in cases:
        a = subprocess.run([executable, case],
                       input="5\n0", universal_newlines=True,
                       cwd=path_checker, stdout=subprocess.PIPE)
        results[case] = re.search(r'SOLUTION VALIDATED SUCCESSFULLY', a.stdout)
    return results


def get_objectives(experiment):
    path = pm.PATHS['results'] + experiment + '/'
    cases = get_all_cases(path)
    solutions = {}
    for c in cases:
        try:
            solutions[c] = sol.Solution.from_io_files(path=path + c + '/', case_name=c)
        except:
            pass
    return {c: s.calculate_objective() for c, s in solutions.items()}


def summary_table(experiment, path_out):
    objs = get_objectives(experiment)
    latex = pd.DataFrame.from_dict(objs, orient='index').reset_index().\
        rename(columns={0: 'objective', 'index': 'instance'}).to_latex(bold_rows=True, index=False)
    with open(path_out, 'w') as f:
        f.write(latex)


def check_experiment(experiment, case=None):
    path = pm.PATHS['results'] + experiment + '/'
    if case is None:
        cases = get_all_cases(path)
    else:
        cases = [case]
    solutions = {c: sol.Solution.from_io_files(path=path + c + '/', case_name=c) for c in cases}
    return {c: s.count_errors() for c, s in solutions.items()}


if __name__ == "__main__":
    # pass
    # graph(experiment='clust1_20180718_venv_pypy', case='A16')
    # graph(experiment='hp_20181210')
    benchmarking('obj', experiments_filter=['hp_20181209', 'hp_20181126', 'hp_20180718_venv_pypy',
                                            'hp_20180911_venv', 'prise_20180917_venv',
                                                  'prise_20180926_venv', 'hp_20190117', 'hp_20190116'])
    experiment = 'hp_20181209'
    experiment = 'test'
    # exp_paths = get_experiments_paths(pm.PATHS['results'])
    # experiment = 'len_20180718_venv_py'
    path_checker = pm.PATHS['checker']
    # summary_table(experiment, pm.PATHS['root'] + 'docs/heuristics/results.tex')
    # rrr = execute_checker(experiment, path_checker=path_checker)
    rrr2 = check_experiment(experiment, 'A2')
    # A2sol = sol.Solution.from_io_files(path=pm.PATHS['results'] + experiment + '/A15' + '/', case_name='A15')
    # A2sol.graph_solution()
    # rrr = get_objectives()
    # dominant_experiments()
    pass