import package.data_input as di
import package.solution as sol
import package.heuristic as heur
import package.model as mod
try:
    import package.params_win as pm
except:
    import package.params as pm

import numpy as np
import pprint as pp
import package.superdict as sd
import pandas as pd
import os
import matplotlib.pyplot as plt
import shutil
import re



def stats():
    cases = ["A{}".format(n) for n in range(1, 21)]
    errors = {}
    errors_len = {}
    solution = {}

    for case in cases:
    # case = cases[0]
        path = pm.PATHS['experiments'] + case + '/'
        solution[case] = sol.Solution.from_io_files(path=path, case_name=case)
        errors[case] = sd.SuperDict(solution[case].check_all())
        errors_len[case] = errors[case].to_lendict()

    [n for n in errors_len if errors_len[n]]
    # [n[0].PLATE_ID for n in errors['A13']['sequence']]
    # defect_nodes = solution['A2'].get_defects_nodes()
    # solution['A20'].check_all()

    pp.pprint(errors_len)


def get_experiments_paths(path):
    cases = ["A{}".format(n) for n in range(1, 21)]
    experiments = {f: path + f + '/' for f in os.listdir(path) if not re.match('^(old)|(test)|(template)', f)}
    return {c: {exp: path + c + '/' for exp, path in experiments.items()} for c in cases}


def get_solutions(exp_paths):
    return \
        {c: {
            exp: sol.Solution.from_io_files(path=p, case_name=c)
            for exp, p in experiments.items() if di.dir_has_solution(p)
        }
            for c, experiments in exp_paths.items()
        }


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
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.read_csv(others_path, sep=';')
    table1.columns = ['INSTANCE', 'others', 'TEAM']
    exp_paths = get_experiments_paths(pm.PATHS['results'])
    experiments_names = list(set(i for k in exp_paths.values() for i in k))

    solutions = get_solutions(exp_paths)

    feasibility = \
        {c: {
            exp: s.count_errors()
            for exp, s in experiments.items()
        }
            for c, experiments in solutions.items()
        }

    objectives = np.zeros((len(experiments_names), len(solutions)))
    case_pos = 0
    for c, experiments in solutions.items():
        exp_pos = 0
        for exp in experiments_names:
            if exp in experiments and feasibility[c][exp] == 0:
                objectives[exp_pos][case_pos] = experiments[exp].calculate_objective()
            else:
                objectives[exp_pos][case_pos] = 9000000000
            exp_pos += 1
        case_pos += 1

    return np.array(experiments_names)[is_pareto_dominated(objectives)]


def benchmarking(value='dif_jumbo'):
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.read_csv(others_path, sep=';')
    table1.columns = ['INSTANCE', 'others', 'TEAM']
    exp_paths = get_experiments_paths(pm.PATHS['results'])

    solutions = get_solutions(exp_paths)

    feasibility = \
        {c: {
            exp: s.count_errors()
            for exp, s in experiments.items()
        }
            for c, experiments in solutions.items()
        }

    objectives = \
        {c: {
            exp: s.calculate_objective()
            for exp, s in experiments.items()
            if feasibility[c][exp] == 0
        }
            for c, experiments in solutions.items()
        }

    f_experiment = 'heuristic1800'
    items_area = {c: s[f_experiment].get_items_area() for c, s in solutions.items()}
    instance_case1 = solutions['A1'][f_experiment]
    jumbo_area = instance_case1.get_param('widthPlates') * instance_case1.get_param('heightPlates')

    table_items = pd.DataFrame.from_dict(items_area, orient='index').reset_index().\
        rename(columns={0: 'items', 'index': 'case'})
    renames = {'index': 'case', 'variable': 'experiment'}
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
    # experiment = 'clust1_20180706'
    exp_paths = get_experiments_paths(pm.PATHS['results'])
    solutions = get_solutions(exp_paths)
    solutions = sd.SuperDict(solutions).get_property(experiment)
    if case is not None:
        solutions.filter([case])
    destination = pm.PATHS['root'] + 'graphs/'
    if os.path.exists(destination):
        shutil.rmtree(destination)
    os.makedirs(destination)
    for k, v in solutions.items():
        path = destination + k
        os.makedirs(path)
        v.graph_solution(path, dpi=dpi)

if __name__ == "__main__":
    benchmarking('obj')
    dominant_experiments()