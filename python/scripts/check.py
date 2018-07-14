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
    experiments = {f: path + f + '/' for f in os.listdir(path) if not f.startswith('old')}
    return {c: {exp: path + c + '/' for exp, path in experiments.items()} for c in cases}


def get_solutions(exp_paths):
    return \
        {c: {
            exp: sol.Solution.from_io_files(path=p, case_name=c)
            for exp, p in experiments.items() if di.dir_has_solution(p)
        }
            for c, experiments in exp_paths.items()
        }


def benchmarking():
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.DataFrame.from_csv(others_path, sep=';')
    table1.columns = ['others', 'TEAM']
    exp_paths = get_experiments_paths(pm.PATHS['results'])

    solutions = get_solutions(exp_paths)

    objectives = \
        {c: {
            exp: s.calculate_objective()
            for exp, s in experiments.items()
        }
            for c, experiments in solutions.items()
        }

    feasibility = \
        {c: {
            exp: s.count_errors()
            for exp, s in experiments.items()
        }
            for c, experiments in solutions.items()
        }

    f_experiment = [*[*exp_paths.values()][0]][1]
    items_area = {c: s[f_experiment].get_items_area() for c, s in solutions.items()}

    table_items = pd.DataFrame.from_dict(items_area, orient='index').reset_index().\
        rename(columns={0: 'items', 'index': 'case'})
    renames = {'index': 'case', 'variable': 'experiment'}
    table_obj = pd.DataFrame.from_dict(objectives, orient='index').reset_index().\
        melt(id_vars='index', value_name='obj').rename(columns=renames)
    table_feas = pd.DataFrame.from_dict(feasibility, orient='index').reset_index().\
        melt(id_vars='index', value_name='feas').rename(columns=renames)
    others = table1[['others']].reset_index().rename(columns={'INSTANCE': 'case'})
    others['experiment'] = 'others'
    table_obj = table_obj.append(others.rename(columns={'others': 'obj'}))

    # df_final = reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)
    params = {'how': 'outer'}
    summary = \
        table_obj.\
            merge(others, **params).\
            merge(table_items, **params).\
            merge(table_feas, **params)

    summary['dif'] = (summary.obj - summary.others) / summary.others * 100
    summary[['case', 'experiment', 'obj']].\
        pivot(index='case', columns='experiment', values='dif').plot.bar()


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
    benchmarking()