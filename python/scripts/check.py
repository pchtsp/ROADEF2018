import package.solution as sol
import package.heuristic as heur
import package.model as mod
import package.params as pm
import numpy as np
import pprint as pp
import package.superdict as sd
import pandas as pd


def test1():

    e = '201804142339/'
    solution = sol.Solution.from_io_files(path=pm.PATHS['experiments'] + e)
    # solution = sol.Solution.from_io_files(path=pm.PATHS['checker_data'], case_name='A0')
# print(self.trees[0].get_ascii(show_internal=True, attributes=['NODE_ID', 'PARENT']))
################################################


def test2():

    model = mod.Model.from_input_files(case_name='A2')

    # r = model.get_smallest_items()
    pieces = model.flatten_stacks(in_list=True)
    min_length = pieces[1][1] + 1
    filter_out_pos = set()
    for pos, i in enumerate(pieces):
        if i[1] >= min_length:
            filter_out_pos.add(pos)
        else:
            min_length = i[1]
    # [i for i in r if i[1] <= r[0][1]]


def test3():
    # e = 'multi2/A2/'
    e = '201804240009/'
    path = pm.PATHS['experiments'] + e
    # path = pm.PATHS['results'] + e
    # solution = mod.Model.from_io_files(path=path)
    solution = mod.Model.from_input_files(path=path)
    # solution.export_input_data()
    cuts = solution.import_cuts(path)
    sorted(cuts.items(), reverse=True)
    solution.load_solution(sd.SuperDict(cuts))
    solution.graph_solution()
    solution.draw(['WIDTH', 'HEIGHT', 'TYPE'], pos=4)
    missing = solution.check_all()['demand'].tolist()
    # solution.get_pieces_by_type()

    demanded_plates = solution.flatten_stacks()
    missing_plates = demanded_plates.filter(missing)
    plates = solution.get_plate_production()

    sorted(demanded_plates.values_l())
    sorted(missing_plates.values_l())
    sorted(plates)


def test4():
    e = '201804242310/'
    path = pm.PATHS['experiments'] + e
    # path = pm.PATHS['results'] + e
    solution = mod.Model.from_io_files(path=path)
    solution.draw(['name', 'TYPE'], pos=1)
    solution.graph_solution()
    self = solution


def test5():
    e = '201805090409/'
    path = pm.PATHS['experiments'] + e
    solution = mod.Model.from_io_files(path=path)
    # plates = solution.arrange_plates()
    checks = solution.check_all()
    len(checks['defects'])
    len(checks['sequence'])
    checks.keys()
    solution.graph_solution(path)


def test6():
    import package.heuristic as heur
    import package.data_input as di
    path = '/home/pchtsp/Documents/projects/ROADEF2018/results/heuristic1800/A13/'
    self = heur.ImproveHeuristic.from_io_files(path=path)

    # re-execute

    options = di.load_data(path=path + 'options.json')
    options['timeLimit'] = 300
    # options['debug'] = True
    self.solve(options, warm_start=True)
    # self.trees = self.best_solution
    self.correct_plate_node_ids()
    # self.export_solution(path=path, prefix=options['case_name'] + '_', name="solution")
    # self.graph_solution(path, name="edited", dpi=50)
    # print(self.check_sequence(solution=self.best_solution


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


def benchmarking():
    others_path = pm.PATHS['data'] + 'solutions_A.csv'
    table1 = pd.DataFrame.from_csv(others_path, sep=';')
    table1.columns = ['others', 'TEAM']

    cases = ["A{}".format(n) for n in range(1, 21)]
    experiments = {'1': pm.PATHS['results'] + "heuristic1/",
                   '2': pm.PATHS['results'] + "heuristic2/"}
    exp_paths = {c: {exp: path + c + '/' for exp, path in experiments.items()} for c in cases}

    solutions = \
        {c: {
            exp: sol.Solution.from_io_files(path=p, case_name=c)
            for exp, p in experiments.items()
        }
            for c, experiments in exp_paths.items()
        }

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

    f_experiment = [*experiments.keys()][0]
    items_area = {c: s[f_experiment].get_items_area() for c, s in solutions.items()}

    table_items = pd.DataFrame.from_dict(items_area, orient='index').\
        rename(columns={0: 'items'})
    table_obj = pd.DataFrame.from_dict(objectives, orient='index').\
        rename(columns={'1': 'obj_1', '2': 'obj_2'})
    table_feas = pd.DataFrame.from_dict(feasibility, orient='index')
    # table2.columns = ['INSTANCE', 'me']

    # df_final = reduce(lambda left, right: pd.merge(left, right, on='name'), dfs)
    params = {'left_index': True, 'right_index': True}
    summary = \
        table1[['others']].\
            merge(table_obj, **params).\
            merge(table_items, **params).\
            merge(table_feas, **params)

    summary['dif_1'] = (summary.obj_1 - summary.others) / summary.others * 100
    summary['dif_2'] = (summary.obj_2 - summary.others) / summary.others * 100
    # summary[["INSTANCE", 'others', 'me', 'dif']]

    # {solutions['A4']['1800'].}



if __name__ == "__main__":
    test6()