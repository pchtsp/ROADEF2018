import package.solution as sol
import package.heuristic as heur
import package.model as mod
import package.params as pm
import numpy as np
import pprint as pp
import package.superdict as sd


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

    e = '201804271903/'
    # e = 'multi2/A6/'
    path = pm.PATHS['experiments'] + e
    # path = pm.PATHS['results'] + e
    solution = sol.Solution.from_io_files(path=path, solutionfile='solution_heur')
    # solution.graph_solution(path, name="edited", dpi=50)
    solution.check_all()


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

if __name__ == "__main__":
    stats()