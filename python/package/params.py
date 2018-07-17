def calculate_paths_root(root):
    return {
    'root': root
    ,'img': root + "OPTIMA/img/"
    ,'latex': root + "OPTIMA/latex/"
    ,'data': root + "resources/data/dataset_A/"
    ,'checker_data': root + "resources/checker/instances_checker/"
}


def calculate_paths_results(results):
    return {
    'results': results
    ,'experiments': results + "test/"
    }


root = '/home/pchtsp/Documents/projects/ROADEF2018/'
results = '/home/pchtsp/Dropbox/ROADEF2018/'

PATHS = {**calculate_paths_root(root), **calculate_paths_results(results)}

ORIENTATIONS = [0, 1]
ORIENT_H = 0
ORIENT_V = 1

cut_level_next_o = {
    0: ORIENT_V
    , 1: ORIENT_H
    , 2: ORIENT_V
    , 3: ORIENT_H
}

OPTIONS = {
    'timeLimit': 3600
    , 'gap': 0.1
    , 'solver': "HEUR"  # CPLEXPY, CPLEX_CMD, GUROBI, HEUR
    , 'integer': False
    , 'path': PATHS['experiments']
    , 'case_name': 'A20'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 2
    # , 'heur_weights': {'space': 10, 'seq': 100000, 'defects': 1000}
    , 'heur_params': {'main_iter': 5, 'max_iter': 10,
                      'temperature': 8000, 'try_rotation': False,
                      'max_candidates': 3, 'extra_jumbos': 0,
                      'cooling_rate': 0.001,
                      'change_first': False, 'tolerance': None,
                      'rotation_probs': [0.99, 0.01, 0, 0],
                      'level_probs': [0.4, 0.5, 0.1],
                      'weights': {'space': 0.00001, 'seq': 100000, 'defects': 1000}
                      }
    , 'heur_optim': {
        'try_rotation': True,
        'max_candidates': 10,
        'rotation_probs': [0.99, 0.01, 0, 0],
        'weights': {'space': 1000, 'seq': 20000, 'defects': 3000}
    }
    , 'debug': False
    , 'graph': True
    # putting a tolerance on the heuristic seems horrible for finding good solutions.
    # putting change_first is also a very bad idea.
    # , 'cluster_tolerance': 50
}