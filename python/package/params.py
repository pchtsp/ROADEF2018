def calculate_paths_root(root):
    return {
    'root': root
    ,'img': root + "img/"
    ,'latex': root + "latex/"
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
    'timeLimit': 600
    , 'gap': 0.1
    , 'solver': "HEUR"  # CPLEXPY, CPLEX_CMD, GUROBI, HEUR
    , 'integer': False
    , 'input_path': PATHS['experiments']
    , 'output_path': PATHS['experiments']
    , 'output_file_name': 'solution'
    , 'case_name': 'A14'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 2
    # , 'heur_weights': {'space': 10, 'seq': 100000, 'defects': 1000}
    , 'heur_params': {'main_iter': 5, 'max_iter': 500,
                      'temperature': 3000,
                      'max_candidates': 5, 'extra_jumbos': 0,
                      'cooling_rate': 0.005,
                      'change_first': False, 'tolerance': None,
                      'try_rotation': True,
                      'rotation_probs': [0.50, 0.50, 0, 0],
                      'rotation_tries': 2,
                      'level_probs': [0.4, 0.4, 0.2],
                      'weights': {'space': 0.00001, 'seq': 40000, 'defects': 40000}
                      }
    , 'heur_remake': {
        'iterations_initial': 100,
        'iterations_remake': 0,
        'rotation': [0.50, 0.50, 0, 0],
        'num_trees': [0.60, 0.1, 0.1, 0.1, 0.1],
    }
    , 'heur_optim': {
        'try_rotation': True,
        'max_candidates': 5,
        'rotation_probs': [0.5, 0.5, 0, 0],
        'weights': {'space': 40000, 'seq': 40000, 'defects': 40000}
    }
    , 'debug': False
    , 'graph': False
    # putting a tolerance on the heuristic seems horrible for finding good solutions.
    # putting change_first is also a very bad idea.
    # , 'cluster_tolerance': 50
}