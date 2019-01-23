def calculate_paths_root(root, data_set='A'):
    path_checker = root + "resources/checker/"
    return {
    'root': root
    ,'img': root + "img/"
    ,'latex': root + "latex/"
    ,'data': root + "resources/data/dataset_{}/".format(data_set)
    ,'checker': path_checker
    ,'checker_data': path_checker + "instances_checker/"
}


def calculate_paths_results(results):
    return {
    'results': results
    ,'experiments': results + "test/"
    }

base = '/home'
# base = 'C:/Users'
root = base + '/pchtsp/Documents/projects/ROADEF2018/'
results = base + '/pchtsp/Dropbox/ROADEF2018/'

PATHS = {**calculate_paths_root(root), **calculate_paths_results(results)}

OPTIONS = {
    'timeLimit': 600
    , 'gap': 0.1
    , 'solver': "HEUR"  # CPLEXPY, CPLEX_CMD, GUROBI, HEUR
    , 'integer': False
    , 'input_path': PATHS['experiments']
    , 'output_path': PATHS['experiments']
    , 'output_file_name': None
    , 'case_name': 'A15'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 2
    , 'num_processors': 24
    , 'multiprocess': True
    , 'seed': 42
    , 'warm_start': False
    , 'heur_params': {'main_iter': 5, 'max_iter': 1000,
                      'temperature': 10000,
                      'max_candidates': 100, 'extra_jumbos': 0,
                      'cooling_rate': 0.0005,
                      'change_first': False, 'tolerance': None,
                      'try_rotation': 1,
                      'rotation_probs': [0.70, 0.30, 0, 0],
                      'rotation_tries': 2,
                      'level_probs': [0.3, 0.3, 0.3, 0.1],
                      'weights': {'space': 0.00001, 'seq': 40000, 'defects': 40000, 'wastes': 40000},
                      'cuts_prob': 0.5,
                      'clean_cuts_prob': 0.5
                      }
    , 'heur_remake': {
        'iterations_initial': 1000,
        'iterations_remake': 50,
        'rotation': [0.50, 0.50, 0, 0],
        'num_trees': [0.20, 0.20, 0.20, 0.20, 0.20],
        'options': ['best', 'partial', 'restart'],
        'probability': [0.05, 0.9, 0.05],
        'max_no_improve': 10,
        'prob_accept_worse': 0.1,
        'prob_accept_worse_def': 0.2,
        'prob_try_improve': 0.5,
        'prob_ref_is_last': 0.2
    }
    , 'heur_optim': {
        # 'try_rotation': True,
        # 'max_candidates': 5,
        # 'rotation_probs': [0.5, 0.5, 0, 0],
        # 'weights': {'space': 400, 'seq': 20000, 'defects': 40000, 'wastes': 40000}
    }
    , 'debug': False
    , 'graph': False
    # putting a tolerance on the heuristic seems horrible for finding good solutions.
    # putting change_first is also a very bad idea.
    # , 'cluster_tolerance': 50
}