path_root = '/home/pchtsp/Documents/projects/ROADEF2018/'

PATHS = {
    'root': path_root
    ,'results': path_root + 'results/'
    ,'experiments': path_root + 'results/' + "heuristic1800/"
    ,'img': path_root + "OPTIMA/img/"
    ,'latex': path_root + "OPTIMA/latex/"
    ,'data': path_root + "resources/data/dataset_A/"
    ,'checker_data': path_root + "resources/checker/instances_checker/"
}

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
    , 'case_name': 'A14'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 2
    , 'heur_weights': {'space': 0.0001, 'seq': 100000, 'defects': 100}
    # , 'heur_weights': {'space': 10, 'seq': 100000, 'defects': 1000}
    , 'heur_params': {'main_iter': 5, 'max_iter': 1000,
                      'temperature': 8000, 'try_rotation': False,
                      'max_candidates': 30, 'extra_jumbos': 2,
                      'cooling_rate': 0.02}
    , 'debug': False

    # , 'cluster_tolerance': 50
}