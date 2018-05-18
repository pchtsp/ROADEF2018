path_root = '/home/pchtsp/Documents/projects/ROADEF2018/'

PATHS = {
    'root': path_root
    ,'results': path_root + 'results/'
    ,'experiments': path_root + 'results/' + "heuristic/"
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
    , 'solver': "HEUR"   # CPLEXPY, CPLEX_CMD, GUROBI, HEUR
    , 'integer': False
    , 'path': PATHS['experiments']
    , 'case_name': 'A1'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 2
    , 'heur_weights': {'space': 100, 'seq': 100000, 'defects': 1000}
    , 'heur_params': {'main_iter': 1000, 'max_iter': 100,
                      'temperature': 1000, 'try_rotation': True}

    # , 'cluster_tolerance': 50
}