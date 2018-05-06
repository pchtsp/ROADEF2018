import os
import datetime as dt

path_root = '/home/pchtsp/Documents/projects/ROADEF2018/'

PATHS = {
    'root': path_root
    ,'results': path_root + 'results/'
    ,'experiments': path_root + 'results/' + "experiments/"
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
    'timeLimit': 600
    , 'gap': 0.1
    , 'solver': "CPLEXPY"
    , 'integer': False
    , 'path': os.path.join(
        PATHS['experiments'],
        dt.datetime.now().strftime("%Y%m%d%H%M")
    ) + '/'
    , 'case_name': 'A6'
    , 'max_plates': 10
    , 'max_width': 6000//4
    , 'max_items': 15
    , 'max_iters': 200000
    , 'ratio_plate_size': 1
    # , 'cluster_tolerance': 50
}