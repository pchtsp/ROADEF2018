import os
import pandas as pd
import re
import numpy as np
import copy
import json
import pickle
import package.params as pm
import package.auxiliar as aux


def read_files_into_tables(filenames):
    for file, path in filenames.items():
        if not os.path.exists(path):
            raise FileNotFoundError('Path {} does not exist'.format(path))
    return {file: pd.read_csv(path, sep=';') for file, path in filenames.items()}


def get_model_data(case_name, path=pm.PATHS['data']):
    # case_name = "A3"
    # path = pm.PATHS['data']
    input_files = ['batch', 'defects']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = read_files_into_tables(filenames)

    defects = \
        tables['defects'].set_index('DEFECT_ID').\
            to_dict(orient='index')

    batch = \
        tables['batch'].set_index('ITEM_ID').\
            to_dict(orient='index')

    parameters = {'plate_width': 6000, 'plate_height': 3210}

    return {'batch': batch, 'defects': defects, 'parameters': parameters}


def get_model_solution(case_name, path=pm.PATHS['checker_data']):
    # case_name = "A0"
    # path = pm.PATHS['checker_data']
    input_files = ['solution']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = read_files_into_tables(filenames)
    return tables['solution'].set_index('NODE_ID').to_dict(orient='index')


if __name__ == "__main__":

    case_name = "A0"
    checker_data_path = pm.PATHS['checker_data']
    # all_files = ['batch', 'defects', 'solution']
    all_files = ['batch', 'defects']
    filenames = {file: checker_data_path + case_name + '_' + file + '.csv' for file in all_files}
    tables = {file: pd.read_csv(fn, sep=';') for file, fn in filenames.items()}


    width, height = (6000, 3210)
