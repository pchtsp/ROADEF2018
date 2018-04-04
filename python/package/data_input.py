import os
import pandas as pd
import re
import numpy as np
import copy
import json
import pickle
import package.params as pm
import package.superdict as sd
import pprint as pp


def read_files_into_tables(filenames):
    for file, path in filenames.items():
        if not os.path.exists(path):
            raise FileNotFoundError('Path {} does not exist'.format(path))
    return sd.SuperDict({file: pd.read_csv(path, sep=';') for file, path in filenames.items()})


def get_model_data(case_name, path=pm.PATHS['data']):
    # case_name = "A3"
    # path = pm.PATHS['data']
    # TODO: maybe import parameters from file if exists?
    input_files = ['batch', 'defects']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = read_files_into_tables(filenames)

    defects = \
        tables['defects'].set_index('DEFECT_ID').\
            to_dict(orient='index')

    batch = \
        tables['batch'].set_index('ITEM_ID').\
            to_dict(orient='index')

    parameters = {
        'nPlates': 100
        , 'widthPlates': 6000
        , 'heightPlates': 3210
        , 'minXX': 100
        , 'maxXX': 3500
        , 'minYY': 100
        , 'minWaste': 20
    }

    return sd.SuperDict({'batch': sd.SuperDict.from_dict(batch),
                         'defects': sd.SuperDict.from_dict(defects),
                         'parameters': sd.SuperDict.from_dict(parameters)})


def get_model_solution(case_name, path=pm.PATHS['checker_data']):
    # case_name = "A0"
    # path = pm.PATHS['checker_data']
    input_files = ['solution']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = read_files_into_tables(filenames)
    result = tables['solution'].set_index('NODE_ID').to_dict(orient='index')
    return sd.SuperDict(result)


if __name__ == "__main__":

    case_name = "A0"
    checker_data_path = pm.PATHS['checker_data']
    # all_files = ['batch', 'defects', 'solution']
    all_files = ['batch', 'defects']
    filenames = {file: checker_data_path + case_name + '_' + file + '.csv' for file in all_files}
    tables = {file: pd.read_csv(fn, sep=';') for file, fn in filenames.items()}
    pp.pprint(get_model_data('A1'))

    width, height = (6000, 3210)
