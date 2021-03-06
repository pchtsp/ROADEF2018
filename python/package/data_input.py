import os
import pandas as pd
import re
import numpy as np
import copy
import json
import pickle
import package.params as pm
import package.superdict as sd
import package.auxiliar as aux
import pprint as pp


def dir_has_solution(path):
    for p in os.listdir(path):
        if re.search(r"solution.csv$", p):
            return True
    return False


def read_files_into_tables(filenames):
    for file, path in filenames.items():
        if not os.path.exists(path):
            raise FileNotFoundError('Path {} does not exist'.format(path))
    return sd.SuperDict({file: pd.read_csv(path, sep=';') for file, path in filenames.items()})


def get_model_data(case_name, path=pm.PATHS['data']):
    input_files = ['batch', 'defects']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = read_files_into_tables(filenames)

    defects = \
        tables['defects']. \
            assign(index=tables['defects'].DEFECT_ID,
                   index2=tables['defects'].PLATE_ID). \
            set_index(['index2', 'index']).\
            to_dict(orient='index')
    defects = sd.SuperDict(defects).to_tuplist().to_dict(result_col=[2], indices=[0])

    batch = \
        tables['batch'].\
            assign(index=tables['batch'].ITEM_ID,
                   index2=tables['batch'].STACK).\
            set_index(['index2', 'index']).\
            to_dict(orient='index')
    batch = sd.SuperDict(batch).to_tuplist().to_dict(result_col=[2], indices=[0])
    for stack in batch:
        batch[stack].sort(key=lambda x: x['SEQUENCE'])

    params_filenames = {file: path + file + '.csv' for file in ['global_param']}
    if os.path.exists(params_filenames['global_param']):
        params = read_files_into_tables(params_filenames)

        parameters = \
            params['global_param'].\
            set_index('NAME')['VALUE'].\
            to_dict()
    else:
        parameters = {
            'nPlates': 100
            , 'widthPlates': 6000
            , 'heightPlates': 3210
            , 'minXX': 100
            , 'maxXX': 3500
            , 'minYY': 100
            , 'minWaste': 20
        }

    return sd.SuperDict({'batch': batch,
                         'defects': defects,
                         'global_param': sd.SuperDict.from_dict(parameters)})


def get_model_solution(case_name, path=pm.PATHS['checker_data'], filename="solution"):
    # case_name = "A0"
    # path = pm.PATHS['checker_data']
    input_files = [filename]
    try:
        filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
        tables = read_files_into_tables(filenames)
    except:
        filenames = {file: path + file + '.csv' for file in input_files}
        tables = read_files_into_tables(filenames)
    result = \
        tables[filename].\
            assign(index=tables[filename].NODE_ID).\
            set_index('index').\
            to_dict(orient='index')
    return sd.SuperDict(result)


def load_data(path, file_type=None):
    if file_type is None:
        splitext = os.path.splitext(path)
        if len(splitext) == 0:
            raise ImportError("file type not given")
        else:
            file_type = splitext[1][1:]
    if file_type not in ['json', 'pickle']:
        raise ImportError("file type not known: {}".format(file_type))
    if not os.path.exists(path):
        raise FileNotFoundError('File {} does not exist'.format(path))
    if file_type == 'pickle':
        with open(path, 'rb') as f:
            return pickle.load(f)
    if file_type == 'json':
        with open(path, 'r') as f:
            return json.load(f)


def export_data(path, obj, name=None, file_type="pickle"):
    if not os.path.exists(path):
        os.mkdir(path)
    if name is None:
        name = aux.get_timestamp()
    path = os.path.join(path, name + "." + file_type)
    if file_type == "pickle":
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
    if file_type == 'json':
        with open(path, 'w') as f:
            json.dump(obj, f)
    return True


# def write_table(data, path=pm.PATHS['experiments'], filename='solution', ext='csv'):
#     data.to_csv(path + aux.get_timestamp() + '/{}.{}'.format(filename, ext))

if __name__ == "__main__":

    case_name = "A0"
    checker_data_path = pm.PATHS['checker_data']
    # all_files = ['batch', 'defects', 'solution']
    all_files = ['batch', 'defects']
    filenames = {file: checker_data_path + case_name + '_' + file + '.csv' for file in all_files}
    tables = {file: pd.read_csv(fn, sep=';') for file, fn in filenames.items()}
    pp.pprint(get_model_data('A1'))

    width, height = (6000, 3210)
