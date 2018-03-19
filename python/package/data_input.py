import os
import pandas as pd
import re
import numpy as np
import copy
import json
import pickle
import package.params as pm
import package.auxiliar as aux


def get_model_data(case_name):
    # case_name = "A3"
    path = pm.PATHS['data']
    input_files = ['batch', 'defects']
    filenames = {file: path + case_name + '_'+ file + '.csv' for file in input_files}
    tables = {file: pd.read_csv(fn, sep=';') for file, fn in filenames.items()}

    defects = \
        tables['defects'].set_index('DEFECT_ID').\
            to_dict(orient='index')

    defects_plate = {int(v['PLATE_ID']): {} for v in defects.values()}
    for k, v in defects.items():
        defects_plate[v['PLATE_ID']][k] = v

    batch = \
        tables['batch'].set_index('ITEM_ID').\
            to_dict(orient='index')

    return {'batch': batch, 'defects': defects_plate}


# checker example
case_name = "A0"
checker_data_path = pm.PATHS['checker_data']
all_files = ['batch', 'defects', 'solution']
filenames = {file: checker_data_path + case_name + '_' + file + '.csv' for file in all_files}
tables = {file: pd.read_csv(fn, sep=';') for file, fn in filenames.items()}


width, height = (6000, 3210)
