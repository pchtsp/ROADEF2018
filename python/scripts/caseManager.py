import os
import package.params as pm
import re
import shutil


def separate_cases():
    destination = pm.PATHS['results'] + 'heuristic/'

    files = os.listdir(pm.PATHS['data'])
    cases = ["A{}".format(n) for n in range(1, 21)]
    for case in cases:
        _files = [os.path.join(pm.PATHS['data'], f) for f in files if f.startswith(case+'_')]
        _dest = destination + case + '/'
        if not os.path.exists(_dest):
            os.makedirs(_dest)
        for _f in _files:
            shutil.copy(_f, _dest)
    # cases = [r for f in files for r in re.findall('A\d+', f)]
    # set(cases)

############################################
############################################

def move_case_checker():
    origin = pm.PATHS['results'] + 'heuristic/'
    cases = ["A{}".format(n) for n in range(1, 21)]
    _files = [os.path.join(origin, f, f + '_{}.csv'.format(f_type))
              for f in cases for f_type in ['solution', 'defects', 'batch']]
    destination = pm.PATHS['checker_data']
    for _f in _files:
        if os.path.exists(_files[0]):
            shutil.copy(_f, destination)