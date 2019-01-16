import os
import package.params as pm
import re
import shutil


def separate_cases(name, data_dir=pm.PATHS['data'], results_dir=pm.PATHS['results'], cases=None):
    destination = results_dir + name + '/'
    if not os.path.exists(destination):
        os.makedirs(destination)

    files = os.listdir(data_dir)
    if cases is None:
        cases = ["A{}".format(n) for n in range(1, 21)]
    for case in cases:
        _files = [os.path.join(data_dir, f) for f in files if f.startswith(case + '_')]
        _dest = destination + case + '/'
        if not os.path.exists(_dest):
            os.makedirs(_dest)
        for _f in _files:
            shutil.copy(_f, _dest)
    # cases = [r for f in files for r in re.findall('A\d+', f)]
    # set(cases)

############################################
############################################

def move_case_checker(experiment):
    origin = pm.PATHS['results'] + experiment
    cases = ["A{}".format(n) for n in range(1, 21)]
    _files = [os.path.join(origin, f, f + '_{}.csv'.format(f_type))
              for f in cases for f_type in ['solution']]
    destination = pm.PATHS['checker_data']
    for _f in _files:
        if os.path.exists(_files[0]):
            shutil.copy(_f, destination)


if __name__ == "__main__":
    separate_cases('ubuntu_20180706')


