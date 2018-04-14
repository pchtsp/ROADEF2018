import package.solution as sol
import package.params as pm
import os
import re
import shutil
import numpy as np


def clean_experiments(path, clean=True, regex=""):
    """
    loads and cleans all experiments that are incomplete
    :param path: path to experiments
    :param clean: if set to false it only shows the files instead of deleting them
    :param regex: optional regex filter
    :return: deleted experiments
    """
    exps_paths = [os.path.join(path, f) + '/' for f in os.listdir(path)
                  if os.path.isdir(os.path.join(path, f))
                  if re.search(regex, f)
                  ]
    to_delete = []
    for e in exps_paths:
        try:
            sol.Solution.from_io_files(path=e)
        except:
            to_delete.append(True)
        else:
            to_delete.append(False)
    exps_to_delete = np.array(exps_paths)[to_delete]
    if clean:
        for ed in exps_to_delete:
            shutil.rmtree(ed)
    return exps_to_delete
#
# path = '/home/pchtsp/Documents/projects/ROADEF2018/results/experiments/201804142346/'
# sol.Solution.from_io_files(path=path)

t = clean_experiments(pm.PATHS['experiments'], regex='^2018', clean=False)
t.sort()
print(t, len(t))

