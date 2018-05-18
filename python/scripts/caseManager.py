import os
import package.params as pm
import re
import shutil


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



