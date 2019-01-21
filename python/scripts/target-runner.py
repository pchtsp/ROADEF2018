#!/home/pchtsp/Documents/projects/ROADEF2018/python/venv/bin/python3
###############################################################################
# This script is the command that is executed every run.
# Check the examples in examples/
#
# This script is run in the execution directory (execDir, --exec-dir).
#
# PARAMETERS:
# argv[1] is the candidate configuration number
# argv[2] is the instance ID
# argv[3] is the seed
# argv[4] is the instance name
# The rest (argv[5:]) are parameters to the run
#
# RETURN VALUE:
# This script should print one numerical value: the cost that must be minimized.
# Exit with 0 if no error, with 1 in case of error
###############################################################################

import datetime
import os.path
import re
import subprocess
import sys
import collections
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import scripts.exec as exec
import package.tuplist as tl
import package.params as params

## This example is for the ACOTSP software. Compare it with
## examples/acotsp/target-runner
# exe = "~/bin/executable"
# fixed_params = ' --tries 1 --time 10 --quiet '

def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]

if len(sys.argv) < 5:
    print("\nUsage: ./target-runner.py <candidate_id> <instance_id> <seed> <instance_path_name> <list of parameters>\n")
    sys.exit(1)

# Get the parameters as command line arguments.
candidate_id = sys.argv[1]
instance_id = sys.argv[2]
seed = sys.argv[3]
instance = sys.argv[4]
# cand_params = sys.argv[5:]

# assuming cand_params is a dictionary with key-value.
cand_params = dict(map(lambda x: x.lstrip('-').split('='),sys.argv[5:]))
f_key = dict(h='heur_params', rem='heur_remake')

options_tup = tl.TupList()
for key, value in cand_params.items():
    tup = key.split('__')
    if '.' in value:
        tup += [float(value)]
    else:
        tup += [int(value)]
    if tup[0] in f_key:
        tup[0] = f_key[tup[0]]
    options_tup.append(tup)

options = params.OPTIONS
new_options = options_tup.to_dict(result_col=-1, is_list=False, indices_heter=True).to_dictdict()

dict_merge(options, new_options)
options['seed'] = int(seed)
options['timeLimit'] = options['g']['timeLimit']

location, case = os.path.split(instance)
cases = [case]
options['input_path'] = location + '/'
options['output_path'] = './'
options['output_file_name'] = 'solution.csv'

result = exec.solve(options)

cost = 1000000000
if result is not None:
    cost = result.evaluate_solution(options['heur_params']['weights'])
print(cost)

sys.exit(0)
