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

# FIXME: group together everything that needs to be edited by the user and put
# in functions everything that does NOT to be edited.

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

# cand_params = dict(h__main_iter=1, h__cooling_rate=4, rem__rotation=6, h__weights__space=5)

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

# Build the command, run it and save the output to a file,
# to parse the result from it.
# 
# Stdout and stderr files have to be opened before the call().
#
# Exit with error if something went wrong in the execution.
# exe = os.path.expanduser(exe)
# command = [exe] + fixed_params.split() + ["-i"] + [instance] + ["--seed"] + [seed] + cand_params

# Define the stdout and stderr files.
# out_file = "c" + str(candidate_id) + "-" + str(instance_id) + str(seed) + ".stdout"
# err_file = "c" + str(candidate_id) + "-" + str(instance_id) + str(seed) + ".stderr"

# def target_runner_error(msg):
#     now = datetime.datetime.now()
#     print(str(now) + " error: " + msg)
#     sys.exit(1)
#
# def check_executable(fpath):
#     fpath = os.path.expanduser(fpath)
#     if not os.path.isfile(fpath):
#         target_runner_error(str(fpath) + " not found")
#     if not os.access(fpath, os.X_OK):
#         target_runner_error(str(fpath) + " is not executable")
#
# check_executable (exe)

# outf = open(out_file, "w")
# errf = open(err_file, "w")
result = exec.solve(options)
# outf.close()
# errf.close()

# if return_code != 0:
#     target_runner_error("command returned code " + str(return_code))

cost = 10000000
if result is not None:
    errors = result.count_errors()
    cost = result.calculate_objective() + errors*10000000
print(cost)

# os.remove(out_file)
# os.remove(err_file)
sys.exit(0)