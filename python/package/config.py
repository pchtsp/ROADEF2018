# /usr/bin/python3

import os
import package.auxiliar as aux
import pulp as pl
import tempfile
from os import dup, dup2, close
import package.params as params


class Config(object):

    def __init__(self, options):
        if options is None:
            options = {}

        default_options = {
            'timeLimit': 300
            , 'gap': 0
            , 'solver': "GUROBI"
        }

        # the following merges the two configurations (replace into):
        options = {**default_options, **options}

        self.gap = options['gap']
        self.path = options['path']
        self.timeLimit = options['timeLimit']
        self.solver = options['solver']

    def config_cbc(self):
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
        return \
            ["presolve on",
             "gomory on",
             "knapsack on",
             "probing on",
             "ratio {}".format(self.gap),
             "sec {}".format(self.timeLimit)]

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        result_path = self.path + 'results.sol'.format()
        log_path = self.path + 'results.log'
        return [('TimeLimit', self.timeLimit),
                ('ResultFile', result_path),
                ('LogFile', log_path),
                ('MIPGap', self.gap)]

    def config_cplex(self):
        # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
        return ['set logfile {}'.format(log_path),
                'set timelimit {}'.format(self.timeLimit),
                'set mip tolerances mipgap {}'.format(self.gap)]

    def config_choco(self):
        # CHOCO parameters https://github.com/chocoteam/choco-parsers/blob/master/MPS.md
        return [('-tl', self.timeLimit * 1000),
                ('-p', 1)]

    def solve_model(self, model):
        if self.solver == "GUROBI":
            return model.solve(pl.GUROBI_CMD(options=self.config_gurobi()))
        if self.solver == "CPLEX":
            return model.solve(pl.CPLEX_CMD(options=self.config_cplex(), keepFiles=1))
        if self.solver == "CHOCO":
            return model.solve(pl.PULP_CHOCO_CMD(options=self.config_choco(), keepFiles=1, msg=0))
        with tempfile.TemporaryFile() as tmp_output:
            orig_std_out = dup(1)
            dup2(tmp_output.fileno(), 1)
            result = model.solve(pl.PULP_CBC_CMD(options=self.config_cbc(), msg=True, keepFiles=1))
            dup2(orig_std_out, 1)
            close(orig_std_out)
            tmp_output.seek(0)
            logFile = [line.decode('ascii') for line in tmp_output.read().splitlines()]
        with open(self.path + "results.log", 'w') as f:
            for item in logFile:
                f.write("{}\n".format(item))
        return result

