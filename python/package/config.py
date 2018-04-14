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

    def config_gurobi(self):
        # GUROBI parameters: http://www.gurobi.com/documentation/7.5/refman/parameters.html#sec:Parameters
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        result_path = self.path + 'results.sol'.format()
        log_path = self.path + 'results.log'
        options = [('TimeLimit', self.timeLimit),
                   ('ResultFile', result_path),
                   ('LogFile', log_path),
                   ('MIPGap', self.gap)]
        return pl.GUROBI_CMD(options=options)

    def config_cplex(self):
        # CPLEX parameters: https://www.ibm.com/support/knowledgecenter/en/SSSA5P_12.6.0/ilog.odms.cplex.help/CPLEX/GettingStarted/topics/tutorials/InteractiveOptimizer/settingParams.html
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        log_path = self.path + 'results.log'
        options = ['set logfile {}'.format(log_path),
                   'set timelimit {}'.format(self.timeLimit),
                   'set mip tolerances mipgap {}'.format(self.gap)]
        return pl.CPLEX_CMD(options=options, keepFiles=1)

    def config_cplexpy(self):
        log_path = self.path + 'results.log'
        return pl.CPLEX_PY(timeLimit=self.timeLimit, epgap=self.gap, logfilename=log_path)

    def config_choco(self):
        # CHOCO parameters https://github.com/chocoteam/choco-parsers/blob/master/MPS.md
        options = [('-tl', self.timeLimit * 1000), ('-p', 1)]
        return pl.PULP_CHOCO_CMD(options=options, keepFiles=1, msg=0)

    def get_solver(self):
        mapping = \
            {
                'GUROBI': self.config_gurobi()
                ,'CPLEX': self.config_cplex()
                ,'CHOCO': self.config_choco()
                ,'CPLEXPY': self.config_cplexpy()
            }
        return mapping.get(self.solver, None)

