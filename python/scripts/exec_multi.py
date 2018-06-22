import package.data_input as di
import package.model as md
import package.params as pm
import scripts.exec as exec
import os
import logging as log
import sys


if __name__ == "__main__":

    # somewhat problematic:
    incomplete = ['A2', 'A13', 'A15', 'A6', 'A14']
    # problematic:
    hard = ['A14', 'A15', 'A13']

    cases = ['A{}'.format(case) for case in range(1, 21)]
    cases = ['A{}'.format(case) for case in range(8, 21)]
    # cases = hard
    c_errors = {}
    for case in cases:
        c_errors[case] = False
        try:
            exec.solve(pm.OPTIONS, case)
        except:
            c_errors[case] = True
            log.error(sys.exc_info()[0])

    print(c_errors)