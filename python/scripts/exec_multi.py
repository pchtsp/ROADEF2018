import package.data_input as di
import package.model as md
import package.params as pm
import scripts.exec as exec
import os

if __name__ == "__main__":

    # cases = ['A{}'.format(case) for case in range(1, 21)]
    cases = ['A{}'.format(case) for case in range(4, 21)]

    for case in cases:
        exec.solve(pm.OPTIONS, case)
