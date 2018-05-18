import package.data_input as di
import package.model as md
import package.params as pm
import scripts.exec as exec
import os

if __name__ == "__main__":

    cases = ['A{}'.format(case) for case in range(1, 21)]

    for case in cases:
        options = pm.OPTIONS
        options['case_name'] = case
        options['path'] += case

        if not os.path.exists(options['path']):
            os.mkdir(options['path'])
        exec.solve_heuristic(options)
        # exec.solve_case(options)
