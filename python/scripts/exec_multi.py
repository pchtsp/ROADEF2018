import package.data_input as di
import package.model as md
import package.params as pm
import scripts.exec as exec
import os

if __name__ == "__main__":

    cases = ['A{}'.format(case) for case in range(15, 21)]
    directory = 'multi2'
    base_path = os.path.join(pm.PATHS['results'], directory)

    if not os.path.exists(base_path):
        os.mkdir(base_path)

    for case in cases:
        options = pm.OPTIONS
        options['case_name'] = case
        options['path'] = os.path.join(base_path, case) + '/'

        if not os.path.exists(options['path']):
            os.mkdir(options['path'])

        exec.solve_case(options)
