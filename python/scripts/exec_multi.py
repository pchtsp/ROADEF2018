import package.data_input as di
import package.model as md
import package.params as pm
import scripts.exec as exec
import os

if __name__ == "__main__":

    # somewhat problematic:
    incomplete = ['A2', 'A13', 'A15', 'A6', 'A14']

    # cases = ['A{}'.format(case) for case in range(1, 21)]
    cases = ['A{}'.format(case) for case in range(1, 21)]
    # problematic:
    hard = ['A14', 'A15', 'A13', 'A2']
    cases = ['A2']
    c_errors = {}
    for case in cases:
        c_errors[case] = False
        try:
            exec.solve(pm.OPTIONS, case)
        except:
            c_errors[case] = True
            pass

    print(c_errors)