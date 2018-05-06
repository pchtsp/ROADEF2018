import package.superdict as sd
import package.data_input as di
import package.model as md
import package.params as pm
import os
import sys


def solve_case(options):

    case = options['case_name']
    self = md.Model.from_input_files(case)
    new_width = options.get('max_width', None)

    # if specified, change max width:
    if new_width is not None:
        self.input_data['global_param']['widthPlates'] = new_width
    prefix = case + '_'

    if not os.path.exists(options['path']):
        os.mkdir(options['path'])

    output_path = options['path']

    self.export_input_data(path=output_path, prefix=prefix)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    try:
        solution = self.solve(options)
    except:
        solution = None

    if solution is None:
        return None

    # exporting part:
    self.export_cuts(solution, path=output_path)
    try:
        self.load_solution(solution)
    except:
        print('There was a problem loading the solution.')

    self.export_solution(path=output_path, prefix=prefix)


def solve_case_iter(options):
    case = options['case_name']
    self = md.Model.from_input_files(case)
    prefix = case + '_'

    if not os.path.exists(options['path']):
        os.mkdir(options['path'])

    output_path = options['path']

    self.export_input_data(path=output_path, prefix=prefix)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    # try:
    self.solve_iterative(options,
                         export=True,
                         max_items=options.get('max_items', 20),
                         sort=True)
    # except:
    #     print('There was an error with the solving!')
    #     print(sys.exc_info()[0])
    #     return None

    return True


if __name__ == "__main__":
    solve_case_iter(pm.OPTIONS)
    # solve_case(options=pm.OPTIONS)
        # checks = self.check_all()
        # checks_ = sd.SuperDict.from_dict(checks).to_dictdict()
        # di.export_data(output_path, checks, name="checks", file_type='json')