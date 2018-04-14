import package.superdict as sd
import package.data_input as di
import package.model as md
import package.params as pm

if __name__ == "__main__":

    options = pm.OPTIONS
    case = options['case_name']
    self = md.Model.from_input_files(case)
    new_width = options.get('max_width', None)
    if new_width is not None:
        self.input_data['parameters']['widthPlates'] = new_width
    prefix = case + '_'

    output_path = options['path']

    self.export_input_data(path=output_path, prefix=prefix)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    # cutting_production = self.plate_generation()
    # solution = None
    solution = self.solve(options)

    if solution is not None:
        self.export_cuts(solution, path=output_path)
        self.load_solution(solution)
        self.export_solution(path=output_path, prefix=prefix)
        # checks = self.check_all()
        # checks_ = sd.SuperDict.from_dict(checks).to_dictdict()
        # di.export_data(output_path, checks, name="checks", file_type='json')