import package.data_input as di
import package.model as md
import package.params as pm
import os

if __name__ == "__main__":

    cases = ['A{}'.format(case) for case in range(1, 20)]
    directory = 'multi1'

    for case in cases:
        # case = pm.OPTIONS['case_name']
        options = pm.OPTIONS
        options['case_name'] = case
        output_path = \
            os.path.join(
                pm.PATHS['results'],
                directory,
                case
                ) + '/'
        options['path'] = output_path
        if not os.path.exists(output_path):
            os.mkdir(output_path)

        self = md.Model.from_input_files(case)
        new_width = options.get('max_width', None)
        if new_width is not None:
            self.input_data['parameters']['widthPlates'] = new_width
        prefix = '{}_'.format(case)



        self.export_input_data(path=output_path, prefix=prefix)
        di.export_data(output_path, options, name="options", file_type='json')

        # solving part:
        try:
            solution = self.solve(options)
        except:
            solution = None

        if solution is not None:
            self.export_cuts(solution, path=output_path)
            self.load_solution(solution)
            self.export_solution(path=output_path, prefix=prefix)
            # checks = self.check_all()
            # checks_ = sd.SuperDict.from_dict(checks).to_dictdict()
            # di.export_data(output_path, checks, name="checks", file_type='json')