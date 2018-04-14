import package.data_input as di
import package.model as md
import package.params as pm

if __name__ == "__main__":

    cases = ['A{}'.format(case) for case in range(2, 10)]

    for max_plates in [2, 10]:
        for case in cases:
            # case = pm.OPTIONS['case_name']
            options = pm.OPTIONS
            options['case_name'] = case
            options['max_plates'] = max_plates

            self = md.Model.from_input_files(case)
            prefix = '{}_'.format(case)

            output_path = options['path']

            self.export_input_data(path=output_path, prefix=prefix)
            di.export_data(output_path, options, name="options", file_type='json')

            # solving part:
            try:
                solution = self.solve(options)
            except:
                solution = None

            if solution is not None:
                self.load_solution(solution)
                self.export_solution(path=output_path, prefix=prefix)
                # checks = self.check_all()
                # checks_ = sd.SuperDict.from_dict(checks).to_dictdict()
                # di.export_data(output_path, checks, name="checks", file_type='json')