import package.superdict as sd
import package.data_input as di
import package.model as md
import package.params as pm

if __name__ == "__main__":

    self = md.Model.from_input_files(pm.OPTIONS['case_name'])

    options = pm.OPTIONS
    output_path = options['path']

    self.export_input_data(path=output_path)
    di.export_data(output_path, options, name="options", file_type='json')

    # solving part:
    solution = self.solve(options)

    if solution is not None:
        self.load_solution(solution)
        self.export_solution(path=output_path)
        # checks = self.check_all()
        # checks_ = sd.SuperDict.from_dict(checks).to_dictdict()
        # di.export_data(output_path, checks, name="checks", file_type='json')
    #
    # self = Model.from_input_files(case_name='A1')
    # # plate0 = self.get_plate0(get_dict=False)
    # # self.flatten_stacks().values()
    # # result = self.get_cut_positions(plate0, 'h')
    # # result2 = self.get_cut_positions(plate0, 'v')
    # # production = self.plate_generation()
    # cut_by_level = self.solve()
    # self.load_solution(cut_by_level)
    # # items = self.flatten_stacks()
    # # tree = self.trees[0]
    # # print(self.trees[1].get_tree_root().get_ascii(show_internal=True))
    # checks = self.check_all()
    # # print(self.trees[1].get_tree_root().get_ascii(show_internal=True, attributes=['name', 'TYPE']))
    # self.graph_solution()
    # # self
    # # len(result2)
    # # pp.pprint(result2)
    # # len(result)
    # # np.unique(result).__len__()