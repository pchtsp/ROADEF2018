import package.solution as sol
import package.model as mod
import package.params as pm
import numpy as np

def test1():

    e = '201804142339/'
    solution = sol.Solution.from_io_files(path=pm.PATHS['experiments'] + e)
    # solution = sol.Solution.from_io_files(path=pm.PATHS['checker_data'], case_name='A0')
# print(self.trees[0].get_ascii(show_internal=True, attributes=['NODE_ID', 'PARENT']))
################################################

def test2():

    model = mod.Model.from_input_files(case_name='A2')

    # r = model.get_smallest_items()
    pieces = model.flatten_stacks(in_list=True)
    min_length = pieces[1][1] + 1
    filter_out_pos = set()
    for pos, i in enumerate(pieces):
        if i[1] >= min_length:
            filter_out_pos.add(pos)
        else:
            min_length = i[1]
    # [i for i in r if i[1] <= r[0][1]]

if __name__ == "__main__":
    test1()