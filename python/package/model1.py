import package.solution as sol
import pulp as pl
import package.tuplist as tl
import package.superdict as sd
import package.params as pm
import numpy as np
import package.config as conf
import pprint as pp


class Model(sol.Solution):
    """
    This solves the MIP problem using Furini2016
    """

    def flatten_stacks(self):
        # self
        batch = self.get_items_per_stack()
        all_batches = {}
        carry = 0
        for stack, contents in batch.items():
            for key, desc in contents.items():
                all_batches[carry + key] = desc
            carry += len(contents)
        items = {k: (v['WIDTH_ITEM'], v['LENGTH_ITEM'])
                 for k, v in all_batches.items()}
        return sd.SuperDict.from_dict(items)

    def plate_generation(self):
        """
        calculates all possible cuts to plates and the resulting plates
        In the paper, this is called "Procedure 1"
        :return:
        """
        # items = self.flatten_stacks()
        # cutting_options_tup = tl.TupList()
        # cutting_options = sd.SuperDict()
        cutting_production = tl.TupList()  # (j, o, q, k)
        plate0 = self.get_plate0(get_dict=False)
        cut_level = 0
        plates = set()
        plates.add((plate0, cut_level))
        non_processed = [(plate0, cut_level, list(self.flatten_stacks().values()))]
        max_iter = 0
        while len(non_processed) > 0 and max_iter < 100000:
            max_iter += 1
            print('Iteration={}; Nonprocessed= {}; plates={}'.format(max_iter, len(non_processed), len(plates)))
            j, cut_level, original_items = non_processed.pop()
            for o in pm.ORIENTATIONS:
                cutting_options, new_original_items = self.get_cut_positions(j, o, original_items)
                for q in cutting_options[1:]:
                    # cut j
                    j1, j2 = self.cut_plate(j, o, q)
                    print('from plate {}: {} and {} at cut_level= {}'.format(j, j1, j2, cut_level + 1))
                    for k in [j1, j2]:
                        if not self.check_plate_can_fit_some_item(k):
                            # if the piece is not useful,
                            # we do not consider it a plate
                            continue
                        # here we register the tuple
                        # of the production of plates
                        cutting_production.add(j, o, q, cut_level + 1, k)
                        if cut_level >= 2:
                            continue
                        if (k, cut_level + 1) in plates:
                            # if we already registered it
                            # we will not re-add it
                            continue
                        plates.add((k, cut_level + 1))
                        non_processed.append((k, cut_level + 1, new_original_items))
        return cutting_production

    def get_cut_positions(self, plate, orientation, original_items=None):
        """
        :param plate: (width, height)
        :param orientation:
        :param original_items:
        :return:
        """
        dim = 1
        dim2 = 0
        if orientation == pm.ORIENT_V:
            dim, dim2 = dim2, dim
        max_size = plate[dim]
        max_size2 = plate[dim2]

        if original_items is None:
            original_items = list(self.flatten_stacks().values())
        items_rotated = [self.rotate_plate(v) for v in original_items]

        total_items = list(set(original_items + items_rotated))
        total_items_dim = [item[dim] for item in total_items if item[dim2] <= max_size2]
        total_items_dim = np.unique(total_items_dim)
        cuts = [0]
        for item in total_items_dim:
            new_cuts = []
            for cut in cuts:
                size = item + cut
                if size < max_size and size not in cuts and \
                    (size <= max_size//2 or (max_size-size) not in cuts):
                    new_cuts.append(size)
            cuts.extend(np.unique(new_cuts))
        return cuts, total_items

    @staticmethod
    def cut_plate(plate, orientation, cut):
        """
        :param plate: (width, height), i.e. (500, 100)
        :param orientation:
        :param cut:
        :return:
        """
        # cut plate at position cut with orientation
        width, height = plate
        if orientation == pm.ORIENT_V:
            if cut > width:
                raise IndexError('plate with width {} is smaller than cut {}',
                                 width, cut)
            part1 = cut, height
            part2 = width - cut, height
        else:
            if cut > height:
                raise IndexError('plate with height {} is smaller than cut {}',
                                 width, cut)
            part1 = cut, width
            part2 = height - cut, width
        return part1, part2

    @staticmethod
    def rotate_plate(plate):
        return plate[1], plate[0]

    @staticmethod
    def vars_to_tups(var, binary=True):
        # because of rounding approximations; we need to check if its bigger than half:
        if binary:
            return tl.TupList([tup for tup in var if var[tup].value() > 0.5])
        return sd.SuperDict({tup: var[tup].value() for tup in var if var[tup].value() > 0.5})

    def plate_inside_plate(self, plate1, plate2, turn=True):
        origin = {'X': 0, 'Y': 0}
        result = self.square_inside_square(
            [origin, {'X': plate1[0], 'Y': plate1[1]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=False
        )
        if not result and turn:
            return self.square_inside_square(
            [origin, {'X': plate1[1], 'Y': plate1[0]}],
            [origin, {'X': plate2[0], 'Y': plate2[1]}],
            both_sides=False
        )

    def check_plate_can_fit_some_item(self, plate):
        items = self.flatten_stacks()
        for key, value in items.items():
            result = self.plate_inside_plate(value, plate)
            if result == 1:
                return True
            # if it doesn't fit
            if not result:
                if self.plate_inside_plate(self.rotate_plate(value), plate) == 1:
                    return True
        return False

    def solve(self):
        """

        :return:
        """
        # parameters:
        cutting_production = self.plate_generation()  # a
        # (j, o, q, level, k)
        # cut "q" with orientation "o" on plate "j" produces plate "k"

        plates = \
            cutting_production.filter(0).unique2() + \
            cutting_production.filter(4).unique2()
        plates = tl.TupList(plates).unique2()
        # plates_dict = {k: p for k, p in enumerate(plates)}
        # pd_inv = {p: k for k, p in plates_dict.items()}
        # cutting_production_c = [(pd_inv[tup[0]], tup[1], tup[2], tup[3], pd_inv[tup[4]])
        #                         for tup in cutting_production]
        # cutting_production_c = tl.TupList(cutting_production_c)
        # first_plate = pd_inv[self.get_plate0(get_dict=False)]
        first_plate = self.get_plate0(get_dict=False)
        items = self.flatten_stacks()  # Ĵ in J
        # items[-1] = first_plate
        # plates_items = tl.TupList([(p, i) for p in plates for i in items.values_l()
        #                 if self.plate_inside_plate(i, p) == 1])

        # [i for i in items.values_l() if i in plates]
        # np.intersect1d(items.values_l(), plates)

        cutting_options_tup = cutting_production.filter([0, 1, 2, 3]).unique2()
        # (j, o, q, level)
        # tuple of cut posibilities based on cutting_options

        # cutting_options = cutting_options_tup.to_dict(result_col=2)  # Q
        # (j, o): [q]
        # set of cut positions we can cut plate j with orientation o.
        cutting_options_tup_0 = \
            tl.TupList([tup for tup in cutting_options_tup if tup[0] == first_plate])
        # j: (o, q, level)
        cutting_production_j = \
            cutting_options_tup.to_dict(result_col=[1, 2, 3])
        # k: (j, o, q, level)
        cutting_production_k = \
            cutting_production.to_dict(result_col=[0, 1, 2, 3])

        # {j: num}
        # for each j in Ĵ, it names the demand
        max_plates = 10

        # model
        model = pl.LpProblem("ROADEF", pl.LpMinimize)

        # variables:
        cuts = pl.LpVariable.dicts(name='cuts', indexs=cutting_options_tup,
                                  lowBound=0, upBound=max_plates, cat='Integer')
        # cut_items = pl.LpVariable.dicts(name='items', indexs=items.values_l(),
        #                                 lowBound=0, upBound=1, cat='Integer')

        # objective function: (11)
        model += pl.lpSum(cuts[tup] for tup in cutting_options_tup_0)

        # constraints (2) + (3)
        for j in plates:
            if j != first_plate:
                model += pl.lpSum(cuts[tup] for tup in cutting_production_k[j]) >= \
                         pl.lpSum(cuts[(j, o, q, l)] for (o, q, l) in cutting_production_j.get(j, [])) + (j in items.values_l())

        default_options = {
            'timeLimit': 300
            , 'gap': 0
            , 'solver': "CPLEX"
            , 'path': '/home/pchtsp/Documents/projects/ROADEF2018/python/log/'
        }

        config = conf.Config(default_options)
        result = config.solve_model(model)

        cuts_ = self.vars_to_tups(cuts, binary=False)
        level = 0

        cuts_.index_by_part_of_tuple(position=3, get_list=True)

        cut = ((738, 1672), 0, 1550, 3)
        plate, o, q, l = cut
        ref_pos = 0, 0
        # if o == pm.ORIENT_V:
        #


        # constraints (12)
        # for j in items:
        #     model += cut_items[j] >=


if __name__ == "__main__":
    self = Model.from_input_files(case_name='A1')
    # plate0 = self.get_plate0(get_dict=False)
    # self.flatten_stacks().values()
    # result = self.get_cut_positions(plate0, 'h')
    # result2 = self.get_cut_positions(plate0, 'v')
    production = self.plate_generation()
    # len(result2)
    # pp.pprint(result2)
    # len(result)
    # np.unique(result).__len__()