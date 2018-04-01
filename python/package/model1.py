import package.solution as sol
import pulp as pl
import package.tuplist as tl
import package.superdict as sd
import package.params as pm
import numpy as np
import pprint as pp


class Model(sol.Solution):

    def flatten_stacks(self):
        # self
        batch = self.get_items_per_stack()
        all_batches = {}
        carry = 0
        for stack, contents in batch.items():
            for key, desc in contents.items():
                all_batches[carry + key] = desc
            carry += len(contents)
        items = {k: {
            'width': v['WIDTH_ITEM']
            ,'height': v['LENGTH_ITEM']
        } for k, v in all_batches.items()}
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
        plate0 = self.get_plate0()
        plates = [plate0]
        non_processed = [plate0]
        while len(non_processed) > 0:
            j = non_processed.pop()
            for o in pm.ORIENTATIONS:
                cutting_options = self.get_cut_positions(j, o)
                for q in cutting_options:
                    # cut j
                    j1, j2 = self.cut_plate(j, o, q)
                    for k in [j1, j2]:
                        if not self.check_plate_can_fit_some_item(k):
                            # if the piece is not useful,
                            # we do not consider it a plate
                            continue
                        # here we register the tuple
                        # of the production of plates
                        cutting_production.add(j, o, q, k)
                        if k in plates:
                            # if we already registered it
                            # we will not re-add it
                            continue
                        plates.append(k)
                        non_processed.append(k)
        return cutting_production

    def get_cut_positions(self, plate, orientation):
        dim = 'height'
        dim2 = 'width'
        if orientation == pm.ORIENT_V:
            dim, dim2 = dim2, dim
        max_size = plate[dim]
        max_size2 = plate[dim2]

        original_items = list(self.flatten_stacks().values())
        items_rotated = [self.rotate_plate(v) for v in original_items]

        total_items = [item[dim] for item in original_items + items_rotated
                       if item[dim2] <= max_size2]
        total_items = np.unique(total_items)
        cuts = [0]
        for item in total_items:
            new_cuts = []
            for cut in cuts:
                size = item + cut
                if size < max_size and size not in cuts and \
                    (size <= max_size//2 or (max_size-size) not in cuts):
                    new_cuts.append(size)
            cuts.extend(np.unique(new_cuts))
        return cuts

    @staticmethod
    def cut_plate(plate, orientation, cut):
        # cut plate at position cut with orientation
        width, height = plate['width'], plate['height']
        if orientation == pm.ORIENT_V:
            if cut > width:
                raise IndexError('plate with width {} is smaller than cut {}',
                                 width, cut)
            part1 = {'width': cut, 'height': height}
            part2 = {'width': width - cut, 'height': height}
        else:
            if cut > height:
                raise IndexError('plate with height {} is smaller than cut {}',
                                 width, cut)
            part1 = {'height': cut, 'width': width}
            part2 = {'height': height - cut, 'width': width}
        return part1, part2

    @staticmethod
    def rotate_plate(plate):
        return {'height': plate['width'], 'width': plate['height']}

    def plate_inside_plate(self, plate1, plate2):
        origin = {'X': 0, 'Y': 0}
        return self.square_inside_square(
            [origin, {'X': plate1['width'], 'Y': plate1['height']}],
            [origin, {'X': plate2['width'], 'Y': plate2['height']}],
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
        This solves the MIP problem using Furini2016
        :return:
        """
        # parameters:
        cutting_production = self.plate_generation()  # a
        # (j, o, q, k)
        # cut "q" with orientation "o" on plate "j" produces plate "k"

        plates = cutting_production.filter(0).unique()  # J
        first_plate = 0
        items = self.flatten_stacks()  # Ĵ in J

        cutting_options_tup = cutting_production.filter(range(3)).unique()
        # (j, o, q)
        # tuple of cut posibilities based on cutting_options

        # cutting_options = cutting_options_tup.to_dict(result_col=2)  # Q
        # (j, o): [q]
        # set of cut positions we can cut plate j with orientation o.
        cutting_options_tup_0 = \
            tl.TupList([tup for tup in cutting_options_tup if tup[0] == first_plate])
        cutting_production_j = \
            cutting_production.to_dict(result_col=[1, 2, 3], is_list=False)
        cutting_production_k = \
            cutting_production.to_dict(result_col=[0, 1, 2], is_list=False)

        # {j: num}
        # for each j in Ĵ, it names the demand
        max_plates = 100

        # model
        model = pl.LpProblem("ROADEF", pl.LpMinimize)

        # variables:
        cuts = pl.LpVariableDict(name='cuts', data=cutting_options_tup,
                                 lowBound=0, upBound=max_plates, cat='Integer')
        # cut_items = pl.LpVariableDict(name='items', data=items,
        #                          lowBound=0, upBound=demand, cat='Integer')

        # objective function: (11)
        model += pl.lpSum(cuts[tup] for tup in cutting_options_tup_0)

        # constraints (2) + (3)
        for j in plates:
            model += pl.lpSum(cuts[tup] for tup in cutting_production_j[j]) >= \
                     pl.lpSum(cuts[tup] for tup in cutting_production_k[j]) + (j in items.values())

        # # constraints (12)
        # for j in items:
        #     model += cut_items[j] >= demand[j]


if __name__ == "__main__":
    self = Model.from_input_files(case_name='A1')
    plate0 = self.get_plate0()
    self.flatten_stacks().values()
    result = self.get_cut_positions(plate0, 'h')
    result2 = self.get_cut_positions(plate0, 'v')
    len(result2)
    # pp.pprint(result2)
    # len(result)
    # np.unique(result).__len__()