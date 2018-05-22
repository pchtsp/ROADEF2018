import package.tuplist as tl
import package.superdict as sd
import pulp as pl
import package.config as conf
import package.params as pm
import numpy as np
import pprint as pp


def solve_model(self, options):

    plates = self.trees
    bins = range(len(self.trees))
    plates_bins = [(p, b) for p in plates for b in bins]

    model = pl.LpProblem("PHASE_2", pl.LpMinimize)

    print('Creating variables')
    # variables:
    # each plate has a position in the bin and a bin assignment.
    # TODO: also a rotation.
    plate_bin = pl.LpVariable.dicts(name='plate_bin', indexs=plates_bins,
                                    lowBound=0, upBound=1, cat=pl.LpInteger)
    plate_pos = pl.LpVariable.dicts(name='plate_pos', indexs=plates,
                                    lowBound=0, upBound=3200, cat=pl.LpContinuous)

    # items have horizontal and vertical movement
    item_pos = pl.LpVariable.dicts(name='plate_pos', indexs=plates,
                                    lowBound=0, upBound=3200, cat=pl.LpContinuous)


    return