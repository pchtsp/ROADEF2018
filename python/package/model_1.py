import package.tuplist as tl
import package.superdict as sd
import pulp as pl
import package.config as conf


def solve_model(self, options):
    """
    This solves the MIP problem using Furini2016
    """

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
            # sumo la produccion de placas y debe ser mayor a la demanda de placas
            model += pl.lpSum(cuts[tup] for tup in cutting_production_k[j]) >= \
                     pl.lpSum(cuts[(j, o, q, l)] for (o, q, l) in cutting_production_j.get(j, [])) + \
                     (j in items.values_l() or self.rotate_plate(j) in items.values_l())

    config = conf.Config(options)
    result = config.solve_model(model)

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    cuts_ = self.vars_to_tups(cuts, binary=False)

    cut_by_level = cuts_.index_by_part_of_tuple(position=3, get_list=False)

    return cut_by_level