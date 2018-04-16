import package.tuplist as tl
import package.superdict as sd
import pulp as pl
import package.config as conf
import package.params as pm
import numpy as np
import pprint as pp


def solve_model(self, options):
    """
    This solves the MIP problem using Furini2016

    :return:
    """
    # parameters:
    cutting_production = self.plate_generation(max_iterations=options.get('max_iters', None))  # a
    # (j, l1, o, q, k, l2)
    # cut "q" with orientation "o" on plate "j", in level l1 produces plate "k" in level l2
    cutting_production = \
        np.array(cutting_production,
                 dtype=[('j', '2i4'),
                        ('l1', 'i4'),
                        ('o', 'i4'),
                        ('q', 'i4'),
                        ('k', '2i4'),
                        ('l2', 'i4')]
                 )

    # plates = \
    #     np.concatenate([
    #         np.unique(cutting_production['j'], axis=0),
    #         np.unique(cutting_production['k'], axis=0)
    #     ])
    # plates = np.unique(plates, axis=0)
    first_plate = np.array(self.get_plate0(get_dict=False), dtype='i')
    items = self.flatten_stacks(in_list=True)  # Ĵ in J
    demand = {v: 0 for v in items}
    for v in items:
        demand[v] += 1

    # (j, l1, o, q)
    cutting_options_tup = np.unique(cutting_production[['j', 'l1', 'o', 'q']])
    # (plate0, o, q, level)
    a = cutting_options_tup['j'] == first_plate
    cutting_options_tup_0 = cutting_options_tup[a.all(axis=1)]

    # a = cutting_options_tup['j'][:, 1]

    plates_level_demand = \
            np.unique(cutting_production[['k', 'l2']])

    plates_level = \
        np.concatenate([
            plates_level_demand.tolist(),
            np.unique(cutting_options_tup[['j', 'l1']]).tolist()
        ])
    plates_level = np.unique(plates_level)

    l_j = {}
    for j, l in plates_level:
        key = tuple(j)
        if key not in l_j:
            l_j[key] = []
        l_j[key].append(l)

    # (j, l1): (o, q)
    cutting_production_j_level = {}
    for ((w, h), l1, o, q) in cutting_options_tup:
        key = w, h, l1
        value = (o, q)
        if key not in cutting_production_j_level:
            cutting_production_j_level[key] = []
            cutting_production_j_level[key].append(value)

    # (k, l2): (j, l1, o, q)
    cutting_production_k_level = {}
    for (j, l1, o, q, k, l2) in cutting_production:
        key = k[0], k[1], l2
        value = (j[0], j[1], l1, o, q)
        if key not in cutting_production_k_level:
            cutting_production_k_level[key] = []
        cutting_production_k_level[key].append(value)

    # {j: num}
    # for each j in Ĵ, it names the demand
    max_plates = options.get("max_plates", 10)

    # model
    model = pl.LpProblem("ROADEF", pl.LpMinimize)

    # variables:
    # this decides the kind of cut that is done on a plate
    cuts = {}
    for (w, h), l, o, q in cutting_options_tup:
            cuts[w, h, l, o, q] = \
                pl.LpVariable(name='cuts_{}_{}_{}_{}_{}'.format(w, h, l, o, q),
                              lowBound=0, upBound=max_plates, cat='Integer')
    # This models if a plate is use to satisfy demand instead than for plate-balancing
    cut_for_demand = {}
    for (w, h), l in plates_level_demand:
        cut_for_demand[w, h, l] = \
            pl.LpVariable(name='cut_for_demand_{}_{}_{}'.format(w, h, l),
                          lowBound=0, upBound=max_plates, cat='Integer')

    # objective function: (11)
    model += pl.lpSum(cuts[w, h, l, o, q] for (w, h), l, o, q in cutting_options_tup_0)

    # constraints (2) + (3)
    for j, l in plates_level:
        np.all(j == first_plate)
        w1, h1 = j
        jl = w1, h1, l
        # if I sum the plate *net* production, it needs to be greater than the demand
        # the production needs to have occured in the level *before*
        model += pl.lpSum(cuts[tup] for tup in cutting_production_k_level[jl]) >=\
                 pl.lpSum(
                     cuts[w1, h1, l, o, q] for (o, q) in cutting_production_j_level.get(jl, [])
                 ) + cut_for_demand.get(jl, 0)

    # for each item: we need to cut at least the correct number of plates:
    # at any level
    for j, q in demand.items():
        w, h = j
        model += pl.lpSum(
            cut_for_demand.get((w, h, l), 0) + cut_for_demand.get((h, w, l), 0)
            for l in l_j[j]
        ) >= q

    # model.writeLP('test.lp')
    config = conf.Config(options)
    solver = config.get_solver()
    result = model.solve(solver)

    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    cuts_ = sd.SuperDict({
        ((k[0], k[1]), k[3], k[4], k[2] + (k[3] == pm.cut_level_next_o[k[2]])): v
        for k, v in self.vars_to_tups(cuts, binary=False).items()
    })
    cut_for_demand_ = self.vars_to_tups(cut_for_demand, binary=False)

    return cuts_