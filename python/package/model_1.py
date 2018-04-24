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
    cutting_production = \
        self.plate_generation(max_iterations=options.get('max_iters', None))  # a
    # (w1, h2, l1, o, q, w2, h2, l2)
    # cut "q" with orientation "o" on plate "j",
    # in level l1 produces plate "k" in level l2
    print('Getting np array')
    cutting_production = \
        np.array(cutting_production,
                 dtype=[('w1', 'i4'),
                        ('h1', 'i4'),
                        ('l1', 'i4'),
                        ('o', 'i4'),
                        ('q', 'i4'),
                        ('w2', 'i4'),
                        ('h2', 'i4'),
                        ('l2', 'i4')]
                 )
    # shortcuts
    j = ['w1', 'h1']
    k = ['w2', 'h2']
    # plates = \
    #     np.concatenate([
    #         np.unique(cutting_production['j'], axis=0),
    #         np.unique(cutting_production['k'], axis=0)
    #     ])
    # plates = np.unique(plates, axis=0)
    first_plate = np.array(self.get_plate0(get_dict=False),
                           dtype=[('w1', 'i4'), ('h1', 'i4')])
    items = self.flatten_stacks(in_list=True)  # Ĵ in J
    demand = {v: 0 for v in items}
    for v in items:
        demand[v] += 1

    print('Getting cutting options')
    # (j, l1, o, q)
    cutting_options_tup = np.unique(cutting_production[j + ['l1', 'o', 'q']])
    # (plate0, o, q, level)
    cutting_options_tup_0 = \
        cutting_options_tup[cutting_options_tup[j] == first_plate]

    plates_level = np.unique(cutting_production[k + ['l2']])
    plates_level1 = plates_level[plates_level['l2'] == 1]

    l_j = {}
    for w, h, l in plates_level:
        key = tuple([w, h])
        if key not in l_j:
            l_j[key] = []
        l_j[key].append(l)

    print('Getting dictionary of cutting options')
    # (j, l1): (o, q)
    cutting_production_j_level = {}
    for (w, h, l1, o, q) in cutting_options_tup:
        key = w, h, l1
        value = (o, q)
        if key not in cutting_production_j_level:
            cutting_production_j_level[key] = []
        cutting_production_j_level[key].append(value)

    print('Getting dictionary of cutting production')
    # (k, l2): (j, l1, o, q)
    cutting_production_k_level = {}
    for (w1, h1, l1, o, q, w2, h2, l2) in cutting_production:
        key = w2, h2, l2
        value = (w1, h1, l1, o, q)
        if key not in cutting_production_k_level:
            cutting_production_k_level[key] = []
        cutting_production_k_level[key].append(value)

    # {j: num}
    # for each j in Ĵ, it names the demand
    max_plates = options.get("max_plates", 10)

    # model
    model = pl.LpProblem("ROADEF", pl.LpMinimize)

    print('Creating variables')
    # variables:
    # this decides the kind of cut that is done on a plate
    cuts = pl.LpVariable.dicts(name='cuts', indexs=cutting_options_tup.tolist(),
                              lowBound=0, upBound=max_plates, cat=pl.LpInteger)
    # This models if a plate is used to satisfy demand instead than for plate-balancing
    cut_for_demand = pl.LpVariable.dicts(
        name='cut_for_demand', indexs=plates_level.tolist(),
        lowBound=0, upBound=max_plates, cat=pl.LpInteger)
    # This models if a plate is used as leftover in the first level
    cut_for_leftover = pl.LpVariable.dicts(
        name='cut_for_leftover', indexs=plates_level1.tolist(),
        lowBound=0, upBound=max_plates, cat=pl.LpInteger)

    # objective function: (11)
    model += pl.lpSum(cuts[tup] * (tup[0]+1) for tup in cutting_options_tup_0.tolist()) - \
             pl.lpSum(cut_for_leftover[tup] * tup[0] for tup in plates_level1.tolist())

    print('Creating constraints')
    # constraints (2) + (3)
    for w1, h1, l in plates_level:
        jl = w1, h1, l
        # if I sum the plate *net* production, it needs to be greater than the demand
        # the production needs to have occurred in the level *before*
        model += pl.lpSum(cuts[tup] for tup in cutting_production_k_level[jl]) >=\
                 pl.lpSum(
                     cuts[w1, h1, l, o, q]
                     for (o, q) in cutting_production_j_level.get(jl, [])
                 ) + cut_for_demand.get(jl, 0) + cut_for_leftover.get(jl, 0)

    # for each item: we need to cut at least the correct number of plates:
    # at any level
    for j, q in demand.items():
        w, h = j
        if j not in l_j:
            print("plate {} cannot be produced!".format(j))
            continue
        model += pl.lpSum(
            cut_for_demand.get((w, h, l), 0) + cut_for_demand.get((h, w, l), 0)
            for l in l_j[j]
        ) >= q

    print('Solving')
    # model.writeLP('test.lp')
    config = conf.Config(options)
    solver = config.get_solver()
    result = model.solve(solver)

    print('Finished solving')
    if result != 1:
        print("Model resulted in non-feasible status")
        return None

    cuts_ = sd.SuperDict({
        ((k[0], k[1]), k[3], k[4], k[2] + (k[3] == pm.cut_level_next_o[k[2]])): v
        for k, v in self.vars_to_tups(cuts, binary=False).items()
    })
    # cut_for_demand_ = self.vars_to_tups(cut_for_demand, binary=False)

    return cuts_