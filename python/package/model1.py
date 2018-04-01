import package.solution as sol
import pulp as pl


class Model(sol.Solution):

    def __init__(self, input_data):
        super().__init__(input_data, [])

    def solve(self):
        """
        This solves the MIP problem using Furini2016
        :return:
        """
        # parameters:
        orientations = ['h', 'v']  # o
        plates = []  # J
        items = []  # Ĵ in J
        cutting_options = {}  # Q
        # (j, o): [q]
        # set of cut positions we can cut plate j with orientation o.
        cutting_options_tup = []
        # (j, o, q)
        # tuple of cut posibilities based on cutting_options
        cutting_production = {}  # a
        # (j, q, o, k)
        # cut "q" with orientation "o" on plate "j" produces plate "k"
        cutting_production_j = {}
        for (j, q, o, k) in cutting_production:
            cutting_production_j[j] = (q, o, k)

        demand = {}  #
        # {j: num}
        # for each j in Ĵ, it names the demand
        max_plates = 100

        # model
        model = pl.LpProblem("ROADEF", pl.LpMinimize)

        # variables:
        cuts = pl.LpVariableDict(name='cuts', data=cutting_options_tup,
                                 lowBound=0, upBound=max_plates, cat='Integer')
        cut_items = pl.LpVariableDict(name='items', data=items,
                                 lowBound=0, upBound=demand, cat='Integer')

        # constraints:
        for j in plates:
            model += pl.lpSum(cuts[(k, o, q)]) >= \
                     pl.lpSum()


