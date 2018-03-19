
class Experiment(object):

    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution

    @classmethod
    def from_dir(cls, path, format='json', prefix="data_"):
        return
        # files = [os.path.join(path, prefix + f + "." + format) for f in ['in', 'out']]
        # if not np.all([os.path.exists(f) for f in files]):
        #     return None
        # instance = di.load_data(files[0])
        # solution = di.load_data(files[1])
        # return cls(inst.Instance(instance), sol.Solution(solution))

    def check_all(self):
        return True

    def check_overlapping(self):
        return True

    def check_sequence(self):
        return True

    def check_cuts_number(self):
        return True

    def check_cuts_guillotine(self):
        return True

    def check_demand_satisfied(self):
        return True
