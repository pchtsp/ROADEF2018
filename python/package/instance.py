import package.params as pm
import package.data_input as di
import package.tuplist as tl
import package.superdict as sd
import pandas as pd
import package.auxiliar as aux
import os
import package.cluster as cl
import math



class Instance(object):

    def __init__(self, model_data):
        self.input_data = model_data
        self.previous_items = self.calc_previous_items()
        self.next_items = self.previous_items.list_reverse()

    def get_batch(self):
        data = self.input_data['batch']
        dict_out = {v['ITEM_ID']: v for stack, items in data.items() for v in items}
        return sd.SuperDict(dict_out)

    def get_items_per_stack(self, stack=None):
        data = self.input_data['batch']
        if stack is None:
            return data
        if stack not in data:
            IndexError('STACK={} was not found in batch'.format(stack))
        return {stack: data[stack]}

    def calc_previous_items(self):
        # for each item: we get the items that need to go out first
        item_prec = sd.SuperDict()
        for stack, items in self.get_items_per_stack().items():
            prec = []
            for i in items:
                item_prec[i['ITEM_ID']] = prec[:]
                prec.append(i['ITEM_ID'])
        return item_prec

    def get_previous_items(self, get_next_items=False):
        if get_next_items:
            return self.next_items
        return self.previous_items

    def get_items_area(self):
        items = self.get_batch()
        return sum(i['LENGTH_ITEM']*i['WIDTH_ITEM'] for i in items.values())

    def get_defects_plate(self, plate):
        data = self.input_data['defects']
        if plate not in data:
            return []
        return data[plate]

    def get_defects_per_plate(self, plates=None):
        # optional filter to tell range of plates.
        # this stops the procedure when we have passed the max one.
        data = self.input_data['defects']
        if plates is None:
            return data
        return {k: data[k] for k in plates if k in data}

    def get_param(self, name=None):
        params = self.input_data['global_param']
        if name is not None:
            if name not in params:
                raise ValueError("param named {} does not exist in global_param".format(name))
            return params[name]
        return params

    def set_param(self, name, value):
        params = self.input_data['global_param']
        if name not in params:
            raise ValueError("param named {} does not exist in global_param".format(name))
        params[name] = value
        return True

    def get_plate0(self, get_dict=False):
        w, h = self.get_param('widthPlates'), self.get_param('heightPlates')
        if not get_dict:
            return w, h
        return {'width': w, 'height': h}

    def flatten_stacks(self, in_list=False):
        """
        :param in_list: return an size-ordered list of plates instead of dictionary?
        :return: dictionary indexed by piece and with a tuple
        of two dimentions. The first one is always smaller.
        """
        pieces = {k: (v['WIDTH_ITEM'], v['LENGTH_ITEM'])
                 for k, v in self.get_batch().items()}
        for k, v in pieces.items():
            if v[0] > v[1]:
                pieces[k] = v[1], v[0]
        if in_list:
            pieces = sorted(pieces.values())
            return tl.TupList(pieces)
        return sd.SuperDict.from_dict(pieces)

    def get_demand_from_items(self, tol=0):
        items = self.flatten_stacks(in_list=True)  # Ĵ in J
        items_dict = {(i, i2): math.sqrt(sum((i[r] - i2[r]) ** 2 for r in range(2)))
                      for i in items for i2 in items}

        clusters = cl.cluster_graph(items_dict, tol=tol)

        # tolerance: we try to join items into their bigger siblings.
        # the items list is sorted from smallest to biggest
        # if tolerance is 0 it will only cluster the ones that have dist= 0
        demand = {v: 0 for v in clusters}
        for v, nodes in clusters.items():
            for n in nodes:
                demand[v] += 1
        return demand

    def flatten_stacks_plus_rotated(self, clustered=False, tol=0):
        if clustered:
            original_items = [*self.get_demand_from_items(tol).keys()]
        else:
            original_items = self.flatten_stacks(in_list=True)
        items_rotated = [self.rotate_plate(v) for v in original_items]
        return [item for item in list(set(original_items + items_rotated))]

    @staticmethod
    def rotate_plate(plate):
        return plate[1], plate[0]

    @classmethod
    def from_input_files(cls, case_name, path=pm.PATHS['data']):
        return cls(di.get_model_data(case_name, path))

    def export_input_data(self, path=pm.PATHS['results'] + aux.get_timestamp(), prefix=''):
        # TODO: order columns!!
        if not os.path.exists(path):
            os.mkdir(path)
        for val in ['defects', 'batch']:
            table = pd.DataFrame.from_dict(self.input_data[val], orient='index')
            table.to_csv(path + '{}{}.csv'.format(prefix, val), index=False, sep=';')

        val = 'global_param'
        table = pd.DataFrame.from_dict(self.input_data[val], orient='index').\
                reset_index().rename(columns={'index': 'NAME', 0: 'VALUE'})
        table.to_csv(path + '{}.csv'.format(val), index=False, sep=';')

        return True