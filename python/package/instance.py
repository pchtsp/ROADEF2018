import package.params as pm
import package.data_input as di
import package.tuplist as tl
import package.superdict as sd


class Instance(object):

    def __init__(self, model_data):
        self.input_data = model_data

    def get_items_per_stack(self, stack=None):
        # TODO: use SuperDict.index_by_property instead
        batch_data = {v['STACK']: {} for v in self.input_data['batch'].values()}
        for k, v in self.input_data['batch'].items():
            batch_data[v['STACK']][k] = v
        batch_data = sd.SuperDict.from_dict(batch_data)
        if stack is None:
            return batch_data
        if stack not in batch_data:
            IndexError('STACK={} was not found in batch'.format(stack))
        return batch_data[stack]

    def get_defects_per_plate(self, plate=None):
        # TODO: use SuperDict.index_by_property instead
        defects = self.input_data['defects']
        defects_plate = {int(v['PLATE_ID']): {} for v in defects.values()}
        for k, v in defects.items():
            defects_plate[v['PLATE_ID']][k] = v
        if plate is None:
            return defects_plate
        if plate not in defects_plate:
            raise IndexError('PLATE_ID={} was not found in defects'.format(plate))
        return defects_plate[plate]

    def get_param(self, name=None):
        params = self.input_data['parameters']
        if name is not None:
            if name not in params:
                raise ValueError("param named {} does not exist in parameters".format(name))
            return params[name]
        return params

    def get_plate0(self, get_dict=True):
        if not get_dict:
            return self.get_param('widthPlates'), self.get_param('heightPlates')
        return {'width': self.get_param('widthPlates'),
                'height': self.get_param('heightPlates')}

    @classmethod
    def from_input_files(cls, case_name, path=pm.PATHS['data']):
        return cls(di.get_model_data(case_name, path))
