

class Instance(object):

    def __init__(self, model_data):
        self.input_data = model_data

    def get_batch_per_stack(self, stack=None):
        batch_data = {v['STACK']: {} for v in self.input_data['batch'].values()}
        for k, v in self.input_data['batch'].items():
            batch_data[v['STACK']][k] = v
        if stack is None:
            return batch_data
        if stack not in batch_data:
            IndexError('STACK={} was not found in batch'.format(stack))
        return batch_data[stack]

    def get_defects_per_plate(self, plate=None):
        defects = self.input_data['defects']
        defects_plate = {int(v['PLATE_ID']): {} for v in defects.values()}
        for k, v in defects.items():
            defects_plate[v['PLATE_ID']][k] = v
        if plate is None:
            return defects_plate
        if plate not in defects_plate:
            raise IndexError('PLATE_ID={} was not found in defects'.format(plate))
        return defects_plate[plate]
