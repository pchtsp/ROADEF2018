

class Instance(object):

    def __init__(self, model_data):
        self.data = model_data

    def get_batch(self, stack=None):
        batch_data = self.data['batch']
        if stack is None:
            return batch_data
        return {k: v for k, v in batch_data.items()
                if v['STACK'] == stack}

    def get_defects(self, plate=None):
        if plate is None:
            return self.data['defects']
        if plate not in self.data['defects']:
            raise IndexError('PLATE_ID={} was not found in defects'.format(plate))
        return self.data['defects'][plate]

