import numpy as np


class TupList(list):

    def tup_filter(self, indeces):
        return [tuple(np.take(tup, indeces)) for tup in self]

    def tup_to_dict(self, result_col=0, is_list=True, indeces=None):
        import package.superdict as sd

        if type(result_col) is not list:
            result_col = [result_col]
        if len(self) == 0:
            return sd.SuperDict()
        if indeces is None:
            indeces = [col for col in range(len(self[0])) if col not in result_col]
        result = sd.SuperDict()
        for tup in self:
            index = tuple(np.take(tup, indeces))
            if len(index) == 1:
                index = index[0]
            content = tuple(np.take(tup, result_col))
            if len(content) == 1:
                content = content[0]
            if not is_list:
                result[index] = content
                continue
            if index not in result:
                result[index] = []
            result[index].append(content)
        return result

