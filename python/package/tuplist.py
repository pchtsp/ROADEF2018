import numpy as np


class TupList(list):

    def filter(self, indices):
        """
        filters the tuple of each element of the list according
        to a list of positions
        :param indices: a list of positions
        :return: a new TuplList with the modifications
        """
        if type(indices) is not list:
            # indices = [indices]
            return TupList([np.take(tup, indices) for tup in self])
        return TupList([tuple(np.take(tup, indices)) for tup in self])

    def to_dict(self, result_col=0, is_list=True, indices=None):
        """
        This magic function converts a tuple list into a dictionary
        by taking one or several of the columns as the result.
        :param result_col: a list of positions of the tuple for the result
        :param is_list: the value of the dictionary will be a list?
        :param indices: optional way of determining the indeces instead of
            being the complement of result_col
        :return: a dictionary
        """
        import package.superdict as sd

        if type(result_col) is not list:
            result_col = [result_col]
        if len(self) == 0:
            return sd.SuperDict()
        if indices is None:
            indices = [col for col in range(len(self[0])) if col not in result_col]
        result = sd.SuperDict()
        for tup in self:
            index = tuple(np.take(tup, indices))
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

    def add(self, *args):
        """
        this is just a shirtcut for doing
            list.append((arg1, arg2, arg3))
        by doing:
            list.add(arg1, arg2, arg3)
        which is a little more friendly and short
        :param args: any number of elements to append
        :return: nothing.
        """
        self.append(tuple(args))

    def unique(self):
        return TupList(np.unique(self))

    def unique2(self):
        return TupList(set(self))