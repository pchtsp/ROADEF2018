import numpy as np


class SuperDict(dict):

    def keys_l(self):
        return list(self.keys())

    def values_l(self):
        return list(self.values())

    def clean(self, default_value=0):
        return SuperDict({key: value for key, value in self.items() if value != default_value})

    def filter(self, indices):
        if type(indices) is not list:
            indices = [indices]
        bad_elem = np.setdiff1d(indices, list(self.keys()))
        if len(bad_elem) > 0:
            raise KeyError("following elements not in keys: {}".format(bad_elem))
        return SuperDict({k: self[k] for k in indices})

    def to_dictdict(self):
        """
        Useful to get json-compatible objects from the solution
        :param self: a dictionary with tuples as keys
        :return: a (recursive) dictionary of dictionaries
        """
        dictdict = SuperDict()
        for tup, value in self.items():
            self.tup_to_dicts(tup, value)
        return dictdict

    def tup_to_dicts(self, tup, value):
        elem = tup[0]
        if elem not in self:
            self[elem] = SuperDict()
        if len(tup) == 1:
            self[elem] = value
            return self
        else:
            self[elem].tup_to_dicts(tup[1:], value)
        return self

    def dicts_to_tup(self, keys, content):
        if type(content) is not dict:
            self[tuple(keys)] = content
            return self
        for key, value in content.items():
            self.dicts_to_tup(keys + [key], value)
        return self

    def to_dictup(self):
        """
        Useful when reading a json and wanting to convert it to tuples.
        Opposite to dicttup_to_dictdict
        :param self: a dictionary of dictionaries
        :return: a dictionary with tuples as keys
        """
        return SuperDict().dicts_to_tup([], self)

    def list_reverse(self):
        """
        :param self: a dictionary with a list as a result
        :return: a dictionary with the list elements as keys and
        old keys as values.
        """
        new_keys = np.unique([val for l in self.values() for val in l])
        dict_out = SuperDict({k: [] for k in new_keys})
        for k, v in self.items():
            for el in v:
                dict_out[el].append(k)
        return dict_out

    def to_tuplist(self):
        """
        The last element of the returned tuple was the dict's value.
        :param self: dictionary indexed by tuples
        :return: a list of tuples.
        """
        import package.tuplist as tl

        tup_list = tl.TupList()
        for key, value in self.items():
            if type(key) is tuple:
                tup_list.append(tuple(list(key) + [value]))
            else:
                tup_list.append(tuple([key, value]))
        return tup_list

    def fill_with_default(self, keys, default=0):
        return SuperDict({**{k: default for k in keys}, **self})

    def get_property(self, property):
        return {key: value[property] for key, value in self.items() if property in value}

    def to_lendict(self):
        return {k: len(v) for k, v in self.items()}

    def index_by_property(self, property, get_list=False):
        el = self.keys_l()[0]
        if property not in self[el]:
            raise IndexError('property {} is not present in el {} of dict {}'.
                             format(property, el, self))

        result = {v[property]: {} for v in self.values()}
        for k, v in self.items():
            result[v[property]][k] = v

        result = SuperDict.from_dict(result)
        if get_list:
            return result.values_l()
        return result

    @classmethod
    def from_dict(cls, dictionary):
        if type(dictionary) is not dict:
            return
        dictionary = cls(dictionary)
        for value in dictionary.values():
            cls.from_dict(value)
        return dictionary
