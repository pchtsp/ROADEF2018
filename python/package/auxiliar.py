# /usr/bin/python3

# import arrow
import pandas as pd
import numpy as np
import os
import datetime


def clean_dict(dictionary, default_value=0):
    return {key: value for key, value in dictionary.items() if value != default_value}


def tup_filter(tuplist, indeces):
    return [tuple(np.take(tup, indeces)) for tup in tuplist]


def dict_filter(dictionary, indeces):
    if type(indeces) is not list:
        indeces = [indeces]
    bad_elem = np.setdiff1d(indeces, list(dictionary.keys()))
    if len(bad_elem) > 0:
        raise KeyError("following elements not in keys: {}".format(bad_elem))
    return {k: dictionary[k] for k in indeces}


def tup_to_dict(tuplist, result_col=0, is_list=True, indeces=None):
    if type(result_col) is not list:
        result_col = [result_col]
    if len(tuplist) == 0:
        return {}
    if indeces is None:
        indeces = [col for col in range(len(tuplist[0])) if col not in result_col]
    result = {}
    for tup in tuplist:
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


def tup_to_dicts(dict_out, tup, value):
    elem = tup[0]
    if elem not in dict_out:
        dict_out[elem] = {}
    if len(tup) == 1:
        dict_out[elem] = value
        return dict_out
    else:
        tup_to_dicts(dict_out[elem], tup[1:], value)
    return dict_out


def dicts_to_tup(result, keys, content):
    if type(content) is not dict:
        result[tuple(keys)] = content
        return result
    for key, value in content.items():
        dicts_to_tup(result, keys + [key], value)
    return result


def dicttup_to_dictdict(tupdict):
    """
    Useful to get json-compatible objects from the solution
    :param tupdict: a dictionary with tuples as keys
    :return: a (recursive) dictionary of dictionaries
    """
    dictdict = {}
    for tup, value in tupdict.items():
        tup_to_dicts(dictdict, tup, value)
    return dictdict


def dictdict_to_dictup(dictdict):
    """
    Useful when reading a json and wanting to convert it to tuples.
    Opposite to dicttup_to_dictdict
    :param dictdict: a dictionary of dictionaries
    :return: a dictionary with tuples as keys
    """
    return dicts_to_tup({}, [], dictdict)


def dict_list_reverse(dict_in):
    """
    :param dict_in: a dictionary with a list as a result
    :return: a dictionary with the list elements as keys and
    old keys as values.
    """
    new_keys = np.unique([val for l in dict_in.values() for val in l])
    dict_out = {k: [] for k in new_keys}
    for k, v in dict_in.items():
        for el in v:
            dict_out[el].append(k)
    return dict_out


def dict_to_tup(dict_in):
    """
    The last element of the returned tuple was the dict's value.
    :param dict_in: dictionary indexed by tuples
    :return: a list of tuples.
    """
    tup_list = []
    for key, value in dict_in.items():
        if type(key) is tuple:
            tup_list.append(tuple(list(key) + [value]))
        else:
            tup_list.append(tuple([key, value]))
    return tup_list


def tup_replace(tup, pos, value):
    _l = list(tup)
    _l[pos] = value
    return tuple(_l)


def vars_to_tups(var):
    # because of rounding approximations; we need to check if its bigger than half:
    # return [tup for tup in var if var[tup].value()]
    return [tup for tup in var if var[tup].value() > 0.5]


def fill_dict_with_default(dict_in, keys, default=0):
    return {**{k: default for k in keys}, **dict_in}


def get_property_from_dic(dic, property):
    return {key: value[property] for key, value in dic.items() if property in value}


def get_timestamp(form="%Y%m%d%H%M"):
    return datetime.datetime.now().strftime(form)


def dict_to_lendict(dict_input):
    return {k: len(v) for k, v in dict_input.items()}


if __name__ == "__main__":

    content = tup_to_dicts({1: {2: {4: 5}}}, (1, 2, 3), 1)
    result = {}
    result = dicts_to_tup(result, [], {1: {2: {4: 5}}})
    result = dicts_to_tup({}, [], content)
    tup = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
    tup_to_dict(tup, is_list=False)
    # tup_to_dict
    pass
    # pd.DataFrame(aux.dict_to_tup(tupdict)).groupby(level=0).apply(lambda df: df.xs(df.name).to_dict()).to_dict()