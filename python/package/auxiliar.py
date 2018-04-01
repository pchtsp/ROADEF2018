# /usr/bin/python3

import datetime


def tup_replace(tup, pos, value):
    _l = list(tup)
    _l[pos] = value
    return tuple(_l)


def vars_to_tups(var):
    # because of rounding approximations; we need to check if its bigger than half:
    # return [tup for tup in var if var[tup].value()]
    import package.tuplist as tl
    
    return tl.TupList([tup for tup in var if var[tup].value() > 0.5])


def get_timestamp(form="%Y%m%d%H%M"):
    return datetime.datetime.now().strftime(form)


if __name__ == "__main__":
    pass