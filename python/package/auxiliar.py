# /usr/bin/python3

import datetime


def tup_replace(tup, pos, value):
    _l = list(tup)
    _l[pos] = value
    return tuple(_l)


def get_timestamp(form="%Y%m%d%H%M"):
    return datetime.datetime.now().strftime(form)


if __name__ == "__main__":
    pass