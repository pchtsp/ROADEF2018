

# These are auxiliary functions for points, squares, plates.
# They involve coordinates and positions, etc.


def point_in_square(point, square, strict=True, lag=None):
    """
    :param point: dict with X and Y values
    :param square: a list of two points that define a square.
    **important**: first point of square is bottom left.
    :param strict: does not count being in the borders as being inside
    :param lag: move the node in some distance
    :return: True if point is inside square (with <)
    """
    if lag is None:
        lag = {'X': 0, 'Y': 0}
    if strict:
        return \
            square[0]['X'] < point['X'] + lag['X'] < square[1]['X'] and \
            square[0]['Y'] < point['Y'] + lag['Y'] < square[1]['Y']
    else:
        return \
            square[0]['X'] <= point['X'] + lag['X'] <= square[1]['X'] and \
            square[0]['Y'] <= point['Y'] + lag['Y'] <= square[1]['Y']


def square_inside_square(square1, square2, both_sides=True):
    """
    Tests if square1 is inside square2.
    :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param both_sides: if true, alse see if square2 is inside square1
    :return: number of square inside the other. or 0 if not
    """
    if point_in_square(square1[0], square2, strict=False):
        if point_in_square(square1[1], square2, strict=False):
            return 1
    if not both_sides:
        return 0
    if point_in_square(square2[0], square1, strict=False):
        if point_in_square(square2[1], square1, strict=False):
            return 2
    return 0


def square_intersects_square(square1, square2):
    """
    Tests if some point in square1 is inside square2 (or viceversa).
    :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :return: True if both squares share some (smaller) area
    """
    for point in square1:
        if point_in_square(point, square2, strict=True):
            return True
    for point in square2:
        if point_in_square(point, square1, strict=True):
            return True
    return False


def defect_to_square(defect):
    """
    Reformats a defect to a list of two points
    :param defect: a dict.
    :return: list of two points {'X': 1, 'Y': 1}
    """
    return [{'X': defect['X'], 'Y': defect['Y']},
            {'X': defect['X'] + defect['WIDTH'], 'Y': defect['Y'] + defect['HEIGHT']}]


def plate_inside_plate(plate1, plate2, turn=True, both_sides=False):
    origin = {'X': 0, 'Y': 0}
    result = square_inside_square(
        [origin, {'X': plate1[0], 'Y': plate1[1]}],
        [origin, {'X': plate2[0], 'Y': plate2[1]}],
        both_sides=both_sides
    )
    if result or not turn:
        return result
    return square_inside_square(
        [origin, {'X': plate1[1], 'Y': plate1[0]}],
        [origin, {'X': plate2[0], 'Y': plate2[1]}],
        both_sides=both_sides
    )


def check_nodespace_in_space(node_space, free_space, insert, min_waste):
    """
    :param node_space: {1: {WIDTH: XX, HEIGHT: XX}, 2: {WIDTH: XX, HEIGHT: XX}}
    :param free_space: {1: {WIDTH: XX, HEIGHT: XX}, 2: {WIDTH: XX, HEIGHT: XX}}
    :param insert: if true, we do not check the other node.
    :param min_waste: min size of waste
    :return:
    """
    # if dimensions are too small, we can't do the change
    # in insert=True we only check node1 movement
    # Important! we want to have at least 20 of waste. Or 0.
    for d in ['HEIGHT', 'WIDTH']:
        dif = free_space[2][d] - node_space[1][d]
        if dif < min_waste and dif != 0:
            return False
        if insert:
            continue
        dif = free_space[1][d] - node_space[2][d]
        if dif < min_waste and dif != 0:
            return False
    return True
