cdef struct Point:
    int X
    int Y

#cdef struct Point Square[2];

cdef struct Square:
    Point DL
    Point UR

# These are auxiliary functions for points, squares, plates.
# They involve coordinates and positions, etc.

#def dict_to_point(point):
#    cdef Point p
#    p.X = point['X']
#    p.Y = point['Y']
#    return p
#
#def dict_to_square(square):
#    cdef Square sq
#    sq.DL.X = square[0].X
#    sq.DL.Y = square[0].Y
#    sq.UR.X = square[1].X
#    sq.UR.Y = square[1].Y
#    return sq

cpdef bint point_in_square(Point point, Square square, bint strict=True):
    # TODO: try with np.any
    """
    :param point: dict with X and Y values
    :param square: a list of two points that define a square.
    **important**: first point of square is bottom left.
    :param strict: does not count being in the borders as being inside
    :param lag: move the node in some distance
    :return: True if point is inside square (with <)
    """

    if strict:
        return \
            square.DL.X < point.X < square.UR.X and \
            square.DL.Y < point.Y < square.UR.Y
    else:
        return \
            square.DL.X <= point.X <= square.UR.X and \
            square.DL.Y <= point.Y <= square.UR.Y


cpdef int square_inside_square(Square square1, Square square2, both_sides=True):
    # TODO: try with np.any
    """
    Tests if square1 is inside square2.
    :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param both_sides: if true, alse see if square2 is inside square1
    :return: number of square inside the other. or 0 if not
    """
    if point_in_square(square1.DL, square2, strict=False):
        if point_in_square(square1.UR, square2, strict=False):
            return 1
    if not both_sides:
        return 0
    if point_in_square(square2.DL, square1, strict=False):
        if point_in_square(square2.UR, square1, strict=False):
            return 2
    return 0


cpdef bint square_intersects_square(square1, square2):
    # TODO: try with np.any
    """
    Tests if some point in square1 is inside square2 (or viceversa).
    :param square1: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :param square2: a list of two dictionaries of type: {'X': 0, 'Y': 0}
    :return: True if both squares share some (smaller) area
    """
    for point in square1.values():
        if point_in_square(point, square2, strict=True):
            return True
    for point in square2.values():
        if point_in_square(point, square1, strict=True):
            return True
    return False


def defect_to_square(defect):
    """
    Reformats a defect to a Square
    :param defect: a dict.
    :return: Square
    """
    return {
        'DL': {'X': defect['X'], 'Y': defect['Y']},
        'UR': {'X': defect['X'] + defect['WIDTH'], 'Y': defect['Y'] + defect['HEIGHT']}
    }


cpdef int plate_inside_plate(plate1, plate2, turn=True, both_sides=False):
    origin = {'X': 0, 'Y': 0}
    result = square_inside_square(
        {'DL': origin, 'UR': {'X': plate1[0], 'Y': plate1[1]}},
        {'DL': origin, 'UR': {'X': plate2[0], 'Y': plate2[1]}},
        both_sides=both_sides
    )
    if result or not turn:
        return result
    return square_inside_square(
        {'DL': origin, 'UR': {'X': plate1[1], 'Y': plate1[0]}},
        {'DL': origin, 'UR': {'X': plate2[0], 'Y': plate2[1]}},
        both_sides=both_sides
    )


def rotate_square(square, ref_pos):
    """
    :param square: 1 element list of dicts with X and Y positions
    :param ref_pos: a single point (dict of X, Y
    :return: square rotated around the ref_pos axis
    """
    inv_v = {'Y': 'X', 'X': 'Y'}
    return {side: {k: point[ik] - ref_pos[ik] + ref_pos[k]
             for k, ik in inv_v.items()}
            for side, point in square.items()}


cpdef bint check_nodespace_in_space(node_space, free_space, bint insert, int min_waste):
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
    nodes_to_check = [(2, 1)]
    cdef int dif
    if not insert:
        nodes_to_check.append((1, 2))
    for n1, n2 in nodes_to_check:
        fs = free_space[n1]
        ns = node_space[n2]
        for d in ['HEIGHT', 'WIDTH']:
            dif = fs[d] - ns[d]
            if dif < min_waste and dif != 0:
                return False
    return True


def get_probs_trees(num_trees, reverse=False):
    step = 2 / (num_trees * (num_trees + 1))
    probs = [(t+1)*step for t in range(num_trees)]
    if reverse:
        probs = list(reversed(probs))
    return probs