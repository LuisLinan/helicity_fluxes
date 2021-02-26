import numpy as np
from dave4m import dave4vm


def odiffxy5(image):
    """Return the gradient of an 2-dimensional array using a five order accurate

    Parameters

    ----------

    image : ndarray
        original 2D-image

    Returns:

    --------

    dx :
        gradient in the first direction
    dy :
        gradient in the second direction

    """


    c1 = 0.12019
    c2 = 0.74038
    dx = (-c1 * np.roll(image, -2, axis=0) + c2 * np.roll(image, -1, axis=0) - c2 * np.roll(image, 1, axis=0) + c1 *
          np.roll(image, 2, axis=0))
    dy = (-c1 * np.roll(image, -2, axis=1) + c2 * np.roll(image, -1, axis=1) - c2 * np.roll(image, 1, axis=1) + c1 *
          np.roll(image, 2, axis=1))

    return dx, dy


def do_dave4vm(bx_start, by_start, bz_start, t_start, bx_stop, by_stop, bz_stop, t_stop, DX, DY, windowsize=20):
    """prepare a structure data for dave4m

    Parameters

    ----------

    bxyz_start :
        vector field at T_start
    bxyz_stop :
        vector field at T_stop


    Returns

    ------

    mag :
        a data structure to be used in dave4m

    """

    DT = float(t_stop - t_start)
    BZT = ((bz_stop - bz_start) / DT).astype('float')

    # use time_centered spatial variables
    BX = (bx_stop + bx_start) / 2.
    BY = (by_stop + by_start) / 2.
    BZ = (bz_stop + bz_start) / 2.

    # compute 5-point optimized derivatives
    BXX, BXY = odiffxy5(BX)
    BYX, BYY = odiffxy5(BY)
    BZX, BZY = odiffxy5(BZ)

    magvm = {'BZT': BZT.copy(), 'BX': BX.copy(), 'BXX': BXX / DX, 'BXY': BXY / DY, 'BY': BY.copy(), 'BYX': BYX / DX,
             'BYY': BYY / DY, 'BZ': BZ.copy(), 'BZX': BZX / DX, 'BZY': BZY / DY, 'DX': DX, 'DY': DY, 'DT': DT}

    wsize = windowsize
    vel4vm, _, _ = dave4vm(magvm, wsize)

    return magvm, vel4vm
