import numpy as np


def place_mirror(im, x1, x2, y1, y2, mr):
    """ Place an image mr in specified locations of an image im. The edge locations in im where mr is to be placed are
    (x1,y1) and (x2,y2)

    Programmer

    ---------

    Manolis K. Georgoulis (JHU/APL, 10/12/05)


    """

    nxa = np.zeros(2)
    nya = np.zeros(2)
    res = im[x1:x2 + 1, y1:y2 + 1].shape
    nxa[0] = res[0]
    nya[0] = res[1]
    res = mr.shape
    nxa[1] = res[0]
    nya[1] = res[1]

    nx = np.min(nxa)
    ny = np.min(nya)
    im[x1:x1 + nx, y1:y1 + ny] = mr[0:nx, 0:ny]

    return im


def expand_image(im, ext_x, ext_y, mirror=0):
    """Enlarge the linear dimensions of an image by a (ext_x,ext_y) and put the initial image at the center.
    If the keyword /mirror is set, the additional space corresponds to a mirror image of the initial image.


    Programmer

    ----------

    Manolis K. Georgoulis (JHU/APL, 09/30/05)

    """

    res = im.shape
    id1 = res[0]
    id2 = res[1]

    mim = np.zeros((int(id1 + ext_x), int(id2 + ext_y)))
    stx = np.fix(np.float(ext_x) / 2. + 0.5)
    sty = np.fix(np.float(ext_y) / 2. + 0.5)
    mim[int(stx):int(stx + id1), int(sty):int(sty + id2)] = im

    if mirror != 0:
        if stx <= id1:
            xmr = int(stx)
        else:
            xmr = int(id1)
        mr1 = im[0:xmr, :]
        mr1 = np.flip(mr1, axis=0)
        mr2 = im[id1 - xmr:id1, :]
        mr2 = np.flip(mr2, axis=0)
        mim = place_mirror(mim, 0, stx - 1, sty, sty + id2 - 1, mr1)
        mim = place_mirror(mim, stx + id1, id1 + ext_x - 1, sty, sty + id2 - 1, mr2)
        if sty <= id2:
            ymr = int(sty)
        else:
            ymr = int(id2)
        mr1 = mim[:, ymr:2 * ymr]
        mr1 = np.flip(mr1, axis=1)
        mr2 = mim[:, id2:ymr + id2]
        mr2 = np.flip(mr2, axis=1)
        mim = place_mirror(mim, 0, id1 + ext_x - 1, 0, sty - 1, mr1)
        mim = place_mirror(mim, 0, id1 + ext_x - 1, sty + id2, id2 + ext_y - 1, mr2)

    return mim, stx, sty
