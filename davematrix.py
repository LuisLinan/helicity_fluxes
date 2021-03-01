import numpy as np
from scipy import ndimage


def corr_convol(image, kernel, size):
    convol = ndimage.filters.correlate(image, kernel)
    s = kernel.shape
    indexi = [int(np.ceil((s[0] - 1) / 2.)), int(np.floor((s[0] - 1) / 2.))]
    indexj = [int(np.ceil((s[1] - 1) / 2.)), int(np.floor((s[1] - 1) / 2.))]
    convol[np.arange(indexi[0]), :] = 0
    convol[-np.arange(indexi[1]) - 1, :] = 0
    convol[:, np.arange(indexj[0])] = 0
    convol[:, -np.arange(indexj[1]) - 1] = 0
    return convol.reshape(1, size[0], size[1])


def dave4vm_matrix(Bx, Bxx, By, Byy, Bz, Bzx, Bzy, Bzt, psf, psfx, psfy, psfxx, psfyy, psfxy):
    """Construct  matrix elements for LKA algoritm

    Remarks

    -------

    Dave elements (depend only on Bz, Bzx, Bzy, Bzt

    """


    sz = Bz.shape
    A = np.zeros((10, 10, sz[0], sz[1]))

    G = corr_convol(Bz * Bz, psf, sz)

    GGx = Bz * Bzx
    Gx = corr_convol(GGx, psf, sz)

    xGx = corr_convol(GGx, psfx, sz)

    yGx = corr_convol(GGx, psfy, sz)

    GGy = Bz * Bzy
    Gy = corr_convol(GGy, psf, sz)

    xGy = corr_convol(GGy, psfx, sz)

    yGy = corr_convol(GGy, psfy, sz)

    GGt = Bzt * Bz
    Ht = corr_convol(GGt, psf, sz)

    GGxx = Bzx * Bzx
    Gxx = corr_convol(GGxx, psf, sz)

    GGyy = Bzy * Bzy
    Gyy = corr_convol(GGyy, psf, sz)

    GGxy = Bzx * Bzy
    Gxy = corr_convol(GGxy, psf, sz)

    GGtx = Bzt * Bzx
    Gtx = corr_convol(GGtx, psf, sz)

    GGty = Bzt * Bzy
    Gty = corr_convol(GGty, psf, sz)

    xGxx = corr_convol(GGxx, psfx, sz)

    xGyy = corr_convol(GGyy, psfx, sz)

    xGxy = corr_convol(GGxy, psfx, sz)

    xGtx = corr_convol(GGtx, psfx, sz)

    xGty = corr_convol(GGty, psfx, sz)

    yGxx = corr_convol(GGxx, psfy, sz)

    yGyy = corr_convol(GGyy, psfy, sz)

    yGxy = corr_convol(GGxy, psfy, sz)

    yGtx = corr_convol(GGtx, psfy, sz)

    yGty = corr_convol(GGty, psfy, sz)

    xxGxx = corr_convol(GGxx, psfxx, sz)

    xxGxy = corr_convol(GGxy, psfxx, sz)

    xxGyy = corr_convol(GGyy, psfxx, sz)

    xyGxx = corr_convol(GGxx, psfxy, sz)

    xyGyy = corr_convol(GGyy, psfxy, sz)

    xyGxy = corr_convol(GGxy, psfxy, sz)

    yyGxx = corr_convol(GGxx, psfyy, sz)

    yyGxy = corr_convol(GGxy, psfyy, sz)

    yyGyy = corr_convol(GGyy, psfyy, sz)

    GGtt = Bzt * Bzt
    Gtt = corr_convol(GGtt, psf, sz)

    BxBx = corr_convol(Bx * Bx, psf, sz)
    ByBy = corr_convol(By * By, psf, sz)
    BxBy = corr_convol(Bx * By, psf, sz)
    BzBx = corr_convol(Bz * Bx, psf, sz)
    BzBy = corr_convol(Bz * By, psf, sz)

    BxBxx = corr_convol(Bx * Bxx, psf, sz)
    BxByy = corr_convol(Bx * Byy, psf, sz)
    BxxBxx = corr_convol(Bxx * Bxx, psf, sz)
    ByyByy = corr_convol(Byy * Byy, psf, sz)
    BxxByy = corr_convol(Bxx * Byy, psf, sz)
    ByBxx = corr_convol(By * Bxx, psf, sz)
    ByByy = corr_convol(By * Byy, psf, sz)

    BzBxx = corr_convol(Bz * Bxx, psf, sz)
    BzByy = corr_convol(Bz * Byy, psf, sz)

    BztBxx = corr_convol(Bzt * Bxx, psf, sz)
    BztByy = corr_convol(Bzt * Byy, psf, sz)

    BzxBx = corr_convol(Bzx * Bx, psf, sz)
    BzxBy = corr_convol(Bzx * By, psf, sz)
    BzxBxx = corr_convol(Bzx * Bxx, psf, sz)
    BzxByy = corr_convol(Bzx * Byy, psf, sz)

    BzyBx = corr_convol(Bzy * Bx, psf, sz)
    BzyBy = corr_convol(Bzy * By, psf, sz)
    BzyBxx = corr_convol(Bzy * Bxx, psf, sz)
    BzyByy = corr_convol(Bzy * Byy, psf, sz)

    BztBx = corr_convol(Bzt * Bx, psf, sz)
    BztBy = corr_convol(Bzt * By, psf, sz)

    xBzxBx = corr_convol(Bzx * Bx, psfx, sz)
    xBzxBy = corr_convol(Bzx * By, psfx, sz)
    xBzyBx = corr_convol(Bzy * Bx, psfx, sz)
    xBzyBy = corr_convol(Bzy * By, psfx, sz)

    yBzyBx = corr_convol(Bzy * Bx, psfy, sz)
    yBzyBy = corr_convol(Bzy * By, psfy, sz)

    yBzxBx = corr_convol(Bzx * Bx, psfy, sz)
    yBzxBy = corr_convol(Bzx * By, psfy, sz)

    yBxBxx = corr_convol(Bx * Bxx, psfy, sz)
    yBxByy = corr_convol(Bx * Byy, psfy, sz)

    yByBxx = corr_convol(By * Bxx, psfy, sz)
    yByByy = corr_convol(By * Byy, psfy, sz)

    xByBxx = corr_convol(By * Bxx, psfx, sz)
    xByByy = corr_convol(By * Byy, psfx, sz)

    xBzxBxx = corr_convol(Bzx * Bxx, psfx, sz)
    xBzxByy = corr_convol(Bzx * Byy, psfx, sz)

    yBzxBxx = corr_convol(Bzx * Bxx, psfy, sz)
    yBzxByy = corr_convol(Bzx * Byy, psfy, sz)

    xBxxBxx = corr_convol(Bxx * Bxx, psfx, sz)
    xBxxByy = corr_convol(Bxx * Byy, psfx, sz)
    xByyByy = corr_convol(Byy * Byy, psfx, sz)

    yBxxBxx = corr_convol(Bxx * Bxx, psfy, sz)
    yBxxByy = corr_convol(Bxx * Byy, psfy, sz)
    yByyByy = corr_convol(Byy * Byy, psfy, sz)

    xBxBxx = corr_convol(Bx * Bxx, psfx, sz)
    xBxByy = corr_convol(Bx * Byy, psfx, sz)

    xBzBxx = corr_convol(Bz * Bxx, psfx, sz)
    xBzByy = corr_convol(Bz * Byy, psfx, sz)

    xBztBxx = corr_convol(Bzt * Bxx, psfx, sz)
    xBztByy = corr_convol(Bzt * Byy, psfx, sz)

    yBztBxx = corr_convol(Bzt * Bxx, psfy, sz)
    yBztByy = corr_convol(Bzt * Byy, psfy, sz)

    xyBxxBxx = corr_convol(Bxx * Bxx, psfxy, sz)
    xyBxxByy = corr_convol(Bxx * Byy, psfxy, sz)
    xyByyByy = corr_convol(Byy * Byy, psfxy, sz)

    xyBzxBxx = corr_convol(Bzx * Bxx, psfxy, sz)
    xyBzxByy = corr_convol(Bzx * Byy, psfxy, sz)
    xyBzyBxx = corr_convol(Bzy * Bxx, psfxy, sz)
    xyBzyByy = corr_convol(Bzy * Byy, psfxy, sz)

    yBzBxx = corr_convol(Bz * Bxx, psfy, sz)
    yBzByy = corr_convol(Bz * Byy, psfy, sz)

    xBzyBxx = corr_convol(Bzy * Bxx, psfx, sz)
    xBzyByy = corr_convol(Bzy * Byy, psfx, sz)
    yBzyBxx = corr_convol(Bzy * Bxx, psfy, sz)
    yBzyByy = corr_convol(Bzy * Byy, psfy, sz)

    xxBxxBxx = corr_convol(Bxx * Bxx, psfxx, sz)
    xxBxxByy = corr_convol(Bxx * Byy, psfxx, sz)
    xxByyByy = corr_convol(Byy * Byy, psfxx, sz)

    xxBzxBxx = corr_convol(Bzx * Bxx, psfxx, sz)
    xxBzyBxx = corr_convol(Bzy * Bxx, psfxx, sz)
    xxBzxByy = corr_convol(Bzx * Byy, psfxx, sz)
    xxBzyByy = corr_convol(Bzy * Byy, psfxx, sz)

    yyBxxBxx = corr_convol(Bxx * Bxx, psfyy, sz)
    yyBxxByy = corr_convol(Bxx * Byy, psfyy, sz)
    yyByyByy = corr_convol(Byy * Byy, psfyy, sz)

    yyBzyBxx = corr_convol(Bzy * Bxx, psfyy, sz)
    yyBzyByy = corr_convol(Bzy * Byy, psfyy, sz)

    yyBzxBxx = corr_convol(Bzx * Bxx, psfyy, sz)
    yyBzxByy = corr_convol(Bzx * Byy, psfyy, sz)

    A = np.array([Gxx, Gxy, Gx + xGxx, Gx + yGxy, yGxx, xGxy, -BzxBxx - BzxByy, -BzxBx - xBzxBxx - xBzxByy, -BzxBy -
                  yBzxBxx - yBzxByy, Gtx, Gxy, Gyy, Gy + xGxy, Gy + yGyy, yGxy, xGyy, -BzyBxx - BzyByy, -BzyBx -
                  xBzyBxx - xBzyByy, -BzyBy - yBzyBxx - yBzyByy, Gty, Gx + xGxx, Gy + xGxy, G + 2 * xGx + xxGxx, G +
                  xGx + xyGxy + yGy, xyGxx + yGx, xGy + xxGxy, -BzBxx - BzByy - xBzxBxx - xBzxByy, -BzBx - xBzBxx -
                  xBzByy - xBzxBx - xxBzxBxx - xxBzxByy, -BzBy - xBzxBy - xyBzxBxx - xyBzxByy - yBzBxx - yBzByy, Ht +
                  xGtx, Gx + yGxy, Gy + yGyy, G + xGx + xyGxy + yGy, G + 2 * yGy + yyGyy, yGx + yyGxy, xGy +
                  xyGyy, -BzBxx - BzByy - yBzyBxx - yBzyByy, -BzBx - xBzBxx - xBzByy - xyBzyBxx - xyBzyByy -
                  yBzyBx, -BzBy - yBzBxx - yBzByy - yBzyBy - yyBzyBxx - yyBzyByy, Ht + yGty, yGxx, yGxy, xyGxx +
                  yGx, yGx + yyGxy, yyGxx, xyGxy, -yBzxBxx - yBzxByy, -xyBzxBxx - xyBzxByy - yBzxBx, -yBzxBy -
                  yyBzxBxx - yyBzxByy, yGtx, xGxy, xGyy, xGy + xxGxy, xGy + xyGyy, xyGxy, xxGyy, -xBzyBxx -
                  xBzyByy, -xBzyBx - xxBzyBxx - xxBzyByy, -xBzyBy - xyBzyBxx - xyBzyByy, xGty, -BzxBxx -
                  BzxByy, -BzyBxx - BzyByy, -BzBxx - BzByy - xBzxBxx - xBzxByy, -BzBxx - BzByy - yBzyBxx -
                  yBzyByy, -yBzxBxx - yBzxByy, -xBzyBxx - xBzyByy, BxxBxx + 2 * BxxByy + ByyByy, BxBxx + BxByy +
                  xBxxBxx + 2 * xBxxByy + xByyByy, ByBxx + ByByy + yBxxBxx + 2 * yBxxByy + yByyByy, -BztBxx -
                  BztByy, -BzxBx - xBzxBxx - xBzxByy, -BzyBx - xBzyBxx - xBzyByy, -BzBx - xBzBxx - xBzByy - xBzxBx -
                  xxBzxBxx - xxBzxByy, -BzBx - xBzBxx - xBzByy - xyBzyBxx - xyBzyByy - yBzyBx, -xyBzxBxx - xyBzxByy -
                  yBzxBx, -xBzyBx - xxBzyBxx - xxBzyByy, BxBxx + BxByy + xBxxBxx + 2 * xBxxByy + xByyByy, BxBx +
                  2 * xBxBxx + 2 * xBxByy + xxBxxBxx + 2 * xxBxxByy + xxByyByy, BxBy + xByBxx + xByByy + xyBxxBxx +
                  2 * xyBxxByy + xyByyByy + yBxBxx + yBxByy, -BztBx - xBztBxx - xBztByy, -BzxBy - yBzxBxx -
                  yBzxByy, -BzyBy - yBzyBxx - yBzyByy, -BzBy - xBzxBy - xyBzxBxx - xyBzxByy - yBzBxx - yBzByy, -BzBy -
                  yBzBxx - yBzByy - yBzyBy - yyBzyBxx - yyBzyByy, -yBzxBy - yyBzxBxx - yyBzxByy, -xBzyBy - xyBzyBxx -
                  xyBzyByy, ByBxx + ByByy + yBxxBxx + 2 * yBxxByy + yByyByy, BxBy + xByBxx + xByByy + xyBxxBxx +
                  2 * xyBxxByy + xyByyByy + yBxBxx + yBxByy, ByBy + 2 * yByBxx + 2 * yByByy + yyBxxBxx + 2 * yyBxxByy +
                  yyByyByy, -BztBy - yBztBxx - yBztByy, Gtx, Gty, Ht + xGtx, Ht + yGty, yGtx, xGty, -BztBxx -
                  BztByy, -BztBx - xBztBxx - xBztByy, -BztBy - yBztBxx - yBztByy, Gtt])

    return A.reshape(10, 10, sz[0], sz[1])
