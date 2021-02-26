import numpy as np
from scipy import ndimage
from scipy import linalg


def convol(image, kernel):
    convol = ndimage.filters.correlate(image, kernel)
    s = kernel.shape
    indexi = [int(np.ceil((s[0] - 1) / 2.)), int(np.floor((s[0] - 1) / 2.))]
    indexj = [int(np.ceil((s[1] - 1) / 2.)), int(np.floor((s[1] - 1) / 2.))]
    convol[np.arange(indexi[0]), :] = 0
    convol[-np.arange(indexi[1]) - 1, :] = 0
    convol[:, np.arange(indexj[0])] = 0
    convol[:, -np.arange(indexj[1]) - 1] = 0
    return convol


def dave_multi(fwhm, first_image, second_image, middle_image=0, adv=0, source=0, np_deriv=3, sigma=0, chisq=0, noise=0):
    """   Determine 6 parameters defining the affine velocity field at a small area at all the points

    Parameters

    ----------

    fwhm :
        FWHM of the window function (shoud be 2, 4, 6, and so on )
    first_image :
        Image at t1
    second_image :
        Image at t2
    middle_image :
        Image at (t1+t2)/2


    Returns

    -------

    Result :
        parameters at  all the pixels of the images
        Result(0,*,*) : x-component of velocity (U_0)
        Result(1,*,*) : y-component of velocity (V_0)
        Result(2,*,*) : x-derivative of x-component (U_x)
        Result(3,*,*) : y-derivative of y-component (V_y)
        Result(4,*,*) : y-derivative of x-component (U_y)
        Result(5,*,*) : x-derivative of y-component (V_x)
        Result(6,*,*) : nu #  image varies with time in proportion to exp(nu*t)

        It is assumed that the velocity field around (x_0, y_0) is of the form
            vx = U_0 + U_x * (x-x_0) + U_y * (y-y_0)
            vy = V_0 + V_x * (x-x_0) + V_y * (y-y_0)

    Remarks

    ------

    The time unit is the time interval between the first and second images, and the length unit is the pixel size.

    References

    ----------

    Schuck, P. W. 2005, ApJL, 632, L53
    Schuck, P. W. 2006 ApJ
    Chae, J. et al.  2008

    History

    -------

    2006 June,  firstly coded by J. Chae
    2007 Jan, J. Chae,  modified the window function array "w"
    2007 April, J. Chae,  introduced a new free parameter  "nu"
    2009 Februrary, J. Chae
    2021 Feburary Python version by L. Linan

    """

    if adv == 1:
        psw = 0
    else:
        psw = 1
    if source == 1:
        qsw = 1
    else:
        qsw = 0
    if np_deriv == 0:
        np_deriv = 3
    if noise == 0:
        noise = 1.

    first_image_nan = np.isfinite(first_image)
    s_first_image_nan = np.where(first_image_nan == True)
    n_first_image_nan = len(s_first_image_nan)

    if n_first_image_nan >= 1:
        first_image[s_first_image_nan] = 0.

    second_image_nan = np.isfinite(second_image)
    s_second_image_nan = np.where(second_image_nan == True)
    n_second_image_nan = len(s_second_image_nan)
    if n_second_image_nan >= 1:
        second_image[s_second_image_nan] = 0.

    s = first_image.shape
    nx = s[0]
    ny = s[1]

    # Constructing derivatives
    if middle_image != 0 and middle_image.shape == first_image.shape:
        im = middle_image
        im_t = (second_image - first_image) / 2.
    else:
        im = 0.5 * (first_image + second_image)
        im_t = (second_image - first_image)

    if np_deriv == 3:
        kernel = np.array([-1.0, 0.0, 1.0]) / 2.
        kernel = kernel.reshape(-1, 1)
    if np_deriv == 5:
        kernel = np.array([0.12019, -0.74038, 0, 0.74038, -0.12019])
        kernel = kernel.reshape(-1, 1)

    im_x = convol(im, kernel)
    im_y = convol(im, kernel.transpose())

    npar = 6 + qsw

    # Constructing window function
    wfunction = 'gaussian'
    if wfunction == 'square':
        mf = 1
    else:
        mf = 2

    hh = min(np.fix(fwhm / 2.), (nx / 2 / mf - 2), (ny / 2 / mf - 2))
    hh = [hh, hh]

    nxs = int(2 * hh[0] * mf + 1)
    nys = int(2 * hh[1] * mf + 1)

    xs = np.array(np.arange(nxs) - int(nxs / 2)).repeat(nys).reshape(nxs, nys)
    ys = np.array(np.arange(nys) - int(nys / 2)).repeat(nxs).reshape(nys, nxs).transpose()

    if wfunction == 'square':
        w = np.zeros((2, 4)) + 1.0
    elif wfunction == 'gaussian':
        w = np.exp(-np.log(2.0) * ((xs / float(hh[0])) ** 2 + (ys / float(hh[1])) ** 2))
    elif wfunction == 'hanning':
        w = (1 + np.cos(np.pi * xs * hh[0] / 2.0)) * (1 + np.cos(np.pi * ys / hh[1] / 2.)) / 4.0

    w = w / noise ** 2

    # Constructing coefficent arrays

    A = np.zeros((nx, ny, npar, npar), dtype='float')

    A[:, :, 0, 0] = convol(im_x * im_x, w)  # U0, U0
    A[:, :, 0, 1] = convol(im_x * im_y, w)  # U0, V0
    A[:, :, 1, 1] = convol(im_y * im_y, w)  # V0, V0
    A[:, :, 0, 2] = convol(im_x * im_x, xs * w) + psw * convol(im_x * im,
                                                               w)  # U0, Ux
    A[:, :, 1, 2] = convol(im_y * im_x, xs * w) + psw * convol(im_y * im,
                                                               w)  # V0, Ux
    A[:, :, 2, 2] = convol(im_x * im_x, xs * xs * w) + \
                    2 * psw * convol(im_x * im, xs * w) + psw ** 2 * convol(
        im * im, w)  # Ux, Ux
    A[:, :, 0, 3] = convol(im_x * im_y, ys * w) + psw * convol(im_x * im,
                                                               w)  # U0, Vy
    A[:, :, 1, 3] = convol(im_y * im_y, ys * w) + psw * convol(im_y * im,
                                                               w)  # V0, Vy
    A[:, :, 2, 3] = convol(im_x * im_y, xs * ys * w) + psw * convol(im * im_y,
                                                                    ys * w) + \
                    psw * convol(im_x * im, xs * w) + psw ** 2 * convol(im * im,
                                                                        w)  # Ux, Vy
    A[:, :, 3, 3] = convol(im_y * im_y, ys * ys * w) + 2 * psw * convol(im_y * im,
                                                                        ys * w) + \
                    psw ** 2 * convol(im * im, w)  # Vy, Vy
    A[:, :, 0, 4] = convol(im_x * im_x, ys * w)  # U0, Uy
    A[:, :, 1, 4] = convol(im_y * im_x, ys * w)  # V0, Uy
    A[:, :, 2, 4] = convol(im_x * im_x, xs * ys * w) + psw * convol(im * im_x,
                                                                    ys * w)  # Ux, Uy
    A[:, :, 3, 4] = convol(im_y * im_x, ys * ys * w) + psw * convol(im * im_x,
                                                                    ys * w)  # Vy, Uy
    A[:, :, 4, 4] = convol(im_x * im_x, ys * ys * w)  # Uy, Uy
    A[:, :, 0, 5] = convol(im_x * im_y, xs * w)  # U0, Vx
    A[:, :, 1, 5] = convol(im_y * im_y, xs * w)  # V0, Vx
    A[:, :, 2, 5] = convol(im_x * im_y, xs * xs * w) + psw * convol(im * im_y,
                                                                    xs * w)  # Ux, Vx
    A[:, :, 3, 5] = convol(im_y * im_y, ys * xs * w) + psw * convol(im * im_y,
                                                                    xs * w)  # Vy, Vx
    A[:, :, 4, 5] = convol(im_x * im_y, ys * xs * w)  # Uy, Vx
    A[:, :, 5, 5] = convol(im_y * im_y, xs * xs * w)

    # Vx, Vx
    if qsw != 0:
        A[:, :, 0, 6] = -qsw * convol(im_x * im, w)  # U0, mu
        A[:, :, 1, 6] = -qsw * convol(im_y * im, w)  # V0, mu
        A[:, :, 2, 6] = -qsw * convol(im_x * im, xs * w) - qsw * psw * convol(
            im * im, w)  # Ux, mu
        A[:, :, 3, 6] = -qsw * convol(im_y * im, ys * w) - psw * qsw * convol(
            im * im, w)  # Vy, mu
        A[:, :, 4, 6] = -qsw * convol(im_x * im, ys * w)  # Uy, mu
        A[:, :, 5, 6] = -qsw * convol(im_y * im, xs * w)  # Vx, mu
        A[:, :, 6, 6] = qsw ** 2 * convol(im * im, w)  # mu, mu

    for i in range(1, npar):
        for j in range(i):
            A[:, :, i, j] = A[:, :, j, i]

    B = np.zeros((nx, ny, npar), dtype='float')
    B[:, :, 0] = convol(im_t * im_x, -w)
    B[:, :, 1] = convol(im_t * im_y, -w)
    B[:, :, 2] = convol(im_t * im, -w) * psw + convol(im_t * im_x, -xs * w)
    B[:, :, 3] = convol(im_t * im, -w) * psw + convol(im_t * im_y, -ys * w)
    B[:, :, 4] = convol(im_t * im_x, -ys * w)
    B[:, :, 5] = convol(im_t * im_y, -xs * w)

    if qsw != 0:
        B[:, :, 6] = qsw * convol(im_t * (-im), -w)

    # Solving the lienear equations

    result = np.zeros((npar, nx, ny))
    sigmacal = 0
    if (chisq != 0) or (sigma != 0):
        chisq = np.zeros((nx, ny), dtype='float')
        sigma = np.zeros((npar, nx, ny,), dtype='float')
        sigmacal = 1

    for xx in range(int(hh[0]), int(nx - hh[0])):
        for yy in range(int(hh[1]), int(ny - hh[1])):
            AA = A[xx, yy, :, :]
            BB = B[xx, yy, :]
            _, ww, vv = linalg.svd(AA)
            X, _, _, _ = np.linalg.lstsq(AA, BB)
            result[:, xx, yy] = X
            if sigmacal != 0:
                delxh = 0.5 * (result[0, xx, yy] + result[2, xx, yy] * xs + result[4, xx, yy] * ys)
                delyh = 0.5 * (result[1, xx, yy] + result[5, xx, yy] * xs + result[3, xx, yy] * ys)
                i = xx + xs
                j = yy + ys
                sx = np.fix(delxh) - [1 if delxh < 0 else 0][0]
                sy = np.fix(delyh) - [1 if delyh < 0 else 0][0]
                ex = delxh - sx
                ey = delyh - sy
                Fv = first_image[i - sx, j - sy] * (1 - ex) * (1 - ey) + first_image[i - sx - 1, j - sy] * ex * (1 - ey) \
                     + first_image[i - sx, j - sy - 1] * (1 - ex) * ey + first_image[i - sx - 1, j - sy - 1] * ex * ey
                Sv = second_image[i + sx, j + sy] * (1 - ex) * (1 - ey) + second_image[i + sx + 1, j + sy] * ex * (
                        1 - ey) \
                     + second_image[i + sx, j + sy + 1] * (1 - ex) * ey + second_image[i + sx + 1, j + sy + 1] * ex * ey
                nu = -(result[2, xx, yy] + result[3, xx, yy]) * psw
                if qsw != 0:
                    nu = nu + qsw * result[6, xx, yy]
                sdiv = np.exp(-nu / 2.)
                fdiv = 1. / sdiv
                gv = Sv * sdiv - Fv * fdiv
                chisq[xx, yy] = (gv ** 2 * w).sum()
                sigma[:, xx, yy] = np.sqrt(chisq[xx, yy] * np.diag((vv @ np.diag(1. / ww ** 2) @ np.transpose(vv))))

    return result, chisq, sigma
