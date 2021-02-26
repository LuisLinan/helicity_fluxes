import numpy as np
from binning import rebin
from expandimage import expand_image


def div_free_solution(mag, scale, mirror=0, no_ring=0):
    """Double precision version of the routine suggested by B. J. LaBonte. It calculates the solution
    a satisfying the gauge conditions of Chae (2001) by means of the fast Fourier transform

    Features

    --------

    Pads the original image with a mirror image of the original if mirror is set

    Programmers

    ----------

    B. J. LaBonte & M. K. Georgoulis (JHU/APL, 10/13/05)

    """

    res = mag.shape
    id1 = res[0]
    id2 = res[1]

    mag_ref = mag
    # Pad the image with zeroes# put image in the middle

    if no_ring == 0:
        for m in range(1, 16):
            if 2.0 ** m <= id1 < 2.0 ** (m + 1):
                enx = int(np.fix(2.0 ** float(m + 1)))
        for m in range(1, 16):
            if 2.0 ** m <= id2 < 2.0 ** (m + 1):
                eny = int(np.fix(2.0 ** (m + 1)))
        if mirror != 0:
            mag, stx, sty = expand_image(mag_ref, enx - id1, eny - id2, mirror=mirror)
        else:
            mag, stx, sty = expand_image(mag_ref, enx - id1, eny - id2)
    else:
        stx = 0
        sty = 0

    # Get the basic transform
    nx = mag[:, 0].size
    ny = mag[0, :].size
    ftm = np.fft.fft2(mag, norm='forward')

    # Multiply by i
    ftr = ftm * complex(0, -1)

    # Generate wavenumber images
    kx = np.array(np.arange(nx)).repeat(ny).reshape(nx, ny) - np.float(nx - 2) / 2.0
    kx = np.roll(kx, -int((nx - 2) / 2), axis=0)
    kx = -kx * 2. * np.pi / np.float(nx)
    ky = np.transpose(np.array(np.arange(ny)).repeat(nx).reshape(ny, nx)) - np.float(ny - 2) / 2.
    ky = np.roll(ky, -int((ny - 2) / 2), axis=1)
    ky = -ky * 2. * np.pi / np.float(ny)
    k2 = kx ** 2 + ky ** 2
    k2[0, 0] = 1.

    # Get vector potential from inverse transform
    apx = (np.fft.ifft2((ky / k2) * ftr, norm='forward')).real
    apy = (np.fft.ifft2((-kx / k2) * ftr, norm='forward')).real

    a = np.zeros((nx, ny, 2))
    a[:, :, 0] = apx
    a[:, :, 1] = apy
    a = a * scale
    if no_ring == 0:
        a = np.zeros((apx[int(stx):int(stx + id1), int(sty):int(sty + id2)].shape[0],
                      apx[int(stx):int(stx + id1), int(sty):int(sty + id2)].shape[1], 2))
        a[:, :, 0] = apx[int(stx):int(stx + id1), int(sty):int(sty + id2)]
        a[:, :, 1] = apy[int(stx):int(stx + id1), int(sty):int(sty + id2)]
        a = a * scale

    return a
