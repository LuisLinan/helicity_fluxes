import numpy as np
from divfree import div_free_solution


def helicity_rate_ko(Bx, By, Bz, Ux, Uy, Uz, lamda, fmask):
    """ Calculation ofthe rate of helicity variation based on the method of Chae (2001)

    Parameters

    ----------

    Bz :
        Vertical magnetic field (G)'
    Bh :
        Horizontal magnetic field (G)'
    phi :
        azimuth angle (radians)'
    Ux :
        x-component of the velocity field (cm/sec)'
    Uy :
        y-component of the velocity field (cm/sec)'
    Uz :
        vertical velocity field (cm/sec)'
    lamda :
        Pixel size of the magnetogram (cm)'
    fmask :
        Calculation mask (1 for strong fields, 0 for weak fields)'

    Keywords arguments

    ------------------

    corrector:
        if corrector=1 helicity rate calculation from Pariat et al. (2005)

    Returns

    ------

    Ga_inj :
        Density of helicity injection rate through emergence, Mx**2/cm**2/s'
    Ga_shu :
        Density of helicity injection rate through shuffling, Mx**2/cm**2/s'
    dHdt_inj :
        Total helicity injection rate through emergence, Mx**2 / s'
    dHdt_shu :
        Total helicity injection rate through shuffling, Mx**2 / s'
    dHdt_a : T
        Total helicity injection rate (Berger & Field 1984), Mx**2 / s'


    """

    Bh = np.sqrt(Bx ** 2 + By ** 2)

    res = Bz.shape
    id1 = res[0]
    id2 = res[1]
    r1 = np.where(np.abs(Bz) > 3500)
    r2 = np.where(Bh > 3500)
    ico = len(r1[0])
    if ico > 0:
        Bz[r1] = 0.
        Bh[r1] = 0.
    ico = len(r2[0])
    if ico > 0:
        Bz[r2] = 0.
        Bh[r2] = 0.

    # vector_potential,Bz,lamda,Ap Kostas's method
    Ap = div_free_solution(Bz, lamda)
    Apx = np.zeros((id1, id2))
    Apx[:, :] = Ap[:, :, 0]
    Apy = np.zeros((id1, id2))
    Apy[:, :] = Ap[:, :, 1]

    Ga_inj = 2. * (Bx * Apx + By * Apy) * Uz * fmask
    Ga_shu = -2. * (Ux * Apx + Uy * Apy) * Bz * fmask
    dHdt_inj = (Ga_inj * fmask).astype('float').sum() * lamda * lamda
    dHdt_shu = (Ga_shu * fmask).astype('float').sum() * lamda * lamda
    dHdt_a = dHdt_shu + dHdt_inj

    return Ga_inj, Ga_shu, dHdt_inj, dHdt_shu, dHdt_a
