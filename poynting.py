import numpy as np


def poynting(lamda, Bz, Bx, By, Ux, Uy, Uz=0, polar=0):
    """PURPOSE: Calculate the Poynting flux given a vector magnetogram and the three components of the velocity field
       NOTE : Based on the analysis of Kusano et al. (2002)

    Parameters

    ---------

    lamda :
        pixel size of magnetogram (cm)'
    Bz :
        z-component of the magnetic field vector'
    Bx :
        x-component (cartesian system) or horizontal (polar system) magnetic field'
    By :
        y-component (cartesian system) or azimuth angle (polar system) of the field vector'
    Ux :
        x-component of the velocity field (cm/s)'
    Uy :
        y-component of the velocity field (cm/s)'

    Keywords arguments

    ------------------

    Uz :
        Set to the z-component of the velocity field (cm/s), if available'
    cartesian :
        Use if the magnetogram is in cartesian coordinates (Bz, Bx, By)'
    polar :
        Use if the magnetogram is in polar coordinates (Bz, Bh, phi)'

    Return

    -----

    shear_den :
        Set to a variable for the poynting flux density due to shear (erg/cm**2 s)'
    emerg_den :
        Set to a variable for the poynting flux density due to flux emergence (erg/cm**2 s)'
    tot_den :
        Set to a variable for the total poynting flux density (erg/cm**2 s)'
    tot_shear :
        Set to a variable for the total poynting flux due to shear (erg/s)'
    tot_emerg :
        Set to a variable for the total poynting flux due to flux emergence (erg/s)'
    tot_flux :
        Set to a variable for the total poynting flux (erg/s)'

"""

    if polar == 1:
        # change coordinate system if needed
        Bh = Bx
        phi = By
        Bx = Bh * np.cos(phi)
        By = Bh * np.sin(phi)

    # shear component of Poynting flux
    shear_den = -(1. / (4. * np.pi)) * (Bx * Ux + By * Uy) * Bz
    tot_shear = shear_den.sum() * lamda * lamda

    # emergence component of Poynting flux
    if type(Uz) == 0:
        print('NO VERTICAL VELOCITY DEFINED - ONLY THE SHEAR COMPONENT WILL BE CALCULATED')
        emerg_den = 0
        tot_emerg = 0
    else:
        emerg_den = (1. / (4. * np.pi)) * (Bx * Bx + By * By) * Uz
        tot_emerg = emerg_den.sum() * lamda * lamda

    # total Poynting flux
    tot_den = shear_den + emerg_den
    tot_flux = tot_emerg + tot_shear

    # return to original values

    return shear_den, emerg_den, tot_den, tot_shear, tot_emerg, tot_flux
