import numpy as np
from scipy import optimize
from davematrix import dave4vm_matrix


def dave4vm(mag, window_size, doublee=1, floatt=1, threshold=1.0, missing_value=np.nan):
    """ DAVE4VM - Differential Affine Velocity Estimator for Vector Magnetograms

    Parameters
    
    ---------
     
        MAG :
            structure of vector magnetic field measurements', 
        MAG.DX :
            X spatial scale (used to compute B?X), 
        MAG.DY :
            Y spatial scale (used to compute B?Y)                                                   ', 
        MAG.BZT :
            Array[NX,NY]  time derivative of Bz 
        MAG.BX  :
            Array[NX,NY]  X component of B (Bx) 
        MAG.BXX  :
            Array[NX,NY]  X derivative of Bx' 
        MAG.BXY 
            Array[NX,NY]  Y derivative of By' 
        MAG.BY    
            Array[NX,NY]  Y component of B (By)' 
        MAG.BYX   
            Array[NX,NY]  X derivative of By'
        MAG.BYY   
            Array[NX,NY]  Y derivative of By' 
        MAG.BZ
            Array[NX,NY]  Z component of B (Bz)' 
        MAG.BZX   
            Array[NX,NY]  X derivative of Bz' 
        MAG.BZY
            Array[NX,NY]  Y derivative of Bz' 
        WINDOW_SIZE :
            A one or two element vector for the window aperture', 
 
    Returns
    
    -------
           
        VEL :
            Array[NX,NY] of structures of coefficients'
        U0  :     
            X-Velocity '
        V0       
            Y-Velocity ' 
        W0  
            Z-Velocity ' 
        UX    
            Local X derivative of the X-Velocity'
        VX   
            Local X derivative of the Y-Velocity' 
        WX   
            Local X derivative of the Z-Velocity' 
        UY   
            Local Y derivative of the X-Velocity' 
        VY   
            Local Y derivative of the Y-Velocity' 
        WY   
            Local Y derivative of the Z-Velocity' 
        WINDOW_SIZE :
            Local window size'
            
    Message
    
    -------
    
        Important! Velocities must be orthogonalized to obtain plasma velocities ', perpendicular to the
        magnetic field!',

        Important : the version presented here is an adaptation of an IDL code developed by Dr. P. W. Schuck.
        The python version written by L. Linan cannot be used for professional purposes.
        
        AUTHORIZATION TO USE AND DISTRIBUTE (for the original IDL version )

        I hereby agree to the following terms governing the use and', redistribution of the DAVE4VM software release 
        originally', written and developed by Dr. P. W. Schuck', 

        Redistribution and use in source and binary forms, with or', without modification, are permitted provided 
        that (1) source', code distributions retain this paragraph in its entirety, (2), distributions including 
        binary code include this paragraph in its entirety in the documentation or other materials provided', 
        with the distribution, (3) improvements, additions and', upgrades to the software will be provided to NRL 
        Authors in', computer readable form, with an unlimited, royalty-free', license to use these improvements, 
        additions and upgrades', and the authority to grant unlimited royalty-free sublicenses, to these improvements 
        and (4) all published research using', this software display the following acknowledgment, This work uses
         the DAVE/DAVE4VM codes written and developed, by the Naval Research Laboratory.
         
         
        Neither the name of NRL or its contributors, nor any entity, of the United States Government may be used to
        endorse or, promote products derived from this software, nor does the inclusion of the NRL written and
        developed software directly or indirectly suggest NRL's or the United States Government's endorsement of this 
        product.
        
        THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS, OR IMPLIED WARRANTIES, INCLUDING, WITHOUT 
        LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.,

    Programmer

    ----------

    Peter W. Schuck

    Reference

    ---------

    Schuck, P. W., Tracking Vector Magnetograms with the Magnetic Induction Equation, Submitted to ApJ, 2008.


    """

    if doublee != 1:
        doublee = 0

    if floatt != 1:
        floatt = 0

    if floatt == doublee:
        if floatt == 0:
            if mag['BZ'].dtype == 'float32':
                floatt = 1
            else:
                floatt = 0
            if mag['BZ'].dtype == 'float64':
                doublee = 1
            else:
                doublee = 0

    if missing_value != np.nan:
        if doublee == 1:
            missing_value = np.float64("nan")
        else:
            missing_value = np.float32("nan")

    if doublee == 1:
        ddtype = 'float64'
    else:
        ddtype = 'float32'

    sz = mag['BZ'].shape

    vel = missing_value
    dum = np.empty((sz[0], sz[1]))

    dum[:] = vel

    v = {'U0': dum.copy(), 'UX': dum.copy(), 'UY': dum.copy(), 'V0': dum.copy(), 'VX': dum.copy(), 'VY': dum.copy(),
         'W0': dum.copy(), 'WX': dum.copy(), 'WY': dum.copy()}

    id = np.arange(9)

    nw = np.fix(2 * np.fix(window_size / 2) + 1)
    nw = [nw, nw]

    x = np.array(np.arange(int(nw[0])) - int(nw[0] / 2)).repeat(int(nw[1])).reshape(int(nw[0]), int(nw[1])) * mag['DX']
    y = np.transpose(
        np.array(np.arange(int(nw[1])) - int(nw[1] / 2)).repeat(int(nw[0])).reshape(int(nw[1]), int(nw[0]))) * mag['DY']

    psf = np.zeros((int(nw[0]), int(nw[1])), dtype=ddtype)

    psf[:] = 1.0

    # normalize
    psf[:] = psf / psf.sum()

    # moments
    psfx = psf * x
    psfy = psf * y
    psfxx = psf * x ** 2
    psfyy = psf * y ** 2
    psfxy = psf * x * y

    v['WINDOW_SIZE'] = nw

    AM = dave4vm_matrix(mag['BX'], mag['BXX'], mag['BY'], mag['BYY'], mag['BZ'], mag['BZX'], mag['BZY'], mag['BZT'],
                        psf,
                        psfx, psfy, psfxx, psfyy, psfxy, doublee)

    kernel = {'psf': psf.copy(), 'psfx': psfx.copy(), 'psfy': psfy.copy(), 'psfxx': psfxx.copy(), 'psfyy': psfyy.copy(),
              'psfxy': psfxy.copy()}

    # estimate trace

    trc = AM.reshape(100, sz[0], sz[1])[id * 10 + id, :, :].sum(axis=0)

    # find locations where the aperture problem could be resolved

    index = np.where(trc > threshold)
    N = len(index[0])

    # loop over good pixels

    for ii in range(N):
        j = index[1][ii]
        i = index[0][ii]
        AA = AM[:, :, i, j]
        GA = AA[0:9, 0:9]
        FA = -AA[0:9, 9]
        DESIGN = GA
        SOURCE = FA
        vector = optimize.lsq_linear(DESIGN, SOURCE)['x']

        v['U0'][i, j] = vector[0]
        v['V0'][i, j] = vector[1]
        v['UX'][i, j] = vector[2]
        v['VY'][i, j] = vector[3]
        v['UY'][i, j] = vector[4]
        v['VX'][i, j] = vector[5]
        v['W0'][i, j] = vector[6]
        v['WX'][i, j] = vector[7]
        v['WY'][i, j] = vector[8]

    return v, kernel, AM
