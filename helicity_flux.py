from binning import congrid, rebin
from helicity_rate import helicity_rate_ko
from poynting import poynting
from scipy import ndimage
from davemulti import dave_multi
from dodave import do_dave4vm
import numpy as np

__author__ = "Luis Linan"
__email__ = "luis.linan@obspm.fr"
___status__ = "Production"


def helicity_energy(br_t0, bt_t0, bp_t0, idx_t0, br_t1, bt_t1, bp_t1, idx_t1, binxy_helicity_energy=1, set_mpil=0,
                    binxy_mpil=1, extend=5.0, fwhm_align=10.0, fwhm_dave4vm=4, mpil_info=0, mpil_xy=0):
    """calculates magnetic helicity and energy injection rates through the photosphere of active regions

    Parameters

    ----------

    br_t0 :
        hmi_sharp_cea_720s Br image at t0 (note that t1 > t0)
    bt_t0 :
        hmi_sharp_cea_720s Bt image at t0
    bp_t0 :
        hmi_sharp_cea_720s Bp image at t0
    idx_t0 :
        hmi_sharp_cea_720s meta data at t0
    br_t1 :
        hmi_sharp_cea_720s Br image at t1
    bt_t1 :
        hmi_sharp_cea_720s Bt image at t1
    bp_t1 :
        hmi_sharp_cea_720s Bp image at t1
    idx_t1 :
        hmi_sharp_cea_720s meta data at t1

    Keyword arguments

    -----------------

    binxy_helicity_energy :
        number for xy spatial binning of the sharp magnetic field input data related to calculation of magnetic
        helicity and energy injection rates [in pixels] # default=1 (i.e., no binning)
    set_mpil :
        if set, calculate magnetic helicity and energy injection rates only near magnetic polarity inversion
        lines (MPILs) # default=0
    binxy_mpil :
        number for xy spatial binning of sharp magnetic field input data related to MPIL detection [pixels] # default=1
    extend :
        length of a square box surrounding each of MPIL pixels [Mm]. The box region will be considered for
        calculating magnetic helicity and energy injection rates therein# default=5
    fwhm_align :
        full width at half maximum for sharp image alignment [arcsecs] # default=10
    fwhm_dave4vm :
        full width at half maximum for flow field calculation [arcsecs] # default=4
    mpil_info :
        information on types of MPILs
    mpil_xy :
        x and y coordinates of MPILs [pixels]

    Returns

    -------

    pos_dhdt_in:
        postive dhdt_in [10**35 Mx**2/sec]
    abs_neg_dhdt_in:
        absolute of negative dhdt_in [10**35 Mx**2/sec]
    abs_tot_dhdt_in:
        absolute of total dhdt_in [10**35 Mx**2/sec]
    tot_uns_dhdt_in:
        total unsigned dhdt_in [10**35 Mx**2/sec]
    pos_dhdt_sh:
        postive dhdt_sh [10**35 Mx**2/sec]
    abs_neg_dhdt_sh:
        absolute of negative dhdt_sh [10**35 Mx**2/sec]
    abs_tot_dhdt_sh:
        absolute of total dhdt_sh [10**35 Mx**2/sec]
    tot_uns_dhdt_sh:
        total unsigned dhdt_sh [10**35 Mx**2/sec]
    abs_tot_dhdt:
        absolute of total dhdt [10**35 Mx**2/sec]
    abs_tot_dhdt_in_plus_sh:
        abs_tot_dhdt_in + abs_tot_dhdt_sh [10**35 Mx**2/sec]
    tot_uns_dhdt:
        total unsigned dhdt [10**35 Mx**2/sec]
    pos_dedt_in:
        postive dedt_in [10**24 erg/sec]
    abs_neg_dedt_in:
        absolute of negative dedt_in [10**24 erg/sec]
    abs_tot_dedt_in:
        absolute of total dedt_in [10**24 erg/sec]
    tot_uns_dedt_in:
        total unsigned dedt_in [10**24 erg/sec]
    pos_dedt_sh:
        postive dedt_sh [10**24 erg/sec]
    abs_neg_dedt_sh:
        absolute of negative dedt_sh [10**24 erg/sec]
    abs_tot_dedt_sh:
        absolute of total dedt_sh [10**24 erg/sec]
    tot_uns_dedt_sh:
        total unsigned dedt_sh [10**24 erg/sec]
    abs_tot_dedt :
        absolute of total dedt [10**24 erg/sec]
    abs_tot_dhdt_in_plus_sh:
        abs_tot_dedt_in + abs_tot_dedt_sh [10**24 erg/sec]
    tot_uns_dedt:
        total unsigned dedt [10**24 erg/sec]

    TO DO

    -------

    set_mpil=1

    """

    extendpix = np.fix(extend / 0.36442476)  # pixel
    delmm = 0.36442476 * binxy_helicity_energy

    if (br_t0[:, 0].size / binxy_mpil > 8) and (br_t0[0, :].size / binxy_mpil) > 8 and (
            br_t1[:, 0].size / binxy_mpil > 8) and (br_t1[0, :].size / binxy_mpil > 8):

        if np.fix(set_mpil) == 0:

            s0 = br_t0.shape
            newdims = (s0[0] / binxy_helicity_energy ** 2, s0[1] / binxy_helicity_energy ** 2)
            br_t0 = congrid(br_t0, newdims)
            br_t0 = rebin(br_t0, newdims[0], newdims[1])
            bt_t0 = congrid(-1. * bt_t0, newdims)
            bt_t0 = rebin(bt_t0, newdims[0], newdims[1])
            bp_t0 = congrid(bp_t0, newdims)
            bp_t0 = rebin(bp_t0, newdims[0], newdims[1])

            s1 = br_t1.shape
            newdims = (s1[0] / binxy_helicity_energy ** 2, s1[1] / binxy_helicity_energy ** 2)
            br_t1 = congrid(br_t1, newdims)
            br_t1 = rebin(br_t1, newdims[0], newdims[1])
            bt_t1 = congrid(-1. * bt_t1, newdims)
            bt_t1 = rebin(bt_t1, newdims[0], newdims[1])
            bp_t1 = congrid(bp_t1, newdims)
            bp_t1 = rebin(bp_t1, newdims[0], newdims[1])

            if idx_t0['londtmax'] < idx_t1['londtmin']:
                output_helicity_energy = {"pos_dhdt_in": 0., "abs_neg_dhdt_in": 0., "abs_tot_dhdt_in": 0.,
                                          "tot_uns_dhdt_in": 0., "pos_dhdt_sh": 0., "abs_neg_dhdt_sh": 0.,
                                          "abs_tot_dhdt_sh": 0., "tot_uns_dhdt_sh": 0., "abs_tot_dhdt": 0.,
                                          "abs_tot_dhdt_in_plus_sh": 0., "tot_uns_dhdt": 0., "pos_dedt_in": 0.,
                                          "abs_neg_dedt_in": 0., "abs_tot_dedt_in": 0., "tot_uns_dedt_in": 0.,
                                          "pos_dedt_sh": 0., "abs_neg_dedt_sh": 0., "abs_tot_dedt_sh": 0.,
                                          "tot_uns_dedt_sh": 0., "abs_tot_dedt": 0., "abs_tot_dedt_in_plus_sh": 0.,
                                          "tot_uns_dedt": 0.}
            else:
                if s0[0] <= s1[0]:
                    if s0[1] <= s1[1]:
                        bp_t0 = bp_t0[0:s0[0], 0:s0[1]]
                        bp_t1 = bp_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)),
                                int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                        bt_t0 = bt_t0[0:s0[0], 0:s0[1]]
                        bt_t1 = bt_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)),
                                int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                        br_t0 = br_t0[0:s0[0], 0:s0[1]]
                        br_t1 = br_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)),
                                int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                    else:
                        bp_t0 = bp_t0[0:s0[0], int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        bp_t1 = bp_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)), 0:s1[1]]
                        bt_t0 = bt_t0[0:s0[0], int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        bt_t1 = bt_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)), 0:s1[1]]
                        br_t0 = br_t0[0:s0[0], int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        br_t1 = br_t1[int((s1[0] - s0[0]) / 2):int(s1[0] - np.around((s1[0] - s0[0]) / 2.)), 0:s1[1]]
                else:
                    if s0[1] <= s1[1]:
                        bp_t0 = bp_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)), 0:s0[1]]
                        bp_t1 = bp_t1[0:s1[0], int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                        bt_t0 = bt_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)), 0:s0[1]]
                        bt_t1 = bt_t1[0:s1[0], int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                        br_t0 = br_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)), 0:s0[1]]
                        br_t1 = br_t1[0:s1[0], int((s1[1] - s0[1]) / 2):int(s1[1] - np.around((s1[1] - s0[1]) / 2.))]
                    else:
                        bp_t0 = bp_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)),
                                int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        bp_t1 = bp_t1[0:s1[0], 0:s1[1]]
                        bt_t0 = bt_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)),
                                int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        bt_t1 = bt_t1[0:s1[0], 0:s1[1]]
                        br_t0 = br_t0[int((s0[0] - s1[0]) / 2):int(s0[0] - np.around((s0[0] - s1[0]) / 2.)),
                                int((s0[1] - s1[1]) / 2):int(s0[1] - np.around((s0[1] - s1[1]) / 2.))]
                        br_t1 = br_t1[0:s1[0], 0:s1[1]]

                # alignment using a correlation method
                s = br_t0.shape

                sht, _, _ = dave_multi(fwhm_align / delmm / 0.725,
                                       ndimage.filters.uniform_filter(br_t0, size=np.around(5. / delmm)),
                                       ndimage.filters.uniform_filter(br_t1, size=np.around(5. / delmm)))

                xsht = sht[0, :, :].reshape(s[0], s[1])
                xshtv = np.mean(xsht[np.where(np.abs(br_t0) > 500)])
                ysht = sht[1, :, :].reshape(s[0], s[1])
                yshtv = np.mean(ysht[np.where(np.abs(br_t0) > 500)])
                br_t1 = np.roll(br_t1, [-1 * np.fix(np.around(xshtv)), -1 * np.fix(np.around(yshtv))], axis=(0, 1))
                bp_t1 = np.roll(bp_t1, [-1 * np.fix(np.around(xshtv)), -1 * np.fix(np.around(yshtv))], axis=(0, 1))
                bt_t1 = np.roll(bt_t1, [-1 * np.fix(np.around(xshtv)), -1 * np.fix(np.around(yshtv))], axis=(0, 1))

                # cut the "co-alinged" sub - regions from the two images

                if np.fix(np.around(xshtv)) >= 0:
                    if np.fix(np.around(yshtv)) >= 0:
                        bp_t0 = bp_t0[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                        bp_t1 = bp_t1[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                        bt_t0 = bt_t0[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                        bt_t1 = bt_t1[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                        br_t0 = br_t0[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                        br_t1 = br_t1[0:int(s[0] - np.fix(np.around(xshtv))), 0:int(s[1] - np.fix(np.around(yshtv)))]
                    else:
                        bp_t0 = bp_t0[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[2]]
                        bp_t1 = bp_t1[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[1]]
                        bt_t0 = bt_t0[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[1]]
                        bt_t1 = bt_t1[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[1]]
                        br_t0 = br_t0[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[1]]
                        br_t1 = br_t1[0:int(s[0] - np.fix(np.around(xshtv))),
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)):s[1]]
                else:
                    if np.fix(np.around(yshtv)) >= 0:
                        bp_t0 = bp_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                        bp_t1 = bp_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                        bt_t0 = bt_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                        bt_t1 = bt_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                        br_t0 = br_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                        br_t1 = br_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                0:int(s[1] - np.fix(np.around(yshtv)))]
                    else:
                        bp_t0 = bp_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]
                        bp_t1 = bp_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]
                        bt_t0 = bt_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]
                        bt_t1 = bt_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]
                        br_t0 = br_t0[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]
                        br_t1 = br_t1[int(np.max(-1 * np.fix(np.around(xshtv)), 0)):s[0],
                                int(np.max((-1 * np.fix(np.around(yshtv))), 0)): s[1]]

                br_t0[np.where(np.isfinite(br_t0) == False)] = 0.
                bp_t0[np.where(np.isfinite(bp_t0) == False)] = 0.
                bt_t0[np.where(np.isfinite(bt_t0) == False)] = 0.
                br_t1[np.where(np.isfinite(br_t1) == False)] = 0.
                bp_t1[np.where(np.isfinite(bp_t1) == False)] = 0.
                bt_t1[np.where(np.isfinite(bt_t1) == False)] = 0.

                s = br_t0.shape
                delt = idx_t1['t_obs'] - idx_t0['t_obs']

                # calculate flow field

                windowsize = fwhm_dave4vm / delmm / 0.725

                dx = delmm * 1e3
                dy = dx

                t_start = 0
                t_stop = delt * 1.0

                magvm, vel4vm = do_dave4vm(bp_t0, bt_t0, br_t0, t_start, bp_t1, bt_t1, br_t1, t_stop, dx, dy,
                                           windowsize=windowsize)

                bx = magvm['BX']
                by = magvm['BY']
                bz = magvm['BZ']

                if len(vel4vm.keys()) != 0:
                    ux0 = vel4vm['U0']
                    uy0 = vel4vm['V0']
                    uz0 = vel4vm['W0']

                    B = np.sqrt(bx * bx + by * by + bz * bz)
                    qx = bx / B
                    qy = by / B
                    qz = bz / B

                    vB = qx * ux0 + qy * uy0 + qz * uz0
                    ux = ux0 - vB * qx
                    uy = uy0 - vB * qy
                    uz = uz0 - vB * qz

                    # calculate helicity and energy injection rates

                    nx = s[0]
                    ny = s[1]

                    ux = ux * 1e5
                    uy = uy * 1e5
                    uz = uz * 1e5

                    px = delmm * 1e8

                    m = np.where((np.isfinite(ux) == True) & (np.isfinite(uy) == True) & (np.isfinite(uz) == True))
                    fmask = np.zeros((nx, ny))
                    fmask[m] = 1.0
                    m2 = np.where((np.isfinite(ux) != True) & (np.isfinite(uy) != True) & (np.isfinite(uz) != True))

                    ux[m2] = 0

                    uy[m2] = 0

                    uz[m2] = 0

                    ux[0:11, :] = 0

                    ux[nx - 11:, :] = 0

                    ux[:, 0:11] = 0

                    ux[:, ny - 11:] = 0

                    uy[0:11, :] = 0

                    uy[nx - 11:, :] = 0

                    uy[:, 0:11] = 0

                    uy[:, ny - 11:] = 0

                    uz[0:11, :] = 0

                    uz[nx - 11:, :] = 0

                    uz[:, 0:11] = 0

                    uz[:, ny - 11:] = 0

                    ga_inj, ga_shu, dhdt_inj, dhdt_shu, dhdt_a = helicity_rate_ko(bx, by, bz, ux, uy, uz, px, fmask)

                    pos_dhdt_in = np.sum((ga_inj * fmask * px * px / (1e35))[np.where(ga_inj > 0)])
                    neg_dhdt_in = np.sum((ga_inj * fmask * px * px / (10e35))[np.where(ga_inj < 0)])
                    abs_neg_dhdt_in = np.abs(np.sum((ga_inj * fmask * px * px / (10e35))[np.where(ga_inj < 0)]))
                    abs_tot_dhdt_in = np.abs(pos_dhdt_in + neg_dhdt_in)
                    tot_uns_dhdt_in = pos_dhdt_in + abs_neg_dhdt_in

                    pos_dhdt_sh = np.sum((ga_shu * fmask * px * px / (1e35))[np.where(ga_shu > 0)])
                    neg_dhdt_sh = np.sum((ga_shu * fmask * px * px / (10e35))[np.where(ga_shu < 0)])
                    abs_neg_dhdt_sh = np.abs(np.sum((ga_shu * fmask * px * px / (10e35))[np.where(ga_shu < 0)]))
                    abs_tot_dhdt_sh = np.abs(pos_dhdt_sh + neg_dhdt_sh)
                    tot_uns_dhdt_sh = pos_dhdt_sh + abs_neg_dhdt_sh

                    abs_tot_dhdt = np.abs(pos_dhdt_in + neg_dhdt_in + pos_dhdt_sh + neg_dhdt_sh)
                    abs_tot_dhdt_in_plus_sh = abs_tot_dhdt_in + abs_tot_dhdt_sh
                    tot_uns_dhdt = pos_dhdt_in + abs_neg_dhdt_in + pos_dhdt_sh + abs_neg_dhdt_sh

                    shear_den, emerg_den, tot_den, tot_shear, tot_emerg, tot_flux = poynting(px, bz, bx, by, ux, uy,
                                                                                             Uz=uz)

                    pos_dedt_in = np.sum((emerg_den * fmask * px * px / (1e24))[np.where(emerg_den > 0)])
                    neg_dedt_in = np.sum((emerg_den * fmask * px * px / (1e24))[np.where(emerg_den < 0)])
                    abs_neg_dedt_in = np.abs(np.sum((emerg_den * fmask * px * px / (1e24))[np.where(emerg_den < 0)]))
                    abs_tot_dedt_in = np.abs(pos_dedt_in + neg_dedt_in)
                    tot_uns_dedt_in = pos_dedt_in + abs_neg_dedt_in

                    pos_dedt_sh = np.sum((shear_den * fmask * px * px / (1e24))[np.where(shear_den > 0)])
                    neg_dedt_sh = np.sum((shear_den * fmask * px * px / (1e24))[np.where(shear_den < 0)])
                    abs_neg_dedt_sh = np.abs(np.sum((shear_den * fmask * px * px / (1e24))[np.where(shear_den < 0)]))
                    abs_tot_dedt_sh = np.abs(pos_dedt_sh + neg_dedt_sh)
                    tot_uns_dedt_sh = pos_dedt_sh + abs_neg_dedt_sh

                    abs_tot_dedt = np.abs(pos_dedt_in + neg_dedt_in + pos_dedt_sh + neg_dedt_sh)
                    abs_tot_dedt_in_plus_sh = abs_tot_dedt_in + abs_tot_dedt_sh
                    tot_uns_dedt = pos_dedt_in + abs_neg_dedt_in + pos_dedt_sh + abs_neg_dedt_sh

                    output_helicity_energy = {"pos_dhdt_in": pos_dhdt_in, "abs_neg_dhdt_in": abs_neg_dhdt_in,
                                              "abs_tot_dhdt_in": abs_tot_dhdt_in, "tot_uns_dhdt_in": tot_uns_dhdt_in,
                                              "pos_dhdt_sh": pos_dhdt_sh, "abs_neg_dhdt_sh": abs_neg_dhdt_sh,
                                              "abs_tot_dhdt_sh": abs_tot_dhdt_sh, "tot_uns_dhdt_sh": tot_uns_dhdt_sh,
                                              "abs_tot_dhdt": abs_tot_dhdt,
                                              "abs_tot_dhdt_in_plus_sh": abs_tot_dhdt_in_plus_sh,
                                              "tot_uns_dhdt": tot_uns_dhdt, "pos_dedt_in": pos_dedt_in,
                                              "abs_neg_dedt_in": abs_neg_dedt_in, "abs_tot_dedt_in": abs_tot_dedt_in,
                                              "tot_uns_dedt_in": tot_uns_dedt_in, "pos_dedt_sh": pos_dedt_sh,
                                              "abs_neg_dedt_sh": abs_neg_dedt_sh, "abs_tot_dedt_sh": abs_tot_dedt_sh,
                                              "tot_uns_dedt_sh": tot_uns_dedt_sh, "abs_tot_dedt": abs_tot_dedt,
                                              "abs_tot_dedt_in_plus_sh": abs_tot_dedt_in_plus_sh,
                                              "tot_uns_dedt": tot_uns_dedt}
                else:
                    output_helicity_energy = {"pos_dhdt_in": 0, "abs_neg_dhdt_in": 0., "abs_tot_dhdt_in": 0,
                                              "tot_uns_dhdt_in": 0., "pos_dhdt_sh": 0., "abs_neg_dhdt_sh": 0.,
                                              "abs_tot_dhdt_sh": 0, "tot_uns_dhdt_sh": 0., "abs_tot_dhdt": 0.,
                                              "abs_tot_dhdt_in_plus_sh": 0., "tot_uns_dhdt": 0., "pos_dedt_in": 0.,
                                              "abs_neg_dedt_in": 0., "abs_tot_dedt_in": 0., "tot_uns_dedt_in": 0.,
                                              "pos_dedt_sh": 0., "abs_neg_dedt_sh": 0., "abs_tot_dedt_sh": 0.,
                                              "tot_uns_dedt_sh": 0., "abs_tot_dedt": 0., "abs_tot_dedt_in_plus_sh": 0.,
                                              "tot_uns_dedt": 0.}
        else:
            print('todo')
    else:
        output_helicity_energy = {"pos_dhdt_in": 0., "abs_neg_dhdt_in": 0., "abs_tot_dhdt_in": 0.,
                                  "tot_uns_dhdt_in": 0.,
                                  "pos_dhdt_sh": 0., "abs_neg_dhdt_sh": 0., "abs_tot_dhdt_sh": 0.,
                                  "tot_uns_dhdt_sh": 0.,
                                  "abs_tot_dhdt": 0., "abs_tot_dhdt_in_plus_sh": 0., "tot_uns_dhdt": 0.,
                                  "pos_dedt_in": 0., "abs_neg_dedt_in": 0., "abs_tot_dedt_in": 0.,
                                  "tot_uns_dedt_in": 0., "pos_dedt_sh": 0., "abs_neg_dedt_sh": 0.,
                                  "abs_tot_dedt_sh": 0.,
                                  "tot_uns_dedt_sh": 0., "abs_tot_dedt": 0., "abs_tot_dedt_in_plus_sh": 0.,
                                  "tot_uns_dedt": 0.}

    return output_helicity_energy
