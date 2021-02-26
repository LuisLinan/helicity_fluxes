from sunpy.net import jsoc
from sunpy.net import Fido, attrs as a
import sunpy.map
from sunpy.time import parse_time
import numpy as np
from helicity_flux import helicity_energy

__author__ = "Luis Linan"
__email__ = "luis.linan@obspm.fr"
___status__ = "Production"

def __main__():
    """launch the computation of relative helicity flux using the HMI magnetic field downloaded from jsoc
    """

    download=0
    if download==1:
        client = jsoc.JSOCClient()
        res = client.search(a.Time('2015-10-30T00:00:00', '2015-10-30T0:24:00'), a.jsoc.Series('hmi.sharp_cea_720s'),
                            a.jsoc.PrimeKey('HARPNUM', '6060'), a.jsoc.Notify('luis.linan@obspm.fr'),
                            a.jsoc.Segment('Bp'))

        requests = client.request_data(res)
        client.get_request(requests, path='./base/')

        res = client.search(a.Time('2015-10-30T00:00:00', '2015-10-30T0:24:00'), a.jsoc.Series('hmi.sharp_cea_720s'),
                            a.jsoc.PrimeKey('HARPNUM', '6060'), a.jsoc.Notify('luis.linan@obspm.fr'),
                            a.jsoc.Segment('Bt'))

        requests = client.request_data(res)
        client.get_request(requests, path='./base/')

        res = client.search(a.Time('2015-10-30T00:00:00', '2015-10-30T0:24:00'), a.jsoc.Series('hmi.sharp_cea_720s'),
                            a.jsoc.PrimeKey('HARPNUM', '6060'), a.jsoc.Notify('luis.linan@obspm.fr'),
                            a.jsoc.Segment('Br'))

        requests = client.request_data(res)
        client.get_request(requests, path='./base/')

    File1Br = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_000000_TAI.Br.fits')
    File1Bp = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_000000_TAI.Bp.fits')
    File1Bt = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_000000_TAI.Bt.fits')
    File2Br = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_001200_TAI.Br.fits')
    File2Bp = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_001200_TAI.Bp.fits')
    File2Bt = sunpy.map.Map('./base/hmi.sharp_cea_720s.6060.20151030_001200_TAI.Bt.fits')


    t1 = File1Br.meta['t_obs']
    t1 = parse_time(t1, format='utime').value
    londtmin=File1Br.meta['londtmin']
    londtmax=File1Br.meta['londtmax']
    idx_t0={'t_obs':t1,'londtmin':londtmin,'londtmax':londtmax}

    t2 = File2Br.meta['t_obs']
    t2 = parse_time(t2, format='utime').value
    londtmin=File2Br.meta['londtmin']
    londtmax=File2Br.meta['londtmax']
    idx_t1={'t_obs':t2,'londtmin':londtmin,'londtmax':londtmax}

    Br1 = np.array(File1Br.data).transpose()
    Br2 = np.array(File2Br.data).transpose()
    Bt1 = np.array(File1Bt.data).transpose()
    Bt2 = np.array(File2Bt.data).transpose()
    Bp1 = np.array(File1Bp.data).transpose()
    Bp2 = np.array(File2Bp.data).transpose()

    helicity_fluxes = helicity_energy(Br1, Bt1, Bp1, idx_t0, Br2, Bt2, Bp2, idx_t1)

if __name__=="__main__":
    __main__()
