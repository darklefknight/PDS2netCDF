


#------------------import modules-------------------------------
from pdsdata import pdsdata
import datetime as datetime
import numpy as np
import numpy.ma as ma
from netCDF4 import Dataset
import os
import time as Time
# import asyncio
import functools
from dask.distributed import Client
from joblib import Parallel,delayed



#--------------------Hildebrand-Sekhon algorythm----------------
def start_HS(i):
        for k in range(0, 2):
             for j in range(len(pds.SPC[0, :, k, 0])):
                 # maxSn = np.nanmax(spectrum)
                 # Sn0    = np.double(sorted(spectrum[:]), reverse=False)
                 # Sn     = np.fliplr([Sn0])
                 # Sn     = Sn[0,:]
                 Sn = np.double(sorted(pds.SPC[i,j,k]), reverse=True)  # ist das nicht das gleiche?

                 SnR2 = Sn[len(Sn) - 1]

                 for l in range(0, len(Sn)):
                     n = sum(np.isfinite(Sn[l:len(Sn)]))
                     P = np.nansum(Sn[l:len(Sn)]) / n
                     Q = np.nansum(Sn[l:len(Sn)] ** 2) / n - P ** 2
                     R2 = P ** 2 / (Q * avc)
                     if (R2 > 1):
                         SnR2 = Sn[l]
                         break

                     WhiteNoise_Thresh[i,j,k] = SnR2


#---------------------Create NetCDF File----------------------------
def createNetCDF(ZeitList_array, spectrum):
    print("Create new NETCDF file")
    MISSING_VALUE=-999

    #Create file
    f_out = Folder + Filename[0:14]+'nc'
    cdf = Dataset(f_out, 'w', format='NETCDF4')

    #Create global attributes
    cdf.location        = "The Barbados Cloud Observatory, Deebles Point, Barbados"
    cdf.instrument      = "MBR2 spectra, white noise removed by Hildebrand-Sekhon algorythm"
    cdf.converted_by    = "Marcus Klingebiel (marcus.klingebiel@mpimet.mpg.de) and Heike Konow (heike.konow@mpimet.mpg.de)"
    cdf.institution     = "Max Planck Institute for Meteorology, Hamburg"
    #cdf.created_with   = os.path.basename(__file__)+" with its last modification on "+ Time.ctime(os.path.getmtime(os.path.realpath(__file__)))
    cdf.creation_date   = Time.asctime()
    cdf.version         ="1.0.0"

    #Create dimensions
    time_dim    = cdf.createDimension('time', len(ZeitList_array))
    range_dim   = cdf.createDimension('range', len(spectrum[0,:,0,0]))
    channel_dim = cdf.createDimension('channel', len(spectrum[0,0,:,0]))
    ffft_dim    = cdf.createDimension('ffft', len(spectrum[0,0,0,:]))


    #Create variables
    time_var                    = cdf.createVariable('time','f8',('time',), fill_value=MISSING_VALUE)
    time_var.units              = "seconds since 1970-1-1 0:00:00 UTC"
    time_var._CoordinateAxisType= "Time"
    time_var.calendar           = "Standard"

    range_var                   = cdf.createVariable('range','f4',('range',), fill_value=MISSING_VALUE)
    range_var.units             = "m"
    range_var.axis              = "Y"

    channel_var                 = cdf.createVariable('channel','f4',('channel',), fill_value=MISSING_VALUE)
    channel_var.units           = ""
    channel_var.axis            = ""

    ffft_var                    = cdf.createVariable('ffft','f4',('ffft',), fill_value=MISSING_VALUE)
    ffft_var.units              = "m"
    ffft_var.axis               = "Y"

    spectra_var                     = cdf.createVariable('spectra','f4',('time','range','channel','ffft'), fill_value=MISSING_VALUE)
    spectra_var.units               = "a.u."
    spectra_var.long_name           = "spectra"

    #insert values
    time_var[:]         = ZeitList_array[:]
    range_var[:]        = np.arange(0,832)
    channel_var[:]      = [0,1]
    ffft_var[:]         = pds.parameter.v_field[:]
    spectra_var[:,:,:,:]= Spectra_HS[:,:,:,:]
    cdf.close()
    print(f_out)

if __name__ == "__main__":

    path = '/data/mpi/mpiaes/obs/m300517/160814_130355.pds'
    Folder = path[:-17]
    Filename = path[len(path) - 17:]

    Begin_time = "1529"  # "1529"
    End_time = "1539"  # "1531"
    minuteEnd_str = End_time[2:4]


    date_str = path[len(path) - 17:len(path) - 17 + 6]
    year_str = "20" + date_str[0:2]
    month_str = date_str[2:4]
    day_str = date_str[4:6]

    hourBeg_str = Begin_time[0:2]
    minuteBeg_str = Begin_time[2:4]
    hourEnd_str = End_time[0:2]

    #-------------------read pds file-------------------------------
    start_date = datetime.datetime(int(year_str),int(month_str),int(day_str),int(hourBeg_str),int(minuteBeg_str),0,0)
    end_date   = datetime.datetime(int(year_str),int(month_str),int(day_str),int(hourEnd_str),int(minuteEnd_str),0,0)

    print(start_date,end_date)

    pds=pdsdata(pdsfile=[path])
    pds.readDataSection(["SPC","SNR","Z"], start_time=start_date,stop_time=end_date,quiet=True)

    pds.SPC[:,:,:,:] = np.roll(pds.SPC[:,:,:,:],128, axis = 3)

    avc = pds.parameter.avc

    # --------------------Noise level estimation for whole pds file-------
    print("Remove white noise (Hildebrand-Sekhon)")
    WhiteNoise_Thresh = np.zeros((len(pds.SPC[:, 0, 0, 0]), len(pds.SPC[0, :, 0, 0]), len(pds.SPC[0, 0, :, 0])))

    Parallel(n_jobs=-1,verbose=1,backend="multiprocessing")(map(delayed(start_HS),range(len(pds.SPC[:, 0, 0, 0]))))

    # futures = client.map(delayed(start_HS),range(len(pds.SPC[:, 0, 0, 0])))
    # temp = client.gather(futures)

    Spectra_HS = pds.SPC[:, :, :, :]
    WhiteNoise_Thresh = np.repeat(WhiteNoise_Thresh[:, :, :, np.newaxis], len(pds.SPC[0, 0, 0, :]), axis=3)

    Spectra_HS[:, :, 0, :] = np.where(ma.less_equal(Spectra_HS[:, :, 0, :], WhiteNoise_Thresh[:, :, 0, :]), np.nan,
                                      Spectra_HS[:, :, 0, :])
    Spectra_HS[:, :, 1, :] = np.where(ma.less_equal(Spectra_HS[:, :, 1, :], WhiteNoise_Thresh[:, :, 1, :]), np.nan,
                                      Spectra_HS[:, :, 1, :])


    # ---------------------Adjust time parameter to epoch time-----------
    ZeitList = []
    for i in range(len(pds.time_data)):
        ZeitList.append((pds.time_data[i] - datetime.datetime(1970, 1, 1)).total_seconds())
        # ZeitList.append(int(pds.time_data[i].strftime('%Z')))

    ZeitList_array = np.asarray(ZeitList)

    createNetCDF(ZeitList_array,pds.SPC)