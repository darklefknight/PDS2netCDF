"""  Read/Process PDS doppler data from METEK MIRA-36 radars

########################################################################

   This file is part of the pyCARE package.

   Copyright (C) 2016 Florian Ewald

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.

########################################################################
 
 Send comments to:
 Florian Ewald, florian.ewald@dlr.de

 Modification History:
 -------
   2010-01 Original python code by Kersten Schmidt (kersten.Schmidt@dlr.de)
   2013-01 Adopted and added IQ reader by Florian Ewald (florian.ewald@dlr.de)
""" 


import os.path
from struct import unpack,calcsize
from scipy.optimize import curve_fit
import datetime
import bisect
import numpy as np
import math

c_const = 299792458.
freq = 35e9

rxspc_normalization=2.69272e+09


# Define model function to be used to fit to the data above:
def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def linear(a):
    return 10.**(0.1*a)

class radar_param():
    def __init__(self):
        # Reference value of the average transmitter power at the receiver output (The internal thermistor is calibrated so that it shows the Power at the receiver output. At the transmitter output the power is usually 1.4 dB greater). [W]
        self.PTAV0 = 30.
        # PRF
        self.prf = 5000.
        # Pulse width = 1/Bandwidth [s]
        self.tau0 = 200e-9
        # Average mean tx pulse power [W]
        self.tpow = 30.
        # Vertical resolution
        self.delta_h = 0.5 * c_const * self.tau0
        # Height for first gate
        self.mom_h_min = 30.
        # Reference Pulse repetition frequency [Hz]
        self.PRF0 = 5000.
        # Reference Height [m] 
        self.H0 = 5000.
        # Wavelength [m]
        self.wvl = 0.00845
        # Speed of light [m/s] 
        self.clight=299792458.
        # Antenna Gain
        self.AGain=linear(49.75)  
        self.ABeamWidth = 0.5 * np.pi/180.
        # Reference Value of the total noise figure at the receiver input (including losses in the TR-switch and Couplers).
        self.NfRec0 = linear(6.7) 
        # value of the receiver noise aktually measured by the noisecom source 
        # calida.nf[Co/Cx] =
        # single path waveguide length including the inside the feed
        self.WGlength = 0.65
        # Antenna Diameter used for the near field correction, if < 0.1 no near field correction is applied
        self.AntennaDiameter4NearFieldCorr = 1.0 
        # [J]
        self.KBT0 = 4e-21 
        self.kwq = 0.93
        # der sollte auf 1.8 dB gesetzt werden
        self.TxRecbandwidthMatchingLoss = 1.

        
def calc_radarconst(zrg):

    pp = radar_param()
    
    # Losses in the waveguides, both ways (receiving and transmitting, assuming 0.6 dB/m loss). 
    LossWG = linear(0.6 * pp.WGlength*2)  
            
    C0 = (1e18 * 1024.0 * np.log(2.0) * pp.wvl**2 * pp.KBT0 * LossWG * pp.TxRecbandwidthMatchingLoss * pp.NfRec0 * pp.PRF0 * pp.H0**2)/(np.pi**3 * pp.clight * pp.kwq * pp.AGain**2 * pp.ABeamWidth**2 * pp.PTAV0 * pp.tau0)
    
    refvals = pp.PTAV0*pp.tau0/pp.PRF0

    H = np.arange(zrg) * pp.delta_h + pp.mom_h_min
    Hq7H0q = (H/pp.H0)**2
            
    RadarConstant5 = ( C0 * pp.prf * refvals / (pp.tpow * pp.tau0) ) 
    RadarConstant  =  RadarConstant5 * Hq7H0q
         
    return RadarConstant
    
    
def estimate_noise_metek(a, n_spcave=1, method=1, ndiv=16, ofdiv=9):

    if method==1:
        ixs=np.argsort(a)
        n = len(ixs)
        sum_a=a*0
        sum_aa=a*0
        sa=0.
        saa=0.
    
        try:
            for i in range(n):
                xa=a[ixs[i]]
                sa = sa + xa
                sum_a[i]=sa
                saa = saa + xa*xa
                sum_aa[i]=saa
            ns=np.arange(n)+1
            aveqs = (sum_a/ns)**2
            varqs = (sum_aa-aveqs*ns) / ((ns-1)>1)
    
            ix_spreu=n-1
            while (ix_spreu>1):
                if varqs[ix_spreu] <= aveqs[ix_spreu]/np.sqrt(n_spcave): break
                ix_spreu = ix_spreu - 1

            noise = sum_a[ix_spreu]/float(ix_spreu)
            ix_signal = ixs[ix_spreu:n-1]
        except IndexError:
            print((a, ixs))
            
    elif method==3:

        n=len(a)
        zdiv=int(n/ndiv)
        adiv=np.zeros(ndiv)
        igdiv=np.arange(zdiv)
        edzdiv = 1./float(zdiv)
        for idiv in range(ndiv):
            adiv[idiv] = edzdiv * sum(a[idiv*zdiv+igdiv])

        mindiv=min(adiv)
        thrdiv = mindiv * (1+ofdiv/np.sqrt(n_spcave*zdiv))
        ixdiv = np.where(adiv <= thrdiv)[0]
        nixdiv = len(ixdiv)
        if nixdiv>0:
            noise = sum(adiv[ixdiv])/float(nixdiv)
        else:
            noise = 0.
        ix_signal_div = np.where(adiv > thrdiv)
        nixs = len(ix_signal_div)
        if nixs >= 1:
            ix_signal = np.arange(nixs*zdiv)
            for i in range(nixs-1):
                ix_signal[i*zdiv] = ix_signal_div[i]*zdiv + igdiv
        else: ix_signal=-1        
        
    return noise, ix_signal
    
def estimate_noise_hs74(spectrum, navg=1):
    """
    Estimate noise parameters of a Doppler spectrum.
    Use the method of estimating the noise level in Doppler spectra outlined
    by Hildebrand and Sehkon, 1974.
    Parameters
    ----------
    spectrum : array like
        Doppler spectrum in linear units.
    navg : int, optional
        The number of spectral bins over which a moving average has been
        taken. Corresponds to the **p** variable from equation 9 of the
        article.  The default value of 1 is appropiate when no moving
        average has been applied to the spectrum.
    Returns
    -------
    mean : float-like
        Mean of points in the spectrum identified as noise.
    threshold : float-like
        Threshold separating noise from signal.  The point in the spectrum with
        this value or below should be considered as noise, above this value
        signal. It is possible that all points in the spectrum are identified
        as noise.  If a peak is required for moment calculation then the point
        with this value should be considered as signal.
    var : float-like
        Variance of the points in the spectrum identified as noise.
    nnoise : int
        Number of noise points in the spectrum.
    References
    ----------
    P. H. Hildebrand and R. S. Sekhon, Objective Determination of the Noise
    Level in Doppler Spectra. Journal of Applied Meteorology, 1974, 13,
    808-811.
    """
    sorted_spectrum = np.sort(spectrum)
    nnoise = len(spectrum)  # default to all points in the spectrum as noise
    for npts in range(1, len(sorted_spectrum)+1):
        partial = sorted_spectrum[:npts]
        mean = partial.mean()
        var = partial.var()
        if var * navg < mean**2.:
            nnoise = npts
        else:
            # partial spectrum no longer has characteristics of white noise
            break

    noise_spectrum = sorted_spectrum[:nnoise]
    mean = noise_spectrum.mean()
    threshold = sorted_spectrum[nnoise-1]
    var = noise_spectrum.var()
    return mean, threshold, var, nnoise
    
class ppar:
    def __init__(self, byte_array, magic=None):
        ppar_structure = []
        for i in range(0,9):
            ppar_structure.append(unpack('=l',byte_array[i*4:i*4+4])[0])
        for i in range(9,11):
            ppar_structure.append(unpack('=f',byte_array[i*4:i*4+4])[0])
        for i in range(11,27):
            ppar_structure.append(unpack('=l',byte_array[i*4:i*4+4])[0])
        for i in range(27,31):
            ppar_structure.append(unpack('=f',byte_array[i*4:i*4+4])[0])
        for i in range(31,35):
            ppar_structure.append(unpack('=l',byte_array[i*4:i*4+4])[0])
        print(ppar_structure)
        c = 2.998e8
        #c = 3e8
        xmt = 35.5e9
        if ( ppar_structure[0] < 10 ):
            self.prf = 2500. * (ppar_structure[0] + 1)
        else:
            self.prf = ppar_structure[0]
        self.Tau = 100. * (ppar_structure[1] + 1)  #* 1e-9 pulse width in ns
        #self.delta_h = 0.5 * c * self.Tau *1e-9    # IDL definition
        self.dH      =  15*self.Tau/100           # showradar perl definition
        #print "delta h: %f %f" %(self.delta_h,self.dH)
        if magic == ('HALO' or 'METL'):
            self.radar_type = 'OLD'
            self.SPC_dtype  = 'long'
        elif magic[2:] == 'XC':
            self.radar_type = 'XCR'
            self.SPC_dtype  = 'float'
        else:
            self.radar_type = 'MBR'
            self.SPC_dtype  = 'float'
        if ppar_structure[2]>=64:
          self.nfft = int(ppar_structure[2])
        else:
          self.nfft = 128 * 2**ppar_structure[2]
        self.avc = ppar_structure[3]
        self.dT = float(self.avc) * float(self.nfft) / self.prf   # ave in IDL
        self.HMin = ppar_structure[4] * self.dH                # mom_h_min
        self.HMax = self.HMin + ppar_structure[5] * self.dH    # mom_zrg ?
        self.chg  = ppar_structure[5] -2                         # count of valid height gates
        self.pol = ppar_structure[6]
        self.att = ppar_structure[7]
        self.tx  = ppar_structure[8]
        self.nspec = ppar_structure[32] - ppar_structure[31]+1 +2  # raw_zrg
        self.osc = ppar_structure[21]
        self.vuar  = 0.5*c*self.prf/xmt # Factor for calculating physical from normalized velocities
                                        # diameter of the range of velocities which can be allocated without ambiguity
                                        # velocity range: -vuar/2 ... + vuar/2 
        self.v_field = np.arange(-self.vuar*0.5+self.vuar/self.nfft, \
                                  self.vuar*0.5+self.vuar/self.nfft, \
                                   self.vuar/self.nfft)
        self.ppar = ppar_structure


    def __str__(self):
        return "ppar prf: %d \n valid height gates %d " % (self.prf,self.chg)

class srvi:

    def __init__(self,byte_array):

        self.values = {}
        self.elements =[("FrmCnt","l"),("Tm","l"),("TPow","f"),("NPw1","f"),("NPw2","f"),("CPw1","f"),("CPw2","f"),("PS_Stat","l"),("RC_Err","l"),("TR_Err","l"),("GRS1","l"),("GRS2","l"),("AzmPos","f"),("AzmVel","f"),("ElvPos","f"),("ElvVel","f"),("NorthAngle","f"),("AzmSetPos","f"),("AzmSetVel","f"),("ElvSetPos","f"),("ElvSetVel","f")]
        # fill dictionary with unpacked values from byte array
        for i in range(0,len(self.elements)):
            self.values["%s"%self.elements[i][0]] = unpack('=%s'%self.elements[i][1],byte_array[i*4:i*4+4])[0]


    def __str__(self):

        d = datetime.datetime(1970,1,1)+datetime.timedelta(seconds=self.values["Tm"])
        content = "%s " % d.strftime("CDR %Y-%m-%d %H:%M:%S UTC")
        content += "FrmCnt=%d " %(self.values["FrmCnt"])
        content += "TPow=%0.1f " %(self.values["TPow"])
        content += "Npw1=%0.1f NPw2=%0.1f " %(10.*math.log10(self.values["NPw1"]),10.*math.log10(self.values["NPw2"]))
        content += "CPw1=%0.1f CPw2=%0.1f " %(10.*math.log10(self.values["CPw1"]),10.*math.log10(self.values["CPw2"]))
        content += "AzmPos=%0.1f AzmVel=%0.1f " %(self.values["AzmPos"],self.values["AzmVel"])
        content += "ElvPos=%0.1f ElvVel=%0.1f " %(self.values["ElvPos"],self.values["ElvVel"])
        content += "AzmSetPos=%0.1f AzmSetVel=%0.1f" %(self.values["AzmSetPos"],self.values["AzmSetVel"])
        #        print " ElvSetPos=$ElvSetPos ElvSetVel=$ElvSetVel";
        #        printf " NorthAngle=%0.1f",$NorthAngle;
        #        printf " GRST=%4.4X%4.4X",$GRS1,$GRS2;
        #        printf " PS_Stat=%4.4X",$PS_Stat;
        #        printf " RC_Err=%4.4X",$RC_Err;
        #        printf " TR_Err=%4.4X",$TR_Err;
        return content


class pdsdata:

    def __init__(self, pdsfile=None):

        self.start_time = 0
        self.stop_time  = 0
        self.time_index = 0

        self.time_list = []  # list of timestamps
        self.ptr_list = []   # list of file positions corresponding to time list
        self.fh_list = []    # lif of file handles

        self.time_data = []  # list of read timestamps
        
        self.param  = []
        self.srvi   = []
        self.srvi_values = []

        self.SNRco  = []
        self.VELco  = []
        self.RMSco  = []
        self.SNRcx  = []
        self.VELcx  = []
        self.RMScx  = []
        self.EXPco  = []
        # self.SPC  = []
        self.HNE  = []
        self.SPC  = []
        self.IQ   = []

        if pdsfile != None:
            if isinstance(pdsfile,str):
                f = pdsfile
                print((' .. opening: %s'%f))
                self.openpdsfile(f)
                self.create_indextable()
            else:                
                for f in pdsfile:
                    print((' .. opening: %s'%f))
                    self.openpdsfile(f)
                    self.create_indextable()


    def openpdsfile(self, pdsfile):

        self.filesize = os.path.getsize(pdsfile)
        self.fh = open(pdsfile,'rb')
        name  = "%s" % self.fh.read(32).rstrip(b'\x00').replace(b'\n',b'')
        time  = "%s" % self.fh.read(32).rstrip(b'\x00').replace(b'\n',b'')
        oper  = "%s" % self.fh.read(64).rstrip(b'\x00').replace(b'\n',b'')
        place = "%s" % self.fh.read(128).rstrip(b'\x00').replace(b'\n',b'')
        desc  = "%s" % self.fh.read(256).rstrip(b'\x00').replace(b'\n',b'')
        self.fh.seek(512, 1)
        magic = self.fh.read(4)
        self.fh.seek(-4-512, 1)
        self.header = { "name":name, "time":time, "oper":oper, "place":place, "desc":desc, "magic":magic }
        self.parameter = ppar(self.fh.read(512), magic=magic)

    def printheader(self):

        print(self.header)
        print("file: ",self.header["name"],\
        ", time: ",self.header["time"],\
        ", operation:   ",self.header["oper"],\
        ", place:       ",self.header["place"],\
        ", description: ",self.header["desc"])

    def print_parameter(self):
        print(self.parameter)


    def getSignature(self):

        byte_structure = self.fh.read(8)
        if len(byte_structure) == 0:
            return ('',0)
        sig = byte_structure[0:4]
        sigSize = unpack('=l',byte_structure[4:8])[0]
        #print sig, sigSize
        return (sig, sigSize)


    def getByteSignature(self,byteStructure,byteCount):

        sig = byteStructure[byteCount:byteCount+4].decode()
        # print(sig)
        sigSize = unpack('=l',byteStructure[byteCount+4:byteCount+8])[0]
        return (sig, sigSize)


    def evaluateByteStructure(self, byteStructure, product_list=[]):

        byteCount = 0
        procFFT = 0
        procHNE = 0

        # new order index list: [127,..0,255,..,128]
        idx_fft = [i for i in list(range(int(self.parameter.nfft/2)-1,-1,-1))\
        +list(range(self.parameter.nfft-1,int(self.parameter.nfft/2)-1,-1))]
        
        expbuf = np.zeros((self.parameter.nfft,self.parameter.chg))
        
        while (byteCount < len(byteStructure)):
                        
            subsig, subsigSize = self.getByteSignature(byteStructure,byteCount)
            byteCount += 8 # count header size from sub structure

            # print(subsig,subsigSize)


            if subsig == "PPAR":
                self.parameter = ppar(byteStructure[byteCount:byteCount+subsigSize])
                self.param.append(self.parameter)
            elif subsig == "SRVI":
                # print(" in SRVI right now!")
                self.srvi_values = srvi(byteStructure[byteCount:byteCount+subsigSize])
                self.srvi.append(self.srvi_values)
            if subsig[0:3] == "SNR" and ("SNR" or "Z" in product_list):
                cocx = self.getCoCxData(byteStructure[byteCount:byteCount+self.parameter.chg*8])
                self.SNRco.append(cocx[:,0])
                self.SNRcx.append(cocx[:,1])
            elif subsig[0:3] == "VEL" and ("VEL" in product_list):
                cocx = self.getCoCxData(byteStructure[byteCount:byteCount+self.parameter.chg*8])
                self.VELco.append(cocx[:,0])
                self.VELcx.append(cocx[:,1])
            elif subsig[0:3] == "RMS" and ("RMS" in product_list):
                cocx = self.getCoCxData(byteStructure[byteCount:byteCount+self.parameter.chg*8])
                self.RMSco.append(cocx[:,0])
                self.RMScx.append(cocx[:,1])
            elif subsig[0:3] == "EXP" and ("SPC" in product_list):
                # caution: field will be cutted! [512,2] -> [488,2]
                #print("EXP sig: ",subsigSize)
                expbuf = self.getCoCxExp(byteStructure[byteCount:byteCount+subsigSize])[0:self.parameter.chg,:]
                self.EXPco.append(expbuf)
            elif subsig[0:3] == "FFT" and ("SPC" in product_list):
                #print subsigSize, self.parameter.chg, self.parameter.nfft

                # spectra written by the new (since FZK) integer DSP
                if self.parameter.osc == 0:

                    (x,c) = self.getCoCxFFT(byteStructure[byteCount:byteCount+subsigSize], dtype=self.parameter.SPC_dtype).shape
                    self.parameter.chg = int (x / self.parameter.nfft)
                    specbuf =\
                    self.getCoCxFFT(byteStructure[byteCount:byteCount+subsigSize], dtype=self.parameter.SPC_dtype)\
                    .flatten()\
                    .reshape(self.parameter.chg,2,self.parameter.nfft)
                    procFFT=1

                # RAW IQ data written by all DSP
                if self.parameter.osc == 1:

                    iqbuf = self.getCoCxIQ(byteStructure[byteCount:byteCount+subsigSize])
                    self.IQ.append(iqbuf.reshape(self.parameter.chg+2,2,256,2))
                
            byteCount += subsigSize # count (body) size from sub structure

        if ("SPC" in product_list and procFFT):
            if self.parameter.radar_type is 'OLD':
                exp  = 2.0**expbuf
                spec = specbuf[:,:,idx_fft]
                spc  = spec*exp[:,:,np.newaxis]
            else:
                spc = specbuf[:,:,idx_fft]*rxspc_normalization
            self.SPC.append(spc)


    def getCoCxData(self, byte_array):

        block_length = len(byte_array)/4  # block length of data array
        #print(int("%d"%(block_length/2)))
        #print(len(unpack('%df' % block_length,byte_array)))
        cocx = np.asarray( unpack('=%df' % block_length,byte_array) ).reshape(int("%d"%(block_length/2)),2)
        #pdb.set_trace()
        return cocx


    def getCoCxExp(self, byte_array):

        block_length = len(byte_array)/2  # block length of data array
        # h: short integer (size: 2)
        #c = unpack('%dh' % int(block_length),byte_array)
        cocx = np.asarray( unpack('=%dh' % block_length,byte_array) ).reshape(int("%d"%(block_length/2)),2)
        return cocx


    def getCoCxFFT(self, byte_array, dtype='float'):

        block_length = len(byte_array)/4  # block length of data array
        # L: unsigned long (size: 4)
        #c = unpack('%dh' % int(block_length),byte_array)
        if dtype=='float':
            cocx = np.asarray( unpack('=%df' % block_length,byte_array) ).reshape(int("%d"%(block_length/2)),2)
        else:
            cocx = np.asarray( unpack('=%dl' % block_length,byte_array) ).reshape(int("%d"%(block_length/2)),2)
        return cocx


    def getCoCxIQ(self, byte_array):

        block_length = len(byte_array)/2  # block length of data array
        cocx = np.asarray( unpack('=%dh' % block_length, byte_array) ).reshape(2,int("%d"%(block_length/4)),2)
        return cocx


    def create_indextable(self):

        # get general parameters, e.g. height_list
        (sig, sigSize) = self.getSignature()
        self.sig = sig

        byteStructure = self.fh.read(sigSize)
        self.height_list = np.arange(self.parameter.HMin,self.parameter.HMax,self.parameter.dH)[:-2]
        while ( self.fh.tell() <= self.filesize-1 ) :
            self.ptr_list.append(self.fh.tell())
            (sig, sigSize) = self.getSignature()
            byteStructure = self.fh.read(sigSize)
            self.evaluateByteStructure(byteStructure)
            # srvi_values_tm = self.srvi[0].values["Tm"]
            # print(self.srvi_values)
            t = datetime.datetime(1970,1,1)+datetime.timedelta(seconds=self.srvi_values.values["Tm"])
            self.time_list.append(t)
            self.fh_list.append(self.fh)
        self.fh.seek(self.ptr_list[0])


    def getHeightList(self):

        #(sig, sigSize) = self.getSignature()
        #byteStructure = self.fh.read(sigSize)
        #self.evaluateByteStructure(byteStructure)
        #self.height_list = np.arange(self.parameter.HMin,self.parameter.HMax,self.parameter.dH)[:-2]
        return(self.height_list)


    def getNextDataset(self, product_list):

        (sig, sigSize) = self.getSignature()
        #print sig, sigSize
        byteStructure = self.fh.read(sigSize)
        self.evaluateByteStructure(byteStructure, product_list)
        return


    def setDataPosition(self, selected_time, product_list=[]):

        self.time_index = bisect.bisect_left(self.time_list ,selected_time)

        # print(self.time_index)

        if self.fh_list[self.time_index] != self.fh:
            self.fh = self.fh_list[self.time_index]
        self.fh.seek(self.ptr_list[self.time_index])


    def readDataSection(self, product_list, start_time=None, stop_time=None, quiet=False):

        import numpy as np

        self.quiet = quiet
        
        if not start_time:
            start_time = self.time_list[0]
        if not stop_time:
            stop_time  = self.time_list[-1]

        # print(start_time,stop_time)
        # print(self.time_list)
        self.setDataPosition(selected_time = start_time)


        while self.time_list[self.time_index]<stop_time:

            if not self.quiet:
              print((self.time_list[self.time_index]))
            self.getNextDataset(product_list)
            self.time_data.append(self.time_list[self.time_index])
            self.time_index += 1
            
            if self.time_index > len(self.time_list)-1:
                break

            if self.fh_list[self.time_index] != self.fh:
                self.fh = self.fh_list[self.time_index]
                
        self.SNRco = np.asarray(self.SNRco)
        self.SNRcx = np.asarray(self.SNRcx)
        self.VELco = np.asarray(self.VELco)
        self.VELcx = np.asarray(self.VELcx)
        self.RMSco = np.asarray(self.RMSco)
        self.RMScx = np.asarray(self.RMScx)
        self.EXPco = np.asarray(self.EXPco)
        self.SPC   = np.asarray(self.SPC)
        
        if self.parameter.osc == 1:
            self.IQ = np.asarray(self.IQ)
            self.IQ = self.IQ.swapaxes(0,4)
            self.IQ = self.IQ.swapaxes(4,3)

if __name__ == '__main__':

    # pds=pdsdata(pdsfile='/Data/miraMACS/pds/130829_150635.pds')
    pds=pdsdata(pdsfile='/scratch/local1/m300517/temp/160814_130355.pds')
    start_time = datetime.datetime(2013, 8, 29, 15, 6)
    stop_time = datetime.datetime(2013, 8, 29, 15, 7)

    pds.readDataSection(["SPC","SNR","Z","VEL","RMS"])
