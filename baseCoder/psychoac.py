import numpy as np
from mdct import *
from window import *
import matplotlib.pyplot as plt

def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    spl = 96 + 10*np.log10(intensity)
    if type(intensity)is np.ndarray:
        spl[np.nonzero(spl<=-30)] = -30 # Lowest possible value is -30 dB SPL (vectors)
    elif spl<=-30:
        spl = -30 # Lowest possible value is -30 dB SPL (scalars)

    return spl

def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    intensity = np.power(10,0.1*(spl-96))
    return intensity

def Thresh(f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    localF = np.array(f,copy=True)
    if type(localF)is np.ndarray:
        localF[np.nonzero(localF<=10)] = 10 # Clip minimum value to 10Hz (vectors)
    elif localF<=10:
        localF = 10 # Clip minimum value to 10Hz (scalars)

    fkHz = localF*0.001 # Convert to kHz
    thresh = (3.64*np.power(fkHz,-0.8))-(6.5*np.exp(-0.6*np.power(fkHz-3.3,2.)))+(np.power(10.,-3.)*np.power(fkHz,4.))
    return thresh

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    fkHz = f*0.001 # Convert to kHz
    bark = 13*np.arctan(0.76*fkHz) + 3.5*np.arctan(np.power(fkHz/7.5,2))
    return bark

class Masker:
    """
    a Masker whose masking curve drops linearly in Bark beyond 0.5 Bark from the
    masker frequency
    """

    def __init__(self,f,SPL,isTonal=True):
        """
        initialized with the frequency and SPL of a masker and whether or not
        it is Tonal
        """
        self.f = f
        self.SPL = SPL
        self.isTonal=isTonal

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        z = Bark(freq)
        inty = self.vIntensityAtBark(z)
        return inty

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        dz = z - Bark(self.f) # Distance from center of curve
        delta = 6 + 10*(self.isTonal==True) # Masking curve drop based on type of masker

        if np.absolute(dz)<=0.5:
            spl = 0 # Flat within half a Bark on either side
        elif dz<-0.5:
            spl = -27*(np.absolute(dz)-0.5) # Constant slope on the left
        elif dz>0.5:
            spl = (-27+0.367*np.max(self.SPL-40,0))*(np.absolute(dz)-0.5)

        spl -= delta
        inty = Intensity(spl) # Convert to intensity value
        return inty

    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        dz = zVec-Bark(self.f) # Distance of each element from center of curve
        delta = 6 + 10*(self.isTonal==True) # Masking curve drop based on type of masker
        Lm = self.SPL
        spl = (-27*(np.absolute(dz)-0.5))*(dz<-0.5) + \
                ((-27+0.367*np.max(Lm-40,0))*(np.absolute(dz)-0.5))*(dz>0.5)
        spl = spl + Lm - delta
        inty = Intensity(spl) # Convert to intensity value

        return inty


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = np.array([100,200,300,400,510,630,770,920,1080,1270,1480,1720,2000,\
                2320,2700,3150,3700,4400,5300,6400,7700,9500,12000,15500,24000])

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    N = 2*nMDCTLines
    fs = sampleRate # Easier Notation

    MDCTLines = np.arange(0,fs/2.,fs/float(N)) + fs/float(2.*N) # MDCT Line locations
    nInScaleBand = np.zeros(len(flimit))

    for i in range(flimit.size):
        if i==0:
            nInScaleBand[i] = np.nonzero(MDCTLines<=flimit[i])[i].size
        else:
            nInScaleBand[i] = np.intersect1d(np.nonzero(MDCTLines>flimit[i-1]),np.nonzero(MDCTLines<=flimit[i])).size

    return nInScaleBand

class ScaleFactorBands:
    """
    A set of scale factor bands (each of which will share a scale factor and a
    mantissa bit allocation) and associated MDCT line mappings.

    Instances know the number of bands nBands; the upper and lower limits for
    each band lowerLimit[i in range(nBands)], upperLimit[i in range(nBands)];
    and the number of lines in each band nLines[i in range(nBands)]
    """

    def __init__(self,nLines):
        """
        Assigns MDCT lines to scale factor bands based on a vector of the number
        of lines in each band
        """
        self.nLines = np.array(nLines).astype(int)
        self.nBands = self.nLines.size
        sumMat = np.tril(np.ones(self.nBands),-1) # Lower Triangular Matrix

        self.lowerLine = np.dot(sumMat,self.nLines).astype(int)
        self.upperLine = (self.nLines + self.lowerLine - 1).astype(int)

def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """
    N = data.size
    freqVec = np.arange(0,sampleRate/2,sampleRate/float(N)) # Vector of frequencies
    freqMDCT = freqVec + (sampleRate/float(2.*N)) # MDCT Lines

    Xn = np.fft.fft(HanningWindow(data),N) # Compute FFT of signal
    hannWin = (1/float(N))*np.sum(np.power(HanningWindow(np.ones_like(data)),2)) # Get avg pow of hann window
    XnI = (4/(np.power(N,2)*hannWin))*(Xn*np.conjugate(Xn)) # Compute values of FFT intensity
    XndB = SPL(XnI) # Convert from intensity to dB SPL

    # Find peaks in spectrum and their levels
    peaks = np.array([])
    freqs = np.array([])

    # Trying to vectorize peak finding
    # lastXvec = XnI[1:N/2-2];
    # lastFvec = freqVec[1:N/2-2];
    # Xvec = XnI[2:N/2-1];
    # Fvec = freqVec[1:N/2-2];
    # nextXvec = XnI[3:N/2];
    # nextFvec = freqVec[3:N/2];
    #
    # peakIndex = np.logical_and(np.greater(Xvec,lastXvec)*1,np.greater(Xvec,nextXvec)*1)
    # peak = (lastXvec[peakIndex]*np.conjugate(lastXvec[peakIndex])) +\
    #         (Xvec[peakIndex]*np.conjugate(Xvec[peakIndex])) + \
    #         (nextXvec[peakIndex]*np.conjugate(nextXvec[peakIndex])) # Aggregate intensity across peak
    # peak = SPL((4/(np.power(N,2.))*hannWin))*peak) # Use this to find SPL of peak
    #
    # freq = ((lastXvec[peakIndex]*lastFvec[peakIndex])+(Xvec[peakIndex]*Fvec[peakIndex])\
    #             (nextXvec[peakIndex]*nextFvec[peakIndex])) / \
    #             (lastXvec[peakIndex]+Xvec[peakIndex]+nextXvec[peakIndex])

    for i in np.arange(1,(N/2)-1):
        if XnI[i]>XnI[i-1] and XnI[i]>XnI[i+1]:
            peak = (Xn[i-1]*np.conjugate(Xn[i-1]))+(Xn[i]*np.conjugate(XnI[i]))\
                    + (Xn[i+1]*np.conjugate(Xn[i+1])) # Aggregate intensity across peak
            peak = SPL((4./(np.power(N,2.)*hannWin))*peak) # Use this to find SPL of peak

            freq = ((XnI[i-1]*freqVec[i-1])+(XnI[i]*freqVec[i])+(XnI[i+1]*freqVec[i+1]))\
                    /(XnI[i-1]+XnI[i]+XnI[i+1]) # Perform intensity weighted average

            freqs = np.append(freqs,np.absolute(freq)) # Current frequency bin
            peaks = np.append(peaks,np.absolute(peak)) # Current amplitude

    maskingCurve = Intensity(Thresh(freqMDCT)) # Initialize masking curve with threshold of hearing
    maskers = np.array([])

    for j in range(peaks.size):
        isTonal=True # Hardcoding for now but we would ideally check if this masker is tonal here
        maskers = np.append(maskers,Masker(freqs[j],peaks[j],isTonal))
        maskingCurve += maskers[j].vIntensityAtBark(Bark(freqMDCT)) # Create masking curve by adding intensities

    maskThresh = SPL(maskingCurve)
    return maskThresh

def CalcSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Set SMR for each critical band in sfBands.

    Arguments:
                data:       is an array of N time domain samples
                MDCTdata:   is an array of N/2 MDCT frequency lines for the data
                            in data which have been scaled up by a factor
                            of 2^MDCTscale
                MDCTscale:  is an overall scale factor for the set of MDCT
                            frequency lines
                sampleRate: is the sampling rate of the time domain samples
                sfBands:    points to information about which MDCT frequency lines
                            are in which scale factor band

    Returns:
                SMR[sfBands.nBands] is the maximum signal-to-mask ratio in each
                                    scale factor band

    Logic:
                Performs an FFT of data[N] and identifies tonal and noise maskers.
                Sums their masking curves with the hearing threshold at each MDCT
                frequency location to the calculate absolute threshold at those
                points. Then determines the maximum signal-to-mask ratio within
                each critical band and returns that result in the SMR[] array.
    """
    N = data.size
    maskThresh = getMaskedThreshold(data,MDCTdata,MDCTscale,sampleRate,sfBands) # Compute our masking curves
    MDCTdata = MDCTdata*np.power(2,float(-1*MDCTscale)) # Remove scale factor

    sineWin = (1/float(N))*np.sum(np.power(SineWindow(np.ones_like(data)),2)) # Get avg pow of KBD window
    sineDB = SPL((2/(sineWin))*(np.power(np.absolute(MDCTdata),2.)))# Find dB SPL of MDCT values

    #print sineDB.size
    #print maskThresh.size

    smrVec = sineDB - maskThresh # This gives SMR for every MDCT line individually
    #print sfBands.lowerLine, sfBands.upperLine, "\n"
    #print smrVec, "\n"
    #print smrVec.size, "\n"
    SMR = np.zeros(sfBands.nBands)

    for i in range(sfBands.nBands):

        if(sfBands.upperLine[i]-(sfBands.lowerLine[i]) > 0):
            SMR[i] = np.max(smrVec[sfBands.lowerLine[i]:sfBands.upperLine[i]+1]) # Look at max SMR in this critical band
        else:
            SMR[i] = smrVec[sfBands.lowerLine[i]]
    return SMR

def DetectTransient(data, codingParams):
    fs = 48000
    #fs = codingParams.sampleRate
    N = data.size
    MDCTdata = MDCT(SineWindow(data),N/2,N/2)
    sineWin = (1/float(N))*np.sum(np.power(SineWindow(np.ones_like(data)),2)) # Get avg pow of KBD window
    sineDB = SPL((2/(sineWin))*(np.power(np.absolute(MDCTdata),2.)))# Find dB SPL of MDCT values
    thresh = getMaskedThreshold(data,MDCTdata,0,fs,ScaleFactorBands(AssignMDCTLinesFromFreqLimits(MDCTdata.size,fs)))
    PE = np.sum(np.log2(1+np.sqrt(Intensity(sineDB)/(Intensity(sineDB-thresh)))))/(MDCTdata.size)
    delta = (PE - codingParams.prevPE)
    # print delta # debug print to check PE change between blocks
    DT = delta > 1
    #print PE
    codingParams.prevPE = PE
    return (DT)
#-----------------------------------------------------------------------------
