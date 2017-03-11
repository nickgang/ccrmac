import numpy as np
from window import HanningWindow

DBTOBITS = 6.02
FULLSCALESPL = 96.
fft = np.fft.fft

MASKTONALDROP = 16
MASKNOISEDROP = 6


def SPL(intensity):
    """
    Returns the SPL corresponding to intensity (in units where 1 implies 96dB)
    """
    return np.maximum(-30,FULLSCALESPL + 10.*np.log10(intensity+np.finfo(float).eps))

def Intensity(spl):
    """
    Returns the intensity (in units of the reference intensity level) for SPL spl
    """
    return np.power(10.,(spl-96)/10.)

def Thresh( f):
    """Returns the threshold in quiet measured in SPL at frequency f (in Hz)"""
    f = np.maximum(f,10.)
    return 3.64*np.power(f/1000.,-0.8) - 6.5*np.exp(-0.6*(f/1000.-3.3) * \
            (f/1000.-3.3)) + 0.001*np.power(f/1000.,4)

def Bark(f):
    """Returns the bark-scale frequency for input frequency f (in Hz) """
    return 13.0*np.arctan(0.76*f/1000.)+3.5*np.arctan((f/7500.)*(f/7500.))



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
        self.SPL = SPL #SPL of the masker
        self.f= f # frequency of the masker
        self.z = Bark(f) #frequency in Bark scale of masker

        self.drop = MASKTONALDROP
        if not isTonal: self.drop = MASKNOISEDROP

    def IntensityAtFreq(self,freq):
        """The intensity of this masker at frequency freq"""
        return self.IntensityAtBark(Bark(freq))

    def IntensityAtBark(self,z):
        """The intensity of this masker at Bark location z"""
        # start at dB near-masking level for type of masker
        maskedDB = self.SPL - self.drop
        #if more than half a critical band away, drop lower at appropriate
        # spreading function rate 
        if abs(self.z-z)>0.5 :
            if self.z>z:
                # masker above maskee
                maskedDB -= 27.*(self.z-z-0.5)
            else:
                # masker below maskee
                iEffect = self.SPL-40.
                if np.abs(iEffect)!=iEffect: iEffect=0.
                maskedDB -= (27.-0.367*iEffect)*(z-self.z-0.5)
        # return resulting intensity
        return Intensity(maskedDB)

    def vIntensityAtBark(self,zVec):
        """The intensity of this masker at vector of Bark locations zVec"""
        # start at dB near-masking level for type of masker
        maskedDB = np.empty(np.size(zVec),np.float)
        maskedDB.fill(self.SPL - self.drop)
        #if more than half a critical band away, drop lower at appropriate
        # spreading function rate adjust lower frequencies (beyond 0.5 Bark away)
        v = ((self.z-0.5) > zVec)
        maskedDB[v] -= 27.*(self.z-zVec[v]-0.5)
        # adjust higher frequencies (beyond 0.5 Bark away)
        iEffect = self.SPL-40.
        if iEffect<0.: iEffect=0.
        v = ((self.z+0.5) < zVec)
        maskedDB[v] -= (27.-0.367*iEffect)*(zVec[v]-(self.z+0.5))
        # return resulting intensity
        return Intensity(maskedDB)


# Default data for 25 scale factor bands based on the traditional 25 critical bands
cbFreqLimits = (  100.,   200.,   300.,   400.,   510.,
                  630.,   770.,   920.,  1080.,  1270.,
                 1480.,  1720.,  2000.,  2320.,  2700.,
                 3150.,  3700.,  4400.,  5300.,  6400.,
                 7700.,  9500., 12000., 15500., 24000.)

def AssignMDCTLinesFromFreqLimits(nMDCTLines, sampleRate, flimit = cbFreqLimits):
    """
    Assigns MDCT lines to scale factor bands for given sample rate and number
    of MDCT lines using predefined frequency band cutoffs passed as an array
    in flimit (in units of Hz). If flimit isn't passed it uses the traditional
    25 Zwicker & Fastl critical bands as scale factor bands.
    """
    # conversion from line number to absolute frequency
    lineToFreq = 0.5*sampleRate/nMDCTLines
    maxFreq = (nMDCTLines-1+0.5)*lineToFreq
    # first get upper line for each band
    nLines = [ ]
    iLast = -1  # the last line before the start of this group
                # (-1 when we start at zero)
    for iLine in xrange(len(flimit)):
        # if we are above last MDCT line, put them all in last band and stop loop
        if flimit[iLine]> maxFreq:
            nLines.append( nMDCTLines-1-iLast )
            break
        # otherwise, find last lin in this band, compute number, and save last
        # for next loop
        iUpper = int(flimit[iLine]/lineToFreq-0.5)
        nLines.append( iUpper-iLast )
        iLast = iUpper
    return nLines

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
        self.nBands = len(nLines)
        self.nLines=np.array(nLines,dtype=np.uint16)
        self.lowerLine=np.empty(self.nBands,dtype=np.uint16)
        self.upperLine=np.empty(self.nBands,dtype=np.uint16)
        self.lowerLine[0]=0
        self.upperLine[0]= nLines[0]-1
        for iBand in range(1,self.nBands):
            self.lowerLine[iBand]=self.upperLine[iBand-1]+1
            self.upperLine[iBand]=self.upperLine[iBand-1]+nLines[iBand]



def getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands):
    """
    Return Masked Threshold evaluated at MDCT lines.

    Used by CalcSMR, but can also be called from outside this module, which may
    be helpful when debugging the bit allocation code.
    """

    N = len(data)
    nLines = N/2                        # there are N/2 indep freq lines for an N FFT
    lineToFreq = 0.5*sampleRate/nLines  # line spacing in Hz
    nBands = sfBands.nBands             # number of sf bands spanned by MDCT lines

    # compute FFT of Hanning-windowed time samples
    fftData=fft(HanningWindow(data))[:nLines]

    # Extract spectral densities from FFT in terms of both Intensity and SPL
    # (equal to 4/N^2 for FFT Parsevals (positive freq only) * 8/3 for Hanning
    # window )
    fftIntensity =32./3./N/N*np.abs(fftData)**2
    fftSPL = SPL(fftIntensity)

    # start collecting tonal and noise maskers
    maskers = []

    # Find peaks in FFT to define tonal maskers and then delete spectral
    # densities in tonal maskers
    # (note: can't detect a peak unless you can see 2 lines on each side)
    for i in range(2,nLines-2):
        if  fftIntensity[i]>fftIntensity[i-1] and \
            fftIntensity[i]>fftIntensity[i+1] and \
            fftSPL[i]-fftSPL[i-2]>7 and \
            fftSPL[i]-fftSPL[i+2]>7 :

            # we have found a tonal masker -- compute SPL and frequency

            # aggregate intensity over 3 lines of peak
            spl = fftIntensity[i]+fftIntensity[i-1]+fftIntensity[i+1]

            # estimate freq of masker
            f = lineToFreq*(i*fftIntensity[i]+(i-1)*fftIntensity[i- 1] + \
                (i+1)*fftIntensity[i+1])/spl

            spl = SPL(spl)  # SPL of masker (note: needed to do
                            # intensity-weighted average before converting)

            # if above threshold-in-quiet, add it to list of maskers
            if spl>Thresh(f) :
                # Masker is above threshold-in-quiet so add it in to maskers[]
                maskers.append(Masker(f,spl))
            # clear out identified tonal maskers so they don't contribute to
            # noise masking
            fftIntensity[i]=fftIntensity[i-1]=fftIntensity[i+1]=0.
        # end of peak detection if
    # end of loop looking for tonal maskers

    # Allocate rest of spectrum into noise maskers -- one per scale-factor band
    for i in range(nBands) : # loop over scale factor bands
        spl=0.
        f=0.
        # loop over lines in current scale-factor band
        for j in range(sfBands.lowerLine[i],sfBands.upperLine[i]+1):
            spl += fftIntensity[j]  # summed intensity
            f += fftIntensity[j]*j  # to compute intensity-weighted average
                                    # frequency
        if spl > 0.:
            f = f*lineToFreq/spl  # intensity-weighted frequency of noise masker
            spl = SPL(spl)      # SPL of noise masker
            # if above threshold-in-quiet, add it to list of maskers
            if spl>Thresh(f): maskers.append(Masker(f,spl,isTonal=False))
    # end of loop over bands to make noise maskers

    # Sum over maskers and thresh in quiet to get global threshold at each MDCT
    # frequency location get frequencies and Bark values at each frequency
    # line location

    # mdct line frequencies (i+0.5)*lineToFreq
    fline = lineToFreq*np.linspace(0.5,nLines+0.5,nLines)
    zline = Bark(fline) # these are the Bark values at the frequencies

    # start at zero intensity everywhere
    maskedSPL = np.zeros(nLines, dtype=np.float64)

    # add in masker intensity from each masker
    for m in maskers: maskedSPL += m.vIntensityAtBark(zline)

    # intensity add in threshold in quiet
    maskedSPL += Intensity(Thresh(fline))

    # convert to SPL and return
    return SPL(maskedSPL)



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

    nBands = sfBands.nBands             # number of sf bands spanned by MDCT lines

    # also get spectral densitites from MDCT data (in SPL to compute SMR later)
    dtemp2= DBTOBITS*MDCTscale      # adjust MDCTdata level for any overall
                                    # scale factor
    mdctSPL = 4.*MDCTdata**2        # 8/N^2 for MDCT Parsevals * 2 for sine
                                    # window but 4/N^2 already in MDCT forward
    mdctSPL = SPL(mdctSPL) - dtemp2

    maskedSPL = getMaskedThreshold(data, MDCTdata, MDCTscale, sampleRate, sfBands)

    # Compute and return SMR for each scale factor band as max value for
    # lines in band
    SMR = np.empty(nBands,dtype=np.float64)
    for i in range(nBands) :
        lower = sfBands.lowerLine[i]
        upper = sfBands.upperLine[i]+1 # slices don't include last item in range
        SMR[i]= np.max(mdctSPL[lower:upper]-maskedSPL[lower:upper])
    return SMR


def CalcFFTSMRs(data, MDCTdata, MDCTscale, sampleRate, sfBands):
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
    N = len(data)
    nLines = N/2                        # there are N/2 indep freq lines for an N FFT
    lineToFreq = 0.5*sampleRate/nLines  # line spacing in Hz
    nBands = sfBands.nBands             # number of sf bands spanned by MDCT lines

    # compute FFT of Hanning-windowed time samples
    fftData=fft(HanningWindow(data))[:nLines]

    # Extract spectral densities from FFT in terms of both Intensity and SPL
    # (equal to 4/N^2 for FFT Parsevals (positive freq only) * 8/3 for Hanning
    # window )
    fftIntensity =32./3./N/N*np.abs(fftData)**2
    fftSPL = SPL(fftIntensity)

    # also get spectral densitites from MDCT data (in SPL to compute SMR later)
    dtemp2= DBTOBITS*MDCTscale      # adjust MDCTdata level for any overall
                                    # scale factor
    mdctSPL = 4.*MDCTdata**2        # 8/N^2 for MDCT Parsevals * 2 for sine
                                    # window but 4/N^2 already in MDCT forward
    mdctSPL = SPL(mdctSPL) - dtemp2

    # start collecting tonal and noise maskers
    maskers = []

    # Find peaks in FFT to define tonal maskers and then delete spectral
    # densities in tonal maskers
    # (note: can't detect a peak unless you can see 2 lines on each side)
    for i in range(2,nLines-2):
        if  fftIntensity[i]>fftIntensity[i-1] and \
            fftIntensity[i]>fftIntensity[i+1] and \
            fftSPL[i]-fftSPL[i-2]>7 and \
            fftSPL[i]-fftSPL[i+2]>7 :

            # we have found a tonal masker -- compute SPL and frequency

            # aggregate intensity over 3 lines of peak
            spl = fftIntensity[i]+fftIntensity[i-1]+fftIntensity[i+1]

            # estimate freq of masker
            f = lineToFreq*(i*fftIntensity[i]+(i-1)*fftIntensity[i- 1] + \
                (i+1)*fftIntensity[i+1])/spl

            spl = SPL(spl)  # SPL of masker (note: needed to do
                            # intensity-weighted average before converting)

            # if above threshold-in-quiet, add it to list of maskers
            if spl>Thresh(f) :
                # Masker is above threshold-in-quiet so add it in to maskers[]
                maskers.append(Masker(f,spl))
            # clear out identified tonal maskers so they don't contribute to
            # noise masking
            fftIntensity[i]=fftIntensity[i-1]=fftIntensity[i+1]=0.
        # end of peak detection if
    # end of loop looking for tonal maskers

    # Allocate rest of spectrum into noise maskers -- one per scale-factor band
    for i in range(nBands) : # loop over scale factor bands
        spl=0.
        f=0.
        # loop over lines in current scale-factor band
        for j in range(sfBands.lowerLine[i],sfBands.upperLine[i]+1):
            spl += fftIntensity[j]  # summed intensity
            f += fftIntensity[j]*j  # to compute intensity-weighted average
                                    # frequency

        if spl > 0.:
            f=f*lineToFreq/spl  # intensity-weighted frequency of noise masker
            spl=SPL(spl)        # SPL of noise masker
            # if above threshold-in-quiet, add it to list of maskers
            if spl>Thresh(f): maskers.append(Masker(f,spl,isTonal=False))
            
    # end of loop over bands to make noise maskers

    # Sum over maskers and thresh in quiet to get global threshold at each MDCT
    # frequency location get frequencies and Bark values at each frequency
    # line location

    # mdct line frequencies (i+0.5)*lineToFreq
    fline=lineToFreq*np.linspace(0.5,nLines+0.5,nLines)
    zline = Bark(fline) # these are the Bark values at the frequencies

    # start at zero intensity everywhere
    maskedSPL = np.zeros(nLines, dtype=np.float64)

    # add in masker intensity from each masker
    for m in maskers: maskedSPL += m.vIntensityAtBark(zline)

    # intensity add in threshold in quiet
    maskedSPL += Intensity(Thresh(fline))

    # convert to SPL
    maskedSPL = SPL(maskedSPL)

    # Compute and return SMR for each scale factor band as max value for
    # lines in band
    SMR = np.empty(nBands,dtype=np.float64)
    for i in range(nBands) :
        lower = sfBands.lowerLine[i]
        upper = sfBands.upperLine[i]+1 # slices don't include last item in range
        SMR[i]= np.max(fftSPL[lower:upper]-maskedSPL[lower:upper])
    return SMR


if __name__ == '__main__':
    """
    Test code (use to generate solution figures)

    NOTE TO THE TA:

        Before generating the .pyc file, make sure to set all test flags to
        `False`.
    """

    run_test = True

    if not run_test:
        import sys
        sys.exit()

    import matplotlib as mat
    import matplotlib.pyplot as plt
    from mdct import *
    from window import *
    mat.rcParams.update({'font.size': 16})

    # Create test signal ######################################################

    Fs = 48000.
    #Ns = (512, 1024, 2048)

    components = ((0.40, 440.), (0.20, 550.), (0.20, 660.), \
                  (0.09, 880.), (0.06, 4400.), (0.05, 8800.))

    #N = 2048                      # Number of samples
    N = 1024                      # Number of samples
    n = np.arange(N, dtype=float) # Sample index

    x = np.zeros_like(n)          # Signal
    for pair in components:
        x += pair[0]*np.cos(2*np.pi*pair[1]*n/Fs)


    # Take FFT ################################################################
    X = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
    f = np.fft.fftfreq(N, 1/Fs)[0:N/2]
    # there are N/2 indep freq lines for an N FFT 
    FFTfreqs = f        
    nFFTLines = N/2             
    lineToFreq = 0.5*Fs/(N/2.)  # line spacing in Hz
    

    # Take FFT SPL #############################################################
    Xspl = SPL( 8./3. * 4./N**2 * np.abs(X)**2)
    f = np.fft.fftfreq(N, 1/Fs)[0:N/2]
    plt.figure(figsize=(14, 6))
    plt.semilogx( f, Thresh(f), 'b', label='Threshold in quiet')
    plt.semilogx( f, Xspl, 'g', label='FFT SPL')
    plt.xlabel('Frequency (Hz)')
    plt.xlim(50, Fs/2)
    plt.ylim(-10, 100)
    ax = plt.axes()
    ax.yaxis.grid()

    # Add scale factor bands ##################################################

    nMDCTLines = N/2
    nLines = AssignMDCTLinesFromFreqLimits(nMDCTLines,Fs)
    mySFB = ScaleFactorBands(nLines)
    scaleplt = plt.vlines(cbFreqLimits, -20, 150,
                          linestyles=':', colors='k', alpha=0.75)

    cbCenters = np.array([50]+[l for l in cbFreqLimits])
    cbCenters = np.sqrt(cbCenters[1:] * cbCenters[:-1])
    for i, val in enumerate(cbCenters, start=1):
        scaletextplt = plt.text(val, -5, str(i), horizontalalignment='center')


    #plot MDCT SPLs
    sinepow = 0.5
    kbdpow  = (1./N)*np.sum(KBDWindow(np.ones(N)))

    MDCTdataSin = MDCT(SineWindow(x),N/2,N/2)
    MDCTscale   = np.zeros_like(MDCTdataSin)
    mdctPSDSin  = np.power(
                    np.absolute(np.power(2.,MDCTscale)*MDCTdataSin*N/2),
                    2
                  )

    MDCTdataKBD = MDCT(KBDWindow(x),N/2,N/2)
    mdctPSDKBD  = np.power(
                    np.absolute(np.power(2.,MDCTscale)*MDCTdataKBD*N/2),
                    2
                  )

    # |X[k]|^2
    mdctIntensitySin = 8.*mdctPSDSin/(np.power(N,2)*sinepow)
    mdctSPLSin       = SPL(mdctIntensitySin)
    mdctIntensityKBD = 8.*mdctPSDKBD/(np.power(N,2)*kbdpow)
    mdctSPLKBD       = SPL(mdctIntensityKBD)

    MDCTfreqs  = (np.arange(N/2) + 0.5)*Fs/float(N)
    plt.plot(MDCTfreqs, mdctSPLSin, 'c-', alpha = 0.8,
            label='MDCT SPL - Sine Window')
    plt.plot(MDCTfreqs, mdctSPLKBD, 'm-', alpha = 0.8,
            label='MDCT SPL - KBD Window')

    # Calculate masking #######################################################
    maskThresh = np.zeros_like(f)
    intensity_sum = np.zeros_like(maskThresh)
    
    # Tonal masker detection
    maskers = []
    Xint = Intensity(Xspl)
        
    for i in range(1,nFFTLines-1):
    
        if  Xint[i]>Xint[i-1] and Xint[i]>Xint[i+1]:

            # we have found a tonal masker -- compute SPL and frequency

            # aggregate intensity over 3 lines of peak
            peak = Xint[i]+Xint[i-1]+Xint[i+1]

            # estimate freq of masker
            fm = lineToFreq*(i*Xint[i]+(i-1)*Xint[i- 1] + \
                (i+1)*Xint[i+1])/peak

            spl = SPL(peak)  # SPL of masker (note: needed to do
                            # intensity-weighted average before converting)

            # if above threshold-in-quiet, add it to list of maskers
            if spl>Thresh(fm) :
                # Masker is above threshold-in-quiet so add it in to maskers[]
                maskers.append(Masker(fm,spl,True))
        # end of peak detection if
    # end of loop looking for tonal maskers
    
    for masker in maskers:
        intensity_sum += masker.vIntensityAtBark(Bark(f+0.5))

    intensity_sum += Intensity(Thresh(f+0.5))
    maskThresh = SPL( intensity_sum )
    plt.plot(f+0.5, maskThresh, 'r--', linewidth=2.0, label='Masked Threshold')

    plt.ylabel('SPL (dB)')
    plt.legend()
    # fix possible layout issues
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.4)
    # plt.savefig('../figures/spectraAndMaskingCurve.png', bbox_inches='tight')
    # plt.show()


    # SMRs ####################################################################

    nLines = N/2                # there are N/2 indep freq lines for an N FFT
    lineToFreq = 0.5*Fs/nLines  # line spacing in Hz
    nBands = mySFB.nBands       # number of sf bands spanned by MDCT lines

    SMRmdctSin = np.empty(nBands,dtype=np.float64)
    SMRmdctKBD = np.empty(nBands,dtype=np.float64)
    SMRfft     = np.empty(nBands,dtype=np.float64)
    Xfft = np.abs(np.fft.fft(HanningWindow(x)))[0:N/2]
    XfftSpl = SPL( 8./3. * 4./float(N**2) * np.abs(Xfft)**2)
    for i in range(nBands) :
        lower = mySFB.lowerLine[i]
        upper = mySFB.upperLine[i]+1 # slices don't include last item in range
        SMRmdctSin[i]= np.amax(mdctSPLSin[lower:upper]-maskThresh[lower:upper])
        SMRmdctKBD[i]= np.amax(mdctSPLKBD[lower:upper]-maskThresh[lower:upper])
        SMRfft[i] = np.amax(XfftSpl[lower:upper]-maskThresh[lower:upper])

    for i in range(nBands) :
        print "{} & {} & {:8.2f} & {:8.2f} & {:8.2f} \\\\".format(i+1,mySFB.upperLine[i]+1-mySFB.lowerLine[i],SMRfft[i],SMRmdctSin[i],SMRmdctKBD[i])
    
    plt.figure(figsize=(14, 6))
    plt.xlabel('Scale factor band')
    plt.ylabel('SMR (dB)')
    plt.ylim(-10, 35)
    plt.title('Maximum SMR per band')
    plt.grid(True)
    plt.plot(SMRmdctSin, 'c', drawstyle="steps-post", label="MDCT sine window")
    plt.plot(SMRmdctKBD, 'm', drawstyle="steps-post", label="MDCT KBD window")
    plt.plot(SMRfft,     'g', drawstyle="steps-post", label="FFT Hanning window")
    plt.legend()
    # fix possible layout issues
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None,
                        wspace=None, hspace=0.4)
    # plt.savefig('../figures/dftVsMDCTsmrs.png', bbox_inches='tight')
    plt.show()
