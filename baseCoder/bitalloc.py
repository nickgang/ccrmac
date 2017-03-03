import numpy as np
# import bitalloc_ as b
import quantize as q
import window as w
import psychoac as p
import mdct as m

# Question 1.b)
def BitAllocUniform(bitBudget, maxMantBits, nBands, nLines, SMR=None):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are uniformely distributed for the mantissas.
    """
    mantBits = np.array([3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,2,2,2])

    return mantBits # TO REPLACE WITH YOUR VECTOR

def BitAllocConstSNR(bitBudget, maxMantBits, nBands, nLines, peakSPL):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep a constant
    quantization noise floor (assuming a noise floor 6 dB per bit below
    the peak SPL line in the scale factor band).
    """
    mantBits = np.array([0,0,4,13,16,16,16,15,11,0,0,0,0,0,0,0,0,13,14,0,0,14,0,0,0])
    return mantBits

def BitAllocConstMNR(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Return a hard-coded vector that, in the case of the signal use in HW#4,
    gives the allocation of mantissa bits in each scale factor band when
    bits are distributed for the mantissas to try and keep the quantization
    noise floor a constant distance below (or above, if bit starved) the
    masked threshold curve (assuming a quantization noise floor 6 dB per
    bit below the peak SPL line in the scale factor band).
    """
    mantBits = np.array([0,4,8,10,13,12,12,12,9,0,2,4,4,4,5,5,2,11,12,0,0,12,0,0,0])
    return mantBits

# Question 1.c)
def BitAlloc(bitBudget, maxMantBits, nBands, nLines, SMR):
    """
    Allocates bits to scale factor bands so as to flatten the NMR across the spectrum

       Arguments:
           bitBudget is total number of mantissa bits to allocate
           maxMantBits is max mantissa bits that can be allocated per line
           nBands is total number of scale factor bands
           nLines[nBands] is number of lines in each scale factor band
           SMR[nBands] is signal-to-mask ratio in each scale factor band

        Return:
            bits[nBands] is number of bits allocated to each scale factor band

        Logic:
           Maximizing SMR over blook gives optimization result that:
               R(i) = P/N + (1 bit/ 6 dB) * (SMR[i] - avgSMR)
           where P is the pool of bits for mantissas and N is number of bands
           This result needs to be adjusted if any R(i) goes below 2 (in which
           case we set R(i)=0) or if any R(i) goes above maxMantBits (in
           which case we set R(i)=maxMantBits).  (Note: 1 Mantissa bit is
           equivalent to 0 mantissa bits when you are using a midtread quantizer.)
           We will not bother to worry about slight variations in bit budget due
           rounding of the above equation to integer values of R(i).
    """
    mantBits = np.zeros_like(nLines,dtype=int)
    localSMR = np.array(SMR,copy=True)
    allocBits = 0

    while allocBits < bitBudget:
        smrSort = np.argsort(localSMR)[::-1]
        maxSMR = smrSort[0]

        if allocBits+nLines[maxSMR] >= bitBudget:
            for i in range(1,nBands):
                maxSMR = smrSort[i]
                if (allocBits)+nLines[maxSMR] >= bitBudget:
                    pass
                else:
                    allocBits += nLines[maxSMR]
                    mantBits[maxSMR] += 1
                    localSMR[maxSMR] -= 6
            break
        else:
            allocBits += nLines[maxSMR]
            mantBits[maxSMR] += 1
            localSMR[maxSMR] -= 6

    # Go back through and reallocate lonely bits and overflowing bits
    badBand = mantBits < maxMantBits
    while (mantBits==1).any() and badBand.any():
        # Pick lonely bit in highest critical band possible
        i = np.max(np.argwhere(mantBits==1))
        mantBits[i] = 0
        badBand[i] = False

        i = np.arange(nBands)[badBand][np.argmax((SMR-mantBits*6)[badBand])]
        if (bitBudget-nLines[i]) >= 0:
            mantBits[i] += 1
            bitBudget -= nLines[i]
            if mantBits[i] >= maxMantBits:
                badBand[i] = False
            else:
                badBand[i] = False
    return mantBits

#-----------------------------------------------------------------------------

#Testing code
if __name__ == "__main__":

    p#Testing code
if __name__ == "__main__":
    part = ' ' # Set this to run test code

    fs = 48000.
    T = 1/float(fs)
    N = 2048
    K = N/2 # Bands per block
    n = np.arange(0,N*T,T)
    freqVec = np.arange(0,fs/2,fs/float(N)) # Vector of frequencies for plotting
    freqMDCT = freqVec + (fs/float(2.*N)) # MDCT Lines
    A = np.array([0.40,0.20,0.20,0.09,0.06,0.05]) # Sinusoidal component amplitudes
    F = np.array([440,550,660,880,4400,8800]) # Sinusoidal frequencies

    xn = A[0]*np.cos(2*np.pi*F[0]*n) + A[1]*np.cos(2*np.pi*F[1]*n) +\
         A[2]*np.cos(2*np.pi*F[2]*n) + A[3]*np.cos(2*np.pi*F[3]*n) + \
         A[4]*np.cos(2*np.pi*F[4]*n) + A[5]*np.cos(2*np.pi*F[5]*n)

    if part=='2c':
        # Calculating bit allocations, assuming I = 128 kb/s/ch
        I = 128000. # Data rate in bits
        Xmdct = m.MDCT(w.KBDWindow(xn),N/2,N/2)
        MDCTscale = 1
        MDCTdata = np.power(2,float(MDCTscale))*Xmdct

        kbdWin = (1/float(N))*np.sum(np.power(w.KBDWindow(np.ones_like(xn)),2)) # Get avg pow of sine window
        kbdDB = p.SPL((2/(kbdWin))*(np.power(MDCTdata,2.)))# Find dB SPL of MDCT values

        nLines = p.AssignMDCTLinesFromFreqLimits(MDCTdata.size,fs)
        sfBands = p.ScaleFactorBands(nLines)

        SMR = p.CalcSMRs(xn,MDCTdata,MDCTscale,fs,sfBands)
        MAX_SPL = np.zeros(sfBands.nBands)
        for i in range(sfBands.nBands):
            MAX_SPL[i] = np.max(kbdDB[sfBands.lowerLine[i]:sfBands.upperLine[i]])

        bitBudget = (I/fs)*K - (4.*25.) - (4.*26.)
        bitBudget = np.floor(bitBudget)
        maxMantBits = 16

        ###### Approach i ##########
        mantBits1 = np.floor(bitBudget/K)*np.ones(sfBands.nBands)
        allocBits = np.sum(mantBits1*sfBands.nLines) # How many bits were allocated using uniform

        if allocBits < bitBudget:
            mantBits1 += np.floor((bitBudget - allocBits)/K)

        for i in range(sfBands.nBands):
            allocBits += sfBands.nLines[i] # Naively add an extra bit
            if allocBits >= bitBudget:
                break
            else:
                mantBits1[i] += 1


        mantBits1 = np.floor(mantBits1).astype(int)

        ###### Approach ii #########
        mantBits2 = np.zeros(sfBands.nBands)
        allocBits2 = 0
        noiseFloor = np.array(MAX_SPL,copy=True)

        while allocBits2 < bitBudget:
            maxPeak = np.argmax(noiseFloor)
            allocBits2 += sfBands.nLines[maxPeak]

            if allocBits2 >= bitBudget:
                break
            else:
                mantBits2[maxPeak] += 1
                noiseFloor[maxPeak] -= 6

        mantBits2[np.nonzero(mantBits2<2)] = 0 # No lonely bits
        mantBits2[np.nonzero(mantBits2>maxMantBits)] = maxMantBits # Enforce max bits

        ####### Approach iii #######
        mantBits3 = np.zeros(sfBands.nBands)
        allocBits3 = 0

        noiseFloor2 = np.array(MAX_SPL,copy=True)
        threshSFB = MAX_SPL - SMR # Get a measure of the masked threshold for each SFB
        while allocBits3 < bitBudget:
            maxThresh = np.argmin(threshSFB-noiseFloor2)
            allocBits3 += sfBands.nLines[maxThresh]

            if allocBits3 >= bitBudget:
                break
            else:
                mantBits3[maxThresh] += 1
                noiseFloor2[maxThresh] -= 6

        mantBits3[np.nonzero(mantBits3<2)] = 0 # No lonely bits
        mantBits3[np.nonzero(mantBits3>maxMantBits)] = maxMantBits # Enforce max bits

        # Plotting everything
        thresh = p.getMaskedThreshold(xn,MDCTdata,MDCTscale,fs,sfBands)
        plt.figure(1)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits1[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2ci. Noise Floor Plotted (in Red) with Uniform Bit Allocation')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

        plt.figure(2)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits2[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2cii. Noise Floor Plotted (in Red) with Constant Noise Floor Strategy')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

        plt.figure(3)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits3[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2ciii. Noise Floor Plotted (in Red) with Constant MNR Strategy')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

    if part=='2d':
        # Calculating bit allocations, assuming I = 192 kb/s/ch
        I = 192000.
        Xmdct = m.MDCT(w.KBDWindow(xn),N/2,N/2)
        MDCTscale = 1
        MDCTdata = np.power(2,float(MDCTscale))*Xmdct

        kbdWin = (1/float(N))*np.sum(np.power(w.KBDWindow(np.ones_like(xn)),2)) # Get avg pow of sine window
        kbdDB = p.SPL((2/(kbdWin))*(np.power(MDCTdata,2.)))# Find dB SPL of MDCT values

        nLines = p.AssignMDCTLinesFromFreqLimits(MDCTdata.size,fs)
        sfBands = p.ScaleFactorBands(nLines)

        SMR = p.CalcSMRs(xn,MDCTdata,MDCTscale,fs,sfBands)
        MAX_SPL = np.zeros(sfBands.nBands)
        for i in range(sfBands.nBands):
            MAX_SPL[i] = np.max(kbdDB[sfBands.lowerLine[i]:sfBands.upperLine[i]])

        bitBudget = (I/fs)*K - (4.*25.) - (4.*26.)
        bitBudget = np.floor(bitBudget)
        maxMantBits = 16

        ###### Approach i ##########
        mantBits1 = np.floor(bitBudget/K)*np.ones(sfBands.nBands)
        allocBits = np.sum(mantBits1*sfBands.nLines) # How many bits were allocated using uniform

        if allocBits < bitBudget:
            mantBits1 += np.floor((bitBudget - allocBits)/K)

        for i in range(sfBands.nBands):
            allocBits += sfBands.nLines[i] # Naively add an extra bit
            if allocBits >= bitBudget:
                break
            else:
                mantBits1[i] += 1


        mantBits1 = np.floor(mantBits1).astype(int)

        ###### Approach ii #########
        mantBits2 = np.zeros(sfBands.nBands)
        allocBits2 = 0
        noiseFloor = np.array(MAX_SPL,copy=True)

        while allocBits2 < bitBudget:
            maxPeak = np.argmax(noiseFloor)
            allocBits2 += sfBands.nLines[maxPeak]

            if allocBits2 >= bitBudget:
                break
            else:
                mantBits2[maxPeak] += 1
                noiseFloor[maxPeak] -= 6

        mantBits2[np.nonzero(mantBits2<2)] = 0 # No lonely bits
        mantBits2[np.nonzero(mantBits2>maxMantBits)] = maxMantBits # Enforce max bits

        ####### Approach iii #######
        mantBits3 = np.zeros(sfBands.nBands)
        allocBits3 = 0

        noiseFloor2 = np.array(MAX_SPL,copy=True)
        threshSFB = MAX_SPL - SMR # Get a measure of the masked threshold for each SFB
        while allocBits3 < bitBudget:
            maxThresh = np.argmin(threshSFB-noiseFloor2)
            allocBits3 += sfBands.nLines[maxThresh]

            if allocBits3 >= bitBudget:
                break
            else:
                mantBits3[maxThresh] += 1
                noiseFloor2[maxThresh] -= 6

        mantBits3[np.nonzero(mantBits3<2)] = 0 # No lonely bits
        mantBits3[np.nonzero(mantBits3>maxMantBits)] = maxMantBits # Enforce max bits

        print 'Mant Bits 1: ', mantBits1
        print 'Mant Bits 2: ', mantBits2
        print 'Mant Bits 3: ', mantBits3

#         test1 = test.TestBitAlloc(mantBits1,bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR,MAX_SPL)
#         test2 = test.TestBitAlloc(mantBits2,bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR,MAX_SPL)
#         test3 = test.TestBitAlloc(mantBits3,bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR,MAX_SPL)

        # Plotting everything
        thresh = p.getMaskedThreshold(xn,MDCTdata,MDCTscale,fs,sfBands)
        plt.figure(1)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits1[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2di. Noise Floor Plotted (in Red) with Uniform Bit Allocation')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

        plt.figure(2)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits2[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2dii. Noise Floor Plotted (in Red) with Constant Noise Floor Strategy')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

        plt.figure(3)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*mantBits3[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('2diii. Noise Floor Plotted (in Red) with Constant MNR Strategy')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)

    if part=='3a':
        I = 192000.
        Xmdct = m.MDCT(w.KBDWindow(xn),N/2,N/2)
        MDCTscale = 1
        MDCTdata = np.power(2,float(MDCTscale))*Xmdct

        kbdWin = (1/float(N))*np.sum(np.power(w.KBDWindow(np.ones_like(xn)),2)) # Get avg pow of sine window
        kbdDB = p.SPL((2/(kbdWin))*(np.power(MDCTdata,2.)))# Find dB SPL of MDCT values

        nLines = p.AssignMDCTLinesFromFreqLimits(MDCTdata.size,fs)
        sfBands = p.ScaleFactorBands(nLines)

        SMR = p.CalcSMRs(xn,MDCTdata,MDCTscale,fs,sfBands)

        MAX_SPL = np.zeros(sfBands.nBands)
        for i in range(sfBands.nBands):
            MAX_SPL[i] = np.max(kbdDB[sfBands.lowerLine[i]:sfBands.upperLine[i]])

        bitBudget = (I/fs)*K - (4.*25.) - (4.*26.)
        bitBudget = np.floor(bitBudget)
        maxMantBits = 16

        print BitAlloc.__doc__

        myBits = BitAlloc(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR)
        bits = b.BitAlloc(bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR)

        test.TestBitAlloc(myBits,bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR,MAX_SPL)
        test.TestBitAlloc(bits,bitBudget,maxMantBits,sfBands.nBands,sfBands.nLines,SMR,MAX_SPL)
        print 'My Bits: ',myBits
        print 'Bits: ',bits

        # Plotting Everything
        thresh = p.getMaskedThreshold(xn,MDCTdata,MDCTscale,fs,sfBands)
        plt.figure(1)
        plt.semilogx(freqVec,thresh,lw=2,label='Masking Threshold')
        plt.hold(True)
        plt.semilogx(freqVec+fs/float(2*N),kbdDB,lw=2,label='MDCT')
        plt.hold(True)
        for k in range(sfBands.nBands):
            plt.semilogx(np.ones(2)*freqMDCT[sfBands.upperLine[k]],[-50,330],color='black',ls='dotted')
            plt.hold(True)
            plt.semilogx([freqMDCT[sfBands.lowerLine[k]],freqMDCT[sfBands.upperLine[k]]],\
                         np.ones(2)*(MAX_SPL[k]-6*myBits[k]),color='red',lw=2)
        a = plt.gca()
        a.set_title('3bii. Noise Floor Plotted (in Red) with Water-Filling Bit Allocation (I=192 kb/s/ch)')
        a.set_ylabel('dB SPL')
        a.set_xlabel('Log Frequency (Hz)')
        plt.xticks([100,250,500,1000,5000,10000,15000],[100,250,500,1000,5000,10000,15000])
        a.axis('tight')
        a.legend()
        plt.rcParams['figure.figsize'] = (12.0, 8.0)
