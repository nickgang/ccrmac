﻿"""
codec.py -- The actual encode/decode functions for the perceptual audio codec

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

import numpy as np  # used for arrays

# used by Encode and Decode
from window import SineWindow,KBDWindow  # current window used for MDCT -- implement KB-derived?
from mdct import MDCT,IMDCT  # fast MDCT implementation (uses numpy FFT)
from quantize import *  # using vectorized versions (to use normal versions, uncomment lines 18,67 below defining vMantissa and vDequantize)
from SBR import * # Methods used for SBR

# used only by Encode
from psychoac import *  # calculates SMRs for each scale factor band
from bitalloc import BitAlloc,BitAllocSBR  #allocates bits to scale factor bands given SMRs
from scipy import signal # signal processing tools

SHORTBLOCKSIZE = 256
LONGBLOCKSIZE = 2048

def Decode(scaleFactor,bitAlloc,mantissa,overallScaleFactor,codingParams,iCh):
    """Reconstitutes a single-channel block of encoded data into a block of
    signed-fraction data based on the parameters in a PACFile object"""

    rescaleLevel = 1.*(1<<overallScaleFactor)
    if(codingParams.blocksize == 3):
        #print "MDCTLines: ", codingParams.nMDCTLines
        a = LONGBLOCKSIZE/2
        b = SHORTBLOCKSIZE/2
    elif (codingParams.blocksize == 2):
        a = SHORTBLOCKSIZE/2
        b = a
    elif (codingParams.blocksize == 1):
        b = LONGBLOCKSIZE/2
        a = SHORTBLOCKSIZE/2
    else:
        a = LONGBLOCKSIZE/2
        b = a
    N = a+b
    halfN = N/2

    #halfN = codingParams.nMDCTLines
    #N = 2*halfN
    # vectorizing the Dequantize function call
#    vDequantize = np.vectorize(Dequantize)

    # reconstitute the first halfN MDCT lines of this channel from the stored data
    mdctLine = np.zeros(halfN,dtype=np.float64)
    iMant = 0
    for iBand in range(codingParams.sfBands.nBands):
        nLines =codingParams.sfBands.nLines[iBand]
        if bitAlloc[iBand]:
            mdctLine[iMant:(iMant+nLines)]=vDequantize(scaleFactor[iBand], mantissa[iMant:(iMant+nLines)],codingParams.nScaleBits, bitAlloc[iBand])
        iMant += nLines
    mdctLine /= rescaleLevel  # put overall gain back to original level

    if codingParams.doSBR == True:
        ### SBR Decoder Module 1 - High Frequency Reconstruction ###
        mdctLine = HiFreqRec(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)
        ### SBR Decoder Module 2 - Additional High Frequency Components ###
        mdctLine = AddHiFreqs(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff)
        ### SBR Decoder Module 3 - Envelope Adjustment ###
        mdctLine = EnvAdjust(mdctLine,codingParams.sampleRate,codingParams.sbrCutoff,codingParams.specEnv[iCh])
        # print codingParams.specEnv # Print envelope for debugging purposes

    # IMDCT and window the data for this channel
    # data = SineWindow( IMDCT(mdctLine, halfN, halfN) )  # takes in halfN MDCT coeffs
    data = SineWindow( IMDCT(mdctLine, a, b) )  # takes in halfN MDCT coeffs
    # end loop over channels, return reconstituted time samples (pre-overlap-and-add)
    return data


def Encode(data,codingParams):
    """Encodes a multi-channel block of signed-fraction data based on the parameters in a PACFile object"""
    scaleFactor = []
    bitAlloc = []
    mantissa = []
    overallScaleFactor = []

    # loop over channels and separately encode each one
    for iCh in range(codingParams.nChannels):
        (s,b,m,o) = EncodeSingleChannel(data[iCh],codingParams,iCh)
        scaleFactor.append(s)
        bitAlloc.append(b)
        mantissa.append(m)
        overallScaleFactor.append(o)
    # return results bundled over channels
    return (scaleFactor,bitAlloc,mantissa,overallScaleFactor)


def EncodeSingleChannel(data,codingParams,iCh):
    """Encodes a single-channel block of signed-fraction data based on the parameters in a PACFile object"""
    # NEW: Determine block type and set a,b
    if(codingParams.blocksize == 3):
        #print "MDCTLines: ", codingParams.nMDCTLines
        a = codingParams.nMDCTLines
        b = int(a*(float(SHORTBLOCKSIZE)/LONGBLOCKSIZE))
    elif (codingParams.blocksize == 2):
        a = int(codingParams.nMDCTLines*(float(SHORTBLOCKSIZE)/LONGBLOCKSIZE))
        b = a
    elif (codingParams.blocksize == 1):
        b = codingParams.nMDCTLines
        a = int(b*(float(SHORTBLOCKSIZE)/LONGBLOCKSIZE))
    else:
        a = codingParams.nMDCTLines
        b = a
    N = a+b
    halfN = N/2
    #print "A: ", a
    #print "B: ", b
    #print "halfN: ", halfN

    # Reclaim nScaleBits from bands with 0 lines
    # vary bark width of bands
    # pass different flimits to AssignMDCTLines...

    nScaleBits = codingParams.nScaleBits
    maxMantBits = (1<<codingParams.nMantSizeBits)  # 1 isn't an allowed bit allocation so n size bits counts up to 2^n
    if maxMantBits>16: maxMantBits = 16  # to make sure we don't ever overflow mantissa holders
    # vectorizing the Mantissa function call
#    vMantissa = np.vectorize(Mantissa)
    sfBands = codingParams.sfBands

    # NEW compute target bit rate based on block type
    bitBudget = codingParams.targetBitsPerSample * halfN  # this is overall target bit rate
    bitBudget -=  nScaleBits*(sfBands.nBands + 1)  # less scale factor bits (including overall scale factor)
    bitBudget -= codingParams.nMantSizeBits*sfBands.nBands  # less mantissa bit allocation bits
    bitBudget -= 2 # block ID size TODO: make this a variable

    if codingParams.doSBR == True:
        # Calculate Spectral Envelope based on original signal
        specEnv = calcSpecEnv(data,codingParams.sbrCutoff,codingParams.sampleRate)
        # Append in spectral envelope for this channel into empty container
        codingParams.specEnv[iCh][:] = specEnv

        #Decimate and lowpass signal by factor determined by cutoff frequency
        doDecimate = False
        if doDecimate==True:
            Wc = codingParams.sbrCutoff/float(codingParams.sampleRate/2.)# Normalized cutoff frequency
            B,A = signal.butter(4,Wn)
            data = signal.lfilter(B,A,data)

    # window data for side chain FFT and also window and compute MDCT
    timeSamples = data
    # Window data based on block size
    mdctTimeSamples = np.append(SineWindow(np.append(timeSamples[:a],np.zeros(a)))[:a],SineWindow(np.append(np.zeros(b),timeSamples[a:]))[b:])
    # Call MDCT with a, b reflecting block size
    mdctLines = MDCT(mdctTimeSamples, a, b)

    # compute overall scale factor for this block and boost mdctLines using it
    maxLine = np.max( np.abs(mdctLines) )
    overallScale = ScaleFactor(maxLine,nScaleBits)  #leading zeroes don't depend on nMantBits
    mdctLines *= (1<<overallScale)

    # compute the mantissa bit allocations
    # compute SMRs in side chain FFT
    SMRs = CalcSMRs(timeSamples, mdctLines, overallScale, codingParams.sampleRate, sfBands)

    if codingParams.doSBR == True:
        # Critical band starting here are above cutoff
        cutBin = freqToBand(codingParams.sbrCutoff)
        # perform bit allocation using SMR results
        bitAlloc = BitAllocSBR(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs,cutBin)
    else:
        bitAlloc = BitAlloc(bitBudget, maxMantBits, sfBands.nBands, sfBands.nLines, SMRs)

    # given the bit allocations, quantize the mdct lines in each band
    scaleFactor = np.empty(sfBands.nBands,dtype=np.int32)
    nMant = halfN

    for iBand in range(sfBands.nBands):
        if not bitAlloc[iBand]: nMant-= sfBands.nLines[iBand]  # account for mantissas not being transmitted
    mantissa=np.empty(nMant,dtype=np.int32)
    iMant=0
    for iBand in range(sfBands.nBands):
        lowLine = sfBands.lowerLine[iBand]
        highLine = sfBands.upperLine[iBand] + 1  # extra value is because slices don't include last value
        nLines= sfBands.nLines[iBand]
        if(highLine - lowLine > 0):
            scaleLine = np.max(np.abs( mdctLines[lowLine:highLine] ) )
        else:
            scaleLine = abs(mdctLines[lowLine])
        scaleFactor[iBand] = ScaleFactor(scaleLine, nScaleBits, bitAlloc[iBand])
        if bitAlloc[iBand]:
            mantissa[iMant:iMant+nLines] = vMantissa(mdctLines[lowLine:highLine],scaleFactor[iBand], nScaleBits, bitAlloc[iBand])
            iMant += nLines
    # end of loop over scale factor bands

    # return results
    return (scaleFactor, bitAlloc, mantissa, overallScale)
