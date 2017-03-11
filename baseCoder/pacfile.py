"""
pacfile.py -- Defines a PACFile class to handle reading and writing audio
data to an audio file holding data compressed using an MDCT-based perceptual audio
coding algorithm.  The MDCT lines of each audio channel are grouped into bands,
each sharing a single scaleFactor and bit allocation that are used to block-
floating point quantize those lines.  This class is a subclass of AudioFile.

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of the AudioFile
class.

Notes on reading and decoding PAC files:

    The OpenFileForReading() function returns a CodedParams object containing:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (block switching not supported)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        sfBands = a ScaleFactorBands object
        overlapAndAdd = decoded data from the prior block (initially all zeros)

    The returned ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand


Notes on encoding and writing PAC files:

    When writing to a PACFile the CodingParams object passed to OpenForWriting()
    should have the following attributes set:

        nChannels = the number of audio channels
        sampleRate = the sample rate of the audio samples
        numSamples = the total number of samples in the file for each channel
        nMDCTLines = half the MDCT block size (format does not support block switching)
        nSamplesPerBlock = MDCTLines (but a name that PCM files look for)
        nScaleBits = the number of bits storing scale factors
        nMantSizeBits = the number of bits storing mantissa bit allocations
        targetBitsPerSample = the target encoding bit rate in units of bits per sample

    The first three attributes (nChannels, sampleRate, and numSamples) are
    typically added by the original data source (e.g. a PCMFile object) but
    numSamples may need to be extended to account for the MDCT coding delay of
    nMDCTLines and any zero-padding done in the final data block

    OpenForWriting() will add the following attributes to be used during the encoding
    process carried out in WriteDataBlock():

        sfBands = a ScaleFactorBands object
        priorBlock = the prior block of audio data (initially all zeros)

    The passed ScaleFactorBands object, sfBands, contains an allocation of
    the MDCT lines into groups that share a single scale factor and mantissa bit
    allocation.  sfBands has the following attributes available:

        nBands = the total number of scale factor bands
        nLines[iBand] = the number of MDCT lines in scale factor band iBand
        lowerLine[iBand] = the first MDCT line in scale factor band iBand
        upperLine[iBand] = the last MDCT line in scale factor band iBand

Description of the PAC File Format:

    Header:

        tag                 4 byte file tag equal to "PAC "
        sampleRate          little-endian unsigned long ("<L" format in struct)
        nChannels           little-endian unsigned short("<H" format in struct)
        numSamples          little-endian unsigned long ("<L" format in struct)
        nMDCTLines          little-endian unsigned long ("<L" format in struct)
        nScaleBits          little-endian unsigned short("<H" format in struct)
        nMantSizeBits       little-endian unsigned short("<H" format in struct)
        nSFBands            little-endian unsigned long ("<L" format in struct)
        for iBand in range(nSFBands):
            nLines[iBand]   little-endian unsigned short("<H" format in struct)

    Each Data Block:  (reads data blocks until end of file hit)

        for iCh in range(nChannels):
            nBytes          little-endian unsigned long ("<L" format in struct)
            as bits packed into an array of nBytes bytes:
                overallScale[iCh]                       nScaleBits bits
                for iBand in range(nSFBands):
                    scaleFactor[iCh][iBand]             nScaleBits bits
                    bitAlloc[iCh][iBand]                nMantSizeBits bits
                    if bitAlloc[iCh][iBand]:
                        for m in nLines[iBand]:
                            mantissa[iCh][iBand][m]     bitAlloc[iCh][iBand]+1 bits
                <extra custom data bits as long as space is included in nBytes>

"""

from audiofile import * # base class
from bitpack import *  # class for packing data into an array of bytes where each item's number of bits is specified
import codec    # module where the actual PAC coding functions reside(this module only specifies the PAC file format)
from psychoac import ScaleFactorBands, AssignMDCTLinesFromFreqLimits, DetectTransient  # defines the grouping of MDCT lines into scale factor bands
import sys
import matplotlib.pyplot as plt

import numpy as np  # to allow conversion of data blocks to numpy's array object
MAX16BITS = 32767
SHORTBLOCKSIZE = 256
LONGBLOCKSIZE = 2048

shortFreqLimits = np.array([300,630,1080,1720,2700,4400,7700,15500])

class PACFile(AudioFile):
    """
    Handlers for a perceptually coded audio file I am encoding/decoding
    """

    # a file tag to recognize PAC coded files
    tag='PAC '

    def ReadFileHeader(self):
        """
        Reads the PAC file header from a just-opened PAC file and uses it to set
        object attributes.  File pointer ends at start of data portion.
        """
        # check file header tag to make sure it is the right kind of file
        tag=self.fp.read(4)
        if tag!=self.tag: raise "Tried to read a non-PAC file into a PACFile object"
        # use struct.unpack() to load up all the header data
        (sampleRate, nChannels, numSamples, nMDCTLines, nScaleBits, nMantSizeBits) \
                 = unpack('<LHLLHH',self.fp.read(calcsize('<LHLLHH')))
        nBands = unpack('<L',self.fp.read(calcsize('<L')))[0]
        nLines=  unpack('<'+str(nBands)+'H',self.fp.read(calcsize('<'+str(nBands)+'H')))
        sfBands=ScaleFactorBands(nLines)
        # load up a CodingParams object with the header data
        myParams=CodingParams()
        myParams.sampleRate = sampleRate
        myParams.nChannels = nChannels
        myParams.numSamples = numSamples
        myParams.nMDCTLines = myParams.nSamplesPerBlock = nMDCTLines
        myParams.nScaleBits = nScaleBits
        myParams.nMantSizeBits = nMantSizeBits
        # SBR Stuff
        myParams.sbrCutoff = 9500. # Specified in Hz
        myParams.doSBR = True # For toggling SBR algorithm
        myParams.nSpecEnvBits = 8 # number of bits per spectral envelope band
        myParams.specEnv = np.zeros((nChannels,24-codec.freqToBand(myParams.sbrCutoff)))
        # Block Switching Stuff
        myParams.blocksize = 0 # 0-3 to indicate which blocktype
        # add in scale factor band information
        myParams.sfBands =sfBands
        # start w/o all zeroes as data from prior block to overlap-and-add for output
        overlapAndAdd = []
        for iCh in range(nChannels): overlapAndAdd.append( np.zeros(nMDCTLines, dtype=np.float64) )
        myParams.overlapAndAdd=overlapAndAdd
        return myParams


    def ReadDataBlock(self, codingParams):
        """
        Reads a block of coded data from a PACFile object that has already
        executed OpenForReading() and returns those samples as reconstituted
        signed-fraction data
        """
        # loop over channels (whose coded data are stored separately) and read in each data block
        data=[]
        for iCh in range(codingParams.nChannels):
            data.append(np.array([],dtype=np.float64))  # add location for this channel's data
            # read in string containing the number of bytes of data for this channel (but check if at end of file!)
            s=self.fp.read(calcsize("<L"))  # will be empty if at end of file
            if not s:
                # hit last block, see if final overlap and add needs returning, else return nothing
                if codingParams.overlapAndAdd:
                    overlapAndAdd=codingParams.overlapAndAdd
                    codingParams.overlapAndAdd=0  # setting it to zero so next pass will just return
                    return overlapAndAdd
                else:
                    return
            # not at end of file, get nBytes from the string we just read
            nBytes = unpack("<L",s)[0] # read it as a little-endian unsigned long
            # read the nBytes of data into a PackedBits object to unpack
            pb = PackedBits()
            pb.SetPackedData( self.fp.read(nBytes) ) # PackedBits function SetPackedData() converts strings to internally-held array of bytes
            if pb.nBytes < nBytes:  raise "Only read a partial block of coded PACFile data"

            # extract the data from the PackedBits object
            # get BlockType / Size data
            codingParams.blocksize = pb.ReadBits(2)
            overallScaleFactor = pb.ReadBits(codingParams.nScaleBits)  # overall scale factor
            scaleFactor=[]
            bitAlloc=[]
            if(codingParams.blocksize == 0):
                codingParams.nMDCTLines = LONGBLOCKSIZE/2
            elif(codingParams.blocksize % 2 == 1):
                codingParams.nMDCTLines = (LONGBLOCKSIZE+SHORTBLOCKSIZE)/4
            elif(codingParams.blocksize == 2):
                codingParams.nMDCTLines = SHORTBLOCKSIZE/2

            #print codingParams.nMDCTLines
            mantissa=np.zeros(codingParams.nMDCTLines,np.int32)  # start w/ all mantissas zero

            # Determine ScaleFactorBands for blocksize
            if(codingParams.blocksize != 2):
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                             codingParams.sampleRate))
            else:
                codingParams.sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                             codingParams.sampleRate, shortFreqLimits))
            #print codingParams.sfBands.nLines
            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                ba = pb.ReadBits(codingParams.nMantSizeBits)
                if ba: ba+=1  # no bit allocation of 1 so ba of 2 and up stored as one less
                bitAlloc.append(ba)  # bit allocation for this band
                scaleFactor.append(pb.ReadBits(codingParams.nScaleBits))  # scale factor for this band
                if bitAlloc[iBand]:
                    # if bits allocated, extract those mantissas and put in correct location in matnissa array
                    m=np.empty(codingParams.sfBands.nLines[iBand],np.int32)
                    for j in range(m.size):
                        m[j]=pb.ReadBits(bitAlloc[iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so encoded as 1 lower than actual allocation
                    mantissa[codingParams.sfBands.lowerLine[iBand]:(codingParams.sfBands.upperLine[iBand]+1)] = m
            # done unpacking data (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can unpack any custom data passed in the nBytes of data >
            # Grab each spectral envelope value and dequantize
            for i in range(len(codingParams.specEnv[iCh])):
                envScale = pb.ReadBits(codingParams.nScaleBits)
                envMant = pb.ReadBits(codingParams.nScaleBits) # Sitcking with 4
                # Dequantize this band of spectral envelope
                codingParams.specEnv[iCh][i] = codec.DequantizeFP(envScale,envMant,\
                                codingParams.nScaleBits,codingParams.nScaleBits)

            # (DECODE HERE) decode the unpacked data for this channel, overlap-and-add first half, and append it to the data array (saving other half for next overlap-and-add)
            decodedData = self.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,iCh)
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

            data[iCh] = np.concatenate( (data[iCh],np.add(codingParams.overlapAndAdd[iCh],decodedData[:a]) ) )  # data[iCh] is overlap-and-added data
            codingParams.overlapAndAdd[iCh] = decodedData[a:]  # save other half for next pass
        # end loop over channels, return signed-fraction samples for this block
        return data


    def WriteFileHeader(self,codingParams):
        """
        Writes the PAC file header for a just-opened PAC file and uses codingParams
        attributes for the header data.  File pointer ends at start of data portion.
        """
        # write a header tag
        self.fp.write(self.tag)
        # make sure that the number of samples in the file is a multiple of the
        # number of MDCT half-blocksize, otherwise zero pad as needed
        if not codingParams.numSamples%codingParams.nMDCTLines:
            codingParams.numSamples += (codingParams.nMDCTLines
                        - codingParams.numSamples%codingParams.nMDCTLines) # zero padding for partial final PCM block

        # # also add in the delay block for the second pass w/ the last half-block (JH: I don't think we need this, in fact it generates a click at the end)
        # codingParams.numSamples+= codingParams.nMDCTLines  # due to the delay in processing the first samples on both sides of the MDCT block

        # write the coded file attributes
        self.fp.write(pack('<LHLLHH',
            codingParams.sampleRate, codingParams.nChannels,
            codingParams.numSamples, codingParams.nMDCTLines,
            codingParams.nScaleBits, codingParams.nMantSizeBits  ))
        # TODO: add param for blockSize switching used
        # create a ScaleFactorBand object to be used by the encoding process and write its info to header
        sfBands = ScaleFactorBands( AssignMDCTLinesFromFreqLimits(codingParams.nMDCTLines,
                                                                codingParams.sampleRate))
        codingParams.sfBands = sfBands
        self.fp.write(pack('<L',sfBands.nBands))
        self.fp.write(pack('<'+str(sfBands.nBands)+'H',*(sfBands.nLines.tolist()) ))
        # start w/o all zeroes as prior block of unencoded data for other half of MDCT block
        priorBlock = []
        for iCh in range(codingParams.nChannels):
            priorBlock.append(np.zeros(codingParams.nMDCTLines,dtype=np.float64) )
        codingParams.priorBlock = priorBlock
        return


    def WriteDataBlock(self,data, codingParams):
        """
        Writes a block of signed-fraction data to a PACFile object that has
        already executed OpenForWriting()"""

        # combine this block of multi-channel data w/ the prior block's to prepare for MDCTs twice as long
        fullBlockData=[]
        for iCh in range(codingParams.nChannels):
            fullBlockData.append( np.concatenate( ( codingParams.priorBlock[iCh], data[iCh]) ) )
            codingParams.priorBlock[iCh] = np.copy(data[iCh])  # current pass's data is next pass's prior block data
        # (ENCODE HERE) Encode the full block of multi=channel data
        (scaleFactor,bitAlloc,mantissa, overallScaleFactor) = self.Encode(fullBlockData,codingParams)  # returns a tuple with all the block-specific info not in the file header

        # for each channel, write the data to the output file
        for iCh in range(codingParams.nChannels):
            # determine the size of this channel's data block and write it to the output file
            nBytes = codingParams.nScaleBits  # bits for overall scale factor
            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to get its bits
                nBytes += codingParams.nMantSizeBits+codingParams.nScaleBits    # mantissa bit allocation and scale factor for that sf band
                if bitAlloc[iCh][iBand]:
                    # if non-zero bit allocation for this band, add in bits for scale factor and each mantissa (0 bits means zero)
                    nBytes += bitAlloc[iCh][iBand]*codingParams.sfBands.nLines[iBand]  # no bit alloc = 1 so actuall alloc is one higher
            # end computing bits needed for this channel's data

            # CUSTOM DATA:
            # < now can add space for custom data, if desired>
            # add two bits for block size if using blockswitching
            #if (BSFlag):
            nBytes += 2  # for blocksize ID
            # Bits for spectral envelope of each channel
            nBytes += codingParams.nSpecEnvBits*len(codingParams.specEnv[iCh])

            # now convert the bits to bytes (w/ extra one if spillover beyond byte boundary)
            if nBytes%BYTESIZE==0:  nBytes /= BYTESIZE
            else: nBytes = nBytes/BYTESIZE + 1
            self.fp.write(pack("<L",int(nBytes))) # stores size as a little-endian unsigned long

            # create a PackedBits object to hold the nBytes of data for this channel/block of coded data
            pb = PackedBits()
            pb.Size(nBytes)

            # now pack the nBytes of data into the PackedBits object
            # NEW: Block size data
            pb.WriteBits(codingParams.blocksize,2)
            pb.WriteBits(overallScaleFactor[iCh],codingParams.nScaleBits)  # overall scale factor
            iMant=0  # index offset in mantissa array (because mantissas w/ zero bits are omitted)
            for iBand in range(codingParams.sfBands.nBands): # loop over each scale factor band to pack its data
                ba = bitAlloc[iCh][iBand]
                if ba: ba-=1  # if non-zero, store as one less (since no bit allocation of 1 bits/mantissa)
                pb.WriteBits(ba,codingParams.nMantSizeBits)  # bit allocation for this band (written as one less if non-zero)
                pb.WriteBits(scaleFactor[iCh][iBand],codingParams.nScaleBits)  # scale factor for this band (if bit allocation non-zero)
                if bitAlloc[iCh][iBand]:
                    for j in range(codingParams.sfBands.nLines[iBand]):
                        pb.WriteBits(mantissa[iCh][iMant+j],bitAlloc[iCh][iBand])     # mantissas for this band (if bit allocation non-zero) and bit alloc <>1 so is 1 higher than the number
                    iMant += codingParams.sfBands.nLines[iBand]  # add to mantissa offset if we passed mantissas for this band
            # done packing (end loop over scale factor bands)

            # CUSTOM DATA:
            # < now can add in custom data if space allocated in nBytes above>
            for i in range(len(codingParams.specEnv[iCh])):
                envScale = codec.ScaleFactor(codingParams.specEnv[iCh][i],codingParams.nScaleBits,4)
                pb.WriteBits(envScale,codingParams.nScaleBits)
                # Hardcoding 4 Mantissa bits, using floating-point to quantize
                pb.WriteBits(codec.MantissaFP(codingParams.specEnv[iCh][i],envScale,codingParams.nScaleBits,4),codingParams.nScaleBits)

            # finally, write the data in this channel's PackedBits object to the output file
            self.fp.write(pb.GetPackedData())
        # end loop over channels, done writing coded data for all channels
        return

    def Close(self,codingParams):
        """
        Flushes the last data block through the encoding process (if encoding)
        and closes the audio file
        """
        # determine if encoding or encoding and, if encoding, do last block
        if self.fp.mode == "wb":  # we are writing to the PACFile, must be encode
            # we are writing the coded file -- pass a block of zeros to move last data block to other side of MDCT block
            data = [ np.zeros(codingParams.nMDCTLines,dtype=np.float),
                     np.zeros(codingParams.nMDCTLines,dtype=np.float) ]
            self.WriteDataBlock(data, codingParams)
        self.fp.close()


    def Encode(self,data,codingParams):
        """
        Encodes multichannel audio data and returns a tuple containing
        the scale factors, mantissa bit allocations, quantized mantissas,
        and the overall scale factor for each channel.
        """
        #Passes encoding logic to the Encode function defined in the codec module
        return codec.Encode(data,codingParams)

    def Decode(self,scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,iCh):
        """
        Decodes a single audio channel of data based on the values of its scale factors,
        bit allocations, quantized mantissas, and overall scale factor.
        """
        #Passes decoding logic to the Decode function defined in the codec module
        return codec.Decode(scaleFactor,bitAlloc,mantissa, overallScaleFactor,codingParams,iCh)








#-----------------------------------------------------------------------------

# Testing the full PAC coder (needs a file called "input.wav" in the code directory)
if __name__=="__main__":

    import sys
    import time
    from pcmfile import * # to get access to WAV file handling

    #TODO: Lowpass all data at cutoff, whole file or just block + adjascent blocks
    input_filename = "sbrTest.wav"
    coded_filename = "coded.pac"
    data_rate = 64000. # User defined data rate in bits/s/ch
    cutoff = 9500 # Global SBR cutoff
    output_filename = "sbrTest_" + str(int(data_rate/1000.)) + "kbps" + str(cutoff) + "Hz.wav"
    nSpecEnvBits = 8 # number of bits per spectral envelope band
    doSBR = True

    if len(sys.argv) > 1:
        input_filename = sys.argv[1]
        coded_filename = sys.argv[1][:-4] + ".pac"
        output_filename = sys.argv[1][:-4] + "_decoded.wav"


    print "\nRunning the PAC coder ({} -> {} -> {}):".format(input_filename, coded_filename, output_filename)
    elapsed = time.time()

    for Direction in ("Encode", "Decode"):
 #   for Direction in ("Decode"):

        # create the audio file objects
        if Direction == "Encode":
            print "\n\tEncoding PCM file ({}) ...".format(input_filename),
            inFile= PCMFile(input_filename)
            outFile = PACFile(coded_filename)
        else: # "Decode"
            print "\n\tDecoding PAC file ({}) ...".format(coded_filename),
            inFile = PACFile(coded_filename)
            outFile= PCMFile(output_filename)
        # only difference is file names and type of AudioFile object

        # open input file
        codingParams=inFile.OpenForReading()  # (includes reading header)

        # pass parameters to the output file
        if Direction == "Encode":
            # set additional parameters that are needed for PAC file
            # (beyond those set by the PCM file on open)
            codingParams.nMDCTLines = 1024
            codingParams.nScaleBits = 4
            codingParams.nMantSizeBits = 4
            codingParams.prevPE = 10
            # tell the PCM file how large the block size is
            codingParams.nSamplesPerBlock = codingParams.nMDCTLines
            # SBR related stuff
            codingParams.sbrCutoff = cutoff # Specified in Hz
            codingParams.doSBR = doSBR # For toggling SBR algorithm
            codingParams.nSpecEnvBits = nSpecEnvBits # Bits per band in spectral envelope
            codingParams.specEnv  = np.zeros((codingParams.nChannels,24-codec.freqToBand(codingParams.sbrCutoff)))

        else: # "Decode"
            # set PCM parameters (the rest is same as set by PAC file on open)
            codingParams.bitsPerSample = 16
        # only difference is in setting up the output file parameters


        # open the output file
        outFile.OpenForWriting(codingParams) # (includes writing header)

        # Read the input file and pass its data to the output file to be written
        firstBlock = True  # when de-coding, we won't write the first block to the PCM file. This flag signifies that
        nextData = []
        data = [[] for x in range(codingParams.nChannels)]
        while True:

            # if blocksize = 3(long-short), only possible next blocksize = 2(short-short)
            #print "priorBlock top of loop: ",len(codingParams.priorBlock[0])


            # only read new large block if previous block was long or nextData used up

            # If encoding set blocksize, detect transients if new block, set K for targetBits, determine if
            # new data or nextData used
            if(Direction == "Encode"):
                newBlock = False
                if(codingParams.blocksize < 2 or (codingParams.blocksize > 1 and len(nextData[0]) < SHORTBLOCKSIZE/2)):
                    nextData = inFile.ReadDataBlock(codingParams)
                    newBlock = True
                    if not nextData: break # end of file
                    # detect Transient, set next window shape
                    # 0 = long long, 1 = short long, 2 = short short, 3 = long short
                # Maybe add >2 channel transient detection
                if(codingParams.blocksize == 3):
                    codingParams.blocksize = 2
                    K = codingParams.nMDCTLines*(2* float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                elif (len(data[0]) != 0 and newBlock and (DetectTransient(nextData[0],codingParams) or DetectTransient(nextData[1], codingParams))):
                    if(codingParams.blocksize < 2):
                        codingParams.blocksize = 3
                        K = codingParams.nMDCTLines*(1 + float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                    else:
                        codingParams.blocksize = 2;
                        K = codingParams.nMDCTLines*(2* float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                elif (newBlock):
                    if(codingParams.blocksize < 2):
                        codingParams.blocksize = 0
                        K = codingParams.nMDCTLines*2
                    else:
                        codingParams.blocksize = 1
                        K = codingParams.nMDCTLines*(1 + float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                # get correct amount of sample data
                if(codingParams.blocksize > 1):
                    for iCh in range(codingParams.nChannels):
                        data[iCh] = nextData [iCh][:SHORTBLOCKSIZE/2]
                        nextData[iCh] = nextData [iCh][SHORTBLOCKSIZE/2:]
                else:
                    data = nextData

                #print "priorBlock data set: ",len(codingParams.priorBlock[0])
                if(codingParams.blocksize != 2):
                    sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(K/2, codingParams.sampleRate))
                    codingParams.doSBR = True # Not short block, we do SBR
                    codingParams.sfBands=sfBands

                    # set targetBitsPerSample
                    codingParams.targetBitsPerSample = (((data_rate/codingParams.sampleRate)*K)- \
                     (6+codingParams.sfBands.nBands*(codingParams.nScaleBits+codingParams.nMantSizeBits)+\
                     codingParams.nSpecEnvBits*len(codingParams.specEnv)))/K
                else:
                    sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(K/2, codingParams.sampleRate, shortFreqLimits))
                    codingParams.doSBR = False
                    codingParams.sfBands=sfBands
                    # set targetBitsPerSample
                    codingParams.targetBitsPerSample = (((data_rate/codingParams.sampleRate)*\
                            K) - (6+codingParams.sfBands.nBands*\
                            (codingParams.nScaleBits+codingParams.nMantSizeBits)))/K
            else:
                data = inFile.ReadDataBlock(codingParams)

            if not data:
                break # we hit the end of the input file
            # don't write the first PCM block (it corresponds to the half-block delay introduced by the MDCT)
            if firstBlock and Direction == "Decode":
                firstBlock = False
                continue
            outFile.WriteDataBlock(data,codingParams)
            sys.stdout.write(".")  # just to signal how far we've gotten to user
            sys.stdout.flush()

        # end loop over reading/writing the blocks

        # close the files
        inFile.Close(codingParams)
        outFile.Close(codingParams)
    # end of loop over Encode/Decode

    elapsed = time.time()-elapsed
    print "\nDone with Encode/Decode test\n"
    print elapsed ," seconds elapsed"
