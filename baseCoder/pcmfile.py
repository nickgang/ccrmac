"""
pcmfile.py -- Defines a PCMFile class to handle reading and writing audio
data to 16-bit PCM WAV audio files.  The class is a subclass of AudioFile.
-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

See the documentation of the AudioFile class for general use of this class.

The OpenFileForReading() function returns a CodedParams object containing
nChannels, bitsPerSample, sampleRate, and numSamples as attributes (where
numSamples is the number of samples for each channel in the file).  Before
using the ReadDataBlock() function, the CodingParameters object should be
given an attribute called nSamplesPerBlock.

When writing to a PCM file the CodingParameters object passed to OpenForWriting()
(and subsequently passed to WriteDataBlock()) should have attributes called
nChannels, bitsPerSample, sampleRate, and numSamples (where numSamples is the
number of samples in the file for each channel).
"""

from audiofile import *  # base class of PCMFile
from struct import *  #allows easy packing/unpacking of an array of bytes into larger (multi-byte) data items
import numpy as np  # to allow conversion of data blocks to numpy's array object
from quantize import vQuantizeUniform, vDequantizeUniform # to convert to/from 16-bit PCM

BYTESIZE = 8  # bits per byte

class PCMFile(AudioFile):
    """Handlers for a PCM file containing audio data in 16-bit PCM WAV file format"""

    def ReadFileHeader(self):
        """Reads the WAV file header from a just-opened WAV file and uses it to set object attributes.  File pointer ends at start of data portion."""
        # read RIFF file header (which says it is a WAV format file)
        tag = self.fp.read(12)
        if tag[0:4] != "RIFF" or tag[8:12]!= "WAVE":  raise "ERROR: File opened for PCMFile is not a RIFF file!"
        # find format chunk
        while True:
             tag=self.fp.read(4)
             if len(tag)<4: raise "ERROR: Didn't find WAV file 'fmt ' chunk following RIFF file header"
             if tag[0:4] == "fmt ":  break
        # found format chunk, read its data
        tag=self.fp.read(20)  # should be size of basic format chunk (following 'fmt ' header)
        (formatSize, formatTag, nChannels, sampleRate, \
              bytesPerSec, blockAlign, bitsPerSample) =  unpack("<LHHLLHH",tag)
        if formatTag != 1: raise Exception("Opened a non-PCM WAV file as a PCMFile")
        if bitsPerSample != 16: raise Exception("PCMFile was not 16-bits per sample")
        # find data chunk
        while True:
             tag=self.fp.read(4)
             if len(tag)<4: raise Exception("Didn't find WAV file 'data' chunk following 'fmt ' chunk")
             if tag[0:4] == "data":  break
        # found data chunk, read its data (leaving file pointer at start of real data)
        numSamples = unpack('<L',self.fp.read(4))[0]
        assert bitsPerSample//BYTESIZE == bitsPerSample/BYTESIZE, "PCMFile bitsPerSample was not integer number of bytes"
        numSamples /= nChannels * (bitsPerSample//BYTESIZE)  #assumes bitsPerSample is integer number of bytes
        # create a CodingParams object and pass header data to it as attributes
        myParams = CodingParams()
        myParams.nChannels = nChannels # number of channels (e.g. 2 for stereo)
        myParams.bitsPerSample = bitsPerSample # bits per audio sample (only 16 currently supported)
        myParams.sampleRate = sampleRate # sample rate in Hz (e.g. 44100.)
        myParams.numSamples = numSamples # total number of samples in file (per channel)
        myParams.bytesReadSoFar = 0
        return myParams

    def ReadDataBlock(self,codingParams):
        """Reads a block of data from a PCMFile object that has already executed OpenForReading and returns those samples as signed-fraction data"""
        # read a block of nSamplesPerBlock*nChannels*bytesPerSample bytes from the file (where nSamples is set by coding file before reading)
        bytesToRead = codingParams.nSamplesPerBlock*codingParams.nChannels*(codingParams.bitsPerSample/BYTESIZE)
        if codingParams.nChannels*codingParams.numSamples*(codingParams.bitsPerSample/BYTESIZE) - codingParams.bytesReadSoFar <= 0:
            dataBlock = None
        elif codingParams.nChannels*codingParams.numSamples*(codingParams.bitsPerSample/BYTESIZE) - codingParams.bytesReadSoFar < bytesToRead:
            dataBlock = self.fp.read(codingParams.nChannels*codingParams.numSamples*(codingParams.bitsPerSample/BYTESIZE) - codingParams.bytesReadSoFar)
        else:
            dataBlock = self.fp.read(bytesToRead)
        codingParams.bytesReadSoFar += bytesToRead
        if dataBlock and len(dataBlock)<bytesToRead:  # got data but not as much as expected
            # this was a partial block, zero pad
            dataBlock += (bytesToRead-len(dataBlock))*"\0"
        elif not dataBlock: return  # stop if nothing read
        # convert block of bytes into block of uniformly quantized codes, dequantize them, and parse into channels
        if codingParams.bitsPerSample == 16:
            # Uses '<h' format code in struct to convert little-endian pairs of bits into short integers
            dataBlock=unpack("<"+str(codingParams.nSamplesPerBlock*codingParams.nChannels)+"h",dataBlock)  # asumes nSamples*nChannels SIGNED short ints
#            dataBlock=np.fromstring(dataBlock,dtype=np.int16)  # uses Local Endian conversion -- use byteswap() method if wrong Endian
        else: raise Exception("PCMFile was not 16-bit PCM in PCMFile.ReadDataBlock!")
        # parse samples into channels and dequantize into signed-fraction floating point numbers
        data = [] # this is where the signed-fraction samples will reside for each channel
        for iCh in range(codingParams.nChannels):
            # slice out this channel's interleaved 16-bit PCM codes (and make sure it is a numpy array)
            codes = np.asarray(dataBlock[iCh::codingParams.nChannels])
            # extract signs
            signs = np.signbit(codes)
            codes[signs] *= -1  # now codes are positive
            # dequantize, return signs, and put result as data[iCh]
            temp = vDequantizeUniform(codes,16)
            temp[signs] *= -1.  # returns signs
            data.append(temp)  # data[iCh]
        # return data
        return data


    def WriteFileHeader(self,codingParams):
        """Writes the WAV file header to a just-opened WAV file and uses object attributes for the header data.  File pointer ends at start of data portion."""
        # prepare header data
        formatTag = 1 # PCM
        dataBytes = codingParams.numSamples*codingParams.nChannels*(codingParams.bitsPerSample/BYTESIZE) # bytes of data
        formatSize = 16  # bytes of format data
        chunkSize = 36 + dataBytes  # bytes of chunk data
        blockAlign = codingParams.nChannels*(codingParams.bitsPerSample/BYTESIZE)
        bytesPerSec = codingParams.sampleRate*codingParams.nChannels*(codingParams.bitsPerSample/BYTESIZE)
        # pack and write header data
        self.fp.write(  pack('<4sL4s4sLHHLLHH4sL', \
            "RIFF", chunkSize, "WAVE", "fmt ", formatSize, formatTag, codingParams.nChannels, codingParams.sampleRate, \
              bytesPerSec, blockAlign, codingParams.bitsPerSample, "data", dataBytes )  )


    def WriteDataBlock(self,data,codingParams):
        """Writes a block of signed-fraction data to a PCMFile object that has already executed OpenForWriting"""
        # get information about the block to write
        nChannels = len(data)
        if nChannels != codingParams.nChannels: raise Exception("Data block to PCMFile did not have expected number of channels")
        nSamples= min([len(data[iCh]) for iCh in range(nChannels)])  # use shortest length of channel data for nSamples
        bytesToWrite = nChannels*nSamples*(codingParams.bitsPerSample/BYTESIZE)
        # convert data to an array of 2s-complement uniformly quantized codes
        codes = []  # PCM quantized codes will go here
        for iCh in range(nChannels):
            temp = data[iCh]
            signs = np.signbit(temp)  # extract signs
            temp[signs] *= -1.  # now temp is positive
            temp = vQuantizeUniform(temp,16)   # returns 16 bit quantized codes (stored as unsigned ints)
            temp = temp.astype(np.int16)  # now it can take a 2s complement sign (it was unsigned before)
            temp[signs] *= -1  # now quantization code has 2s complement sign attached
            codes.append(temp)  # codes[iCh]
        # interleave the codes to be written out (because that's the WAV format)
        dataBlock = [codes[iCh][iSample] for iSample in xrange(nSamples) for iCh in range(nChannels) ]
        dataBlock = np.asarray(dataBlock, dtype = np.int16)  # notice that this is SIGNED 16-bit int
        # pack the interleaved codes into a block of bytes
        if codingParams.bitsPerSample == 16:
            # Uses '<h' format code in struct to convert short integers into little-endian pairs of bytes
 #           dataString = ""
 #           for i in range(len(dataBlock)):  dataString += pack('<h',dataBlock[i])
            dataString=dataBlock.tostring()  # converts (local Endian) data of dataBlock into a string to write  -- use byteswap() method if wrong Endian
        else: raise Exception("Asked to write to a PCM file with other than 16-bits per sample in PCMFile.WriteDataBlock!")
        # write those bytes to the file and return
        self.fp.write(dataString)
        return



# Testing the PCMFile class
if __name__=="__main__":

    # create the audio file objects of the appropriate audioFile type
    inFile= PCMFile("input.wav")
    outFile = PCMFile("output.wav")

    # open input file and get its coding parameters
    codingParams= inFile.OpenForReading()

    # set additional coding parameters that are needed for encoding/decoding
    codingParams.nSamplesPerBlock = 1024

    # open the output file for writing, passing needed format/data parameters
    outFile.OpenForWriting(codingParams)

    # Read the input file and pass its data to the output file to be written
    while True:
        data=inFile.ReadDataBlock(codingParams)
        if not data: break  # we hit the end of the input file
        outFile.WriteDataBlock(data,codingParams)
    # end loop over reading/writing the blocks

    # close the files
    inFile.Close(codingParams)
    outFile.Close(codingParams)


