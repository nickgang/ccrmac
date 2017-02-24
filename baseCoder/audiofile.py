"""
audiofile.py -- Abstract AudioFile class definition for audio file read/write.
Data is converted to/from arrays of signed-fraction data (i.e. floating point
numbers between -1.0 and 1.0) as an intermediary data format.
-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------

Any audio file format should inherit from this class and be set up to override
the following methods:

    ReadFileHeader()
    WriteFileHeader()
    ReadDataBlock()
    WriteDataBlock()

The Close() method will also need to be overridden to handle any extra data
that the coding scheme requires passed after the last data block has been processed.
(For example, MDCT-based approaches need to pass the last block of data through
a second encode pass to avoid time-domain aliasing).

Example usage (using generic AudioFile class objects):

    # create the audio file objects of the appropriate AudioFile type
    inFile= AudioFile(inputFilename)
    outFile = AudioFile(outputFileName)

    # open input file and get its coding parameters
    codingParams= inFile.OpenForReading()

    # set additional coding parameters that are needed for encoding/decoding
     codingParams.myParam = myParamValue

    # open the output file for writing, passing needed format/data parameters
    outFile.OpenForWriting(codingParams)

    # Read the input file and pass its data to the output file to be written
    while True:
        data=inFile.ReadDataBlock(codingParams)
        if not data: break  # we hit the end of the input file
        outFile.WriteDataBlock(data,codingParams)
    # end loop over reading/writing the blocks

    # close the files (and do any necessary end-of-coding cleanup)
    inFile.Close(codingParams)
    outFile.Close(codingParams)

"""


class CodingParams:
    """A class to hold coding parameters to share across files"""
    pass # will just add attributes at runtime as needed


class AudioFile:
    """An abstract class defining handlers expected for a data file containing audio data"""

    def __init__(self, filename):
        """Object is initialized with its filename"""
        self.filename = filename

    def OpenForReading(self):
        """Opens the file for reading input data, extracts any file header, and returns a CodingParams object w/ data from file header as attributes"""
        self.fp = open(self.filename,"rb")
        codingParams =self.ReadFileHeader()  # this leaves the file pointer at the start of data and returns a CodingParams object w/ data from header
        return codingParams

    def OpenForWriting(self,codingParams):
        """Opens the file for writing output data and writes the file Header (getting info from passed CodingParams object attributes as needed)"""
        self.fp = open(self.filename,"wb")
        self.WriteFileHeader(codingParams)  # this writes the file header and leaves the file pointer at the start of data portion

    def Close(self,codingParams):
        """Closes the audio file and does any needed end-of-coding steps"""
        self.fp.close()

    def ReadFileHeader(self):
        """Reads the file header from a just-opened audio file and uses it to set object attributes.  File pointer ends at start of data portion."""
        return CodingParams() # default is to return an empty CodingParams object

    def ReadDataBlock(self,codingParams):
        """Reads the next block of audio data from an audio file that has already executed OpenForReading and returns those samples as signed-fraction data"""
        pass  # default is do nothing

    def WriteFileHeader(self,codingParams):
        """Writes the audio file header to a just-opened audio file and uses data in passed CodingParams object for the header data.  File pointer ends at start of data portion."""
        pass  # default is do nothing

    def WriteDataBlock(self,data,codingParams):
        """Writes the next block of signed-fraction data to an audio file that has already executed OpenForWriting"""
        pass  # default is do nothing

