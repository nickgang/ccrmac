#!/usr/bin/env python
"""
A simple user interface for the Music 422 PAC Coder

-----------------------------------------------------------------------
© 2009 Marina Bosi & Richard E. Goldberg -- All rights reserved
-----------------------------------------------------------------------
"""

# Uses the wxPython library for the interface
import wx
import os
from pcmfile import * # to get access to PCM files
from pacfile import * # to get access to perceptually coded files


class MyFrame(wx.Frame):
    def __init__(self, *args, **kwds):
        kwds["style"] = wx.DEFAULT_FRAME_STYLE
        wx.Frame.__init__(self, *args, **kwds)
        self.notebook_1 = wx.Notebook(self, -1, style=0)
        self.notebook_1_pane_2 = wx.Panel(self.notebook_1, -1)
        self.notebook_1_pane_1 = wx.Panel(self.notebook_1, -1)
        self.label_8 = wx.StaticText(self.notebook_1_pane_1, -1, "Input File")
        self.inFile = wx.TextCtrl(self.notebook_1_pane_1, -1, "")
        self.GetInputFilename = wx.Button(self.notebook_1_pane_1, -1, "...")
        self.label_9 = wx.StaticText(self.notebook_1_pane_1, -1, "Append to create output file name")
        self.outFileAppend = wx.TextCtrl(self.notebook_1_pane_1, -1, "_decoded_128kbps_hbr")
        self.goButton = wx.Button(self.notebook_1_pane_1, -1, "Go")
        self.label_7 = wx.StaticText(self.notebook_1_pane_1, -1, "Encode Progress")
        self.encodeGauge = wx.Gauge(self.notebook_1_pane_1, -1, 100)
        self.label_6 = wx.StaticText(self.notebook_1_pane_1, -1, "Decode Progress")
        self.decodeGauge = wx.Gauge(self.notebook_1_pane_1, -1, 100)
        self.label_1 = wx.StaticText(self.notebook_1_pane_2, -1, "Number of MDCT Lines (1/2 Block)")
        self.nMDCTLines = wx.TextCtrl(self.notebook_1_pane_2, -1, "1024")
        self.label_10 = wx.StaticText(self.notebook_1_pane_2, -1, "Short Block Half Size")
        self.shortBlockSize = wx.TextCtrl(self.notebook_1_pane_2, -1, "128")
        self.label_2 = wx.StaticText(self.notebook_1_pane_2, -1, "Number of Scale Factor Bits")
        self.nScaleBits = wx.TextCtrl(self.notebook_1_pane_2, -1, "4")
        self.label_4 = wx.StaticText(self.notebook_1_pane_2, -1, "Number of Mantissa Size Bits")
        self.nMantSizeBits = wx.TextCtrl(self.notebook_1_pane_2, -1, "4")
        self.label_11 = wx.StaticText(self.notebook_1_pane_2, -1, "SBR Cutoff Frequency")
        self.SBRCutoff = wx.TextCtrl(self.notebook_1_pane_2, -1, "5300")
        #self.label_12 = wx.StaticText(self.notebook_1_pane_2, -1, "Enable SBR")
        self.enableSBR = wx.CheckBox(self.notebook_1_pane_2, -1, "Low Bit-Rate Encoding (< 96000 kbps)")
        self.enableBS = wx.CheckBox(self.notebook_1_pane_2, -1, "Enable Block Switching")
        #self.label_12 = wx.StaticText(self.notebook_1_pane_2, -1, "Coupling Cutoff Frequency")
        #self.CPLCutoff = wx.TextCtrl(self.notebook_1_pane_2, -1, "3700")
        self.enableCPL = wx.CheckBox(self.notebook_1_pane_2, -1, "Enable Coupling")
        self.label_5 = wx.StaticText(self.notebook_1_pane_2, -1, "Target Bit Rate (bits per second per channel)")
        self.dataRatePerChannel = wx.TextCtrl(self.notebook_1_pane_2, -1, "128000")

        self.__set_properties()
        self.__do_layout()

        self.Bind(wx.EVT_BUTTON, self.SetInputFile, self.GetInputFilename)
        self.Bind(wx.EVT_BUTTON, self.DoCoding, self.goButton)

    def __set_properties(self):
        self.SetTitle("PAC Coder")
        self.GetInputFilename.SetMinSize((30, 30))

    def __do_layout(self):
        sizer_1 = wx.BoxSizer(wx.VERTICAL)
        grid_sizer_1 = wx.GridSizer(8, 2, 2, 2)
        sizer_2 = wx.BoxSizer(wx.VERTICAL)
        sizer_3 = wx.BoxSizer(wx.VERTICAL)
        sizer_5 = wx.BoxSizer(wx.VERTICAL)
        sizer_4 = wx.BoxSizer(wx.HORIZONTAL)
        sizer_3.Add(self.label_8, 0, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_4.Add(self.inFile, 1, wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_4.Add(self.GetInputFilename, 0, wx.ALIGN_RIGHT, 0)
        sizer_3.Add(sizer_4, 0, wx.EXPAND, 0)
        sizer_5.Add(self.label_9, 0, 0, 0)
        sizer_5.Add(self.outFileAppend, 0, wx.EXPAND|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_3.Add(sizer_5, 0, wx.EXPAND, 0)
        sizer_3.Add(self.goButton, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(sizer_3, 0, wx.EXPAND, 0)
        sizer_2.Add(self.label_7, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(self.encodeGauge, 0, wx.EXPAND, 0)
        sizer_2.Add(self.label_6, 0, wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_CENTER_VERTICAL, 0)
        sizer_2.Add(self.decodeGauge, 0, wx.EXPAND, 0)
        self.notebook_1_pane_1.SetSizer(sizer_2)
        grid_sizer_1.Add(self.label_1, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.nMDCTLines, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.label_10, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.shortBlockSize, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.label_2, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.nScaleBits, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.label_4, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.nMantSizeBits, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)  
        grid_sizer_1.Add(self.label_5, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.dataRatePerChannel, 1, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.label_11, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.SBRCutoff, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.enableSBR, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.enableBS, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        grid_sizer_1.Add(self.enableCPL, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        #grid_sizer_1.Add(self.label_12, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        #grid_sizer_1.Add(self.CPLCutoff, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 0)
        
        
        self.notebook_1_pane_2.SetSizer(grid_sizer_1)
        self.notebook_1.AddPage(self.notebook_1_pane_1, "Select Input File and Run")
        self.notebook_1.AddPage(self.notebook_1_pane_2, "Change Coding Parameters")
        sizer_1.Add(self.notebook_1, 1, wx.EXPAND, 0)
        self.SetSizer(sizer_1)
        sizer_1.Fit(self)
        self.Layout()

    def SetInputFile(self,event):
        """ picks the input WAV file"""

        # file type specifier -- only allow WAV files right now
        wildcard = "WAV file (*.wav)|*.wav"

        # create the dialog
        dlg = wx.FileDialog(
            self, message="Choose an input file",
            defaultDir=os.getcwd(),
            defaultFile="",
            wildcard=wildcard,
            style=wx.OPEN | wx.CHANGE_DIR
            )

        # Show the dialog and retrieve the user response. If it is the OK response,
        # process the data.
        if dlg.ShowModal() == wx.ID_OK:
            # User didn't cancel, set input filename
            self.inFile.Value=dlg.GetFilename()
        dlg.Destroy()


    def CheckInputs(self):
        # check that the input values are acceptable and, if so, return True
        # check inputs, give message and return False if they are not
        inFilename=self.inFile.GetValue()
        if not os.path.exists(inFilename):
            dlg = wx.MessageDialog(self, 'Input file does not exist - please check it.',
                       'Input Error!',
                       wx.OK | wx.ICON_ERROR
                       )
            dlg.ShowModal()
            dlg.Destroy()
            return False
        return True

    def DoCoding(self, event):
        """The main control of the encode/decode process"""
#old
        #import sys
        #import time
        # start w/ progress bars at zero percent
        self.encodeGauge.SetValue(0)
        self.decodeGauge.SetValue(0)

        # if inputs are OK, carry out coding
        if self.CheckInputs():
            self.goButton.Disable() # can't push go till done
            self.goButton.SetLabel("Coding...")

            # get info from GUI widgets
            inFilename=self.inFile.GetValue()
            outFilename=inFilename.replace(".wav",self.outFileAppend.Value+".wav")
            codeFilename=outFilename.replace(".wav",".pac")
            nMDCTLines = int(self.nMDCTLines.GetValue())
            nScaleBits = int(self.nScaleBits.GetValue())
            nMantSizeBits = int(self.nMantSizeBits.GetValue())
            #targetBitsPerSample = float(self.targetBitsPerSample.GetValue())
            # get data rate
            data_rate = float(self.dataRatePerChannel.GetValue())
            #cutoff = 5300 # Global SBR cutoff
            cutoff = int(self.SBRCutoff.GetValue())
            #couplingFrequency = 3700
            #couplingFrequency = int(self.CPLCutoff.GetValue())
            shortBlockSize = int(self.shortBlockSize.GetValue())*2
            if (shortBlockSize < 256):
                shortBlockSize = 256
            
            LONGBLOCKSIZE = nMDCTLines * 2
            if (shortBlockSize > LONGBLOCKSIZE):
                shortBlockSize = LONGBLOCKSIZE
            
            #output_filename = input_filename[:-4] + "_12_6" + str(int(data_rate/1000.)) + "kbps" + str(cutoff) + "Hz.wav"
            #coded_filename = output_filename[:-4] + ".pac"
            nSpecEnvBits = 8 # number of bits per spectral envelope band
            doSBR = bool(self.enableSBR.GetValue())
            doCoupling = bool(self.enableCPL.GetValue())
            doBS = bool(self.enableBS.GetValue())
            SHORTBLOCKSIZE = shortBlockSize
            
            # encode and then decode the selected input file
            for Direction in ("Encode", "Decode"):

                # create the audio file objects
                if Direction == "Encode":
                    inFile= PCMFile(inFilename)
                    outFile = PACFile(codeFilename)
                else: # "Decode"
                    inFile = PACFile(codeFilename)
                    outFile= PCMFile(outFilename)

                # open input file
                codingParams=inFile.OpenForReading()
                
                nBlocks = codingParams.numSamples/nMDCTLines #roughly number of blocks to process
                

                # pass parameters to the output file
                if Direction == "Encode":
                    # set additional parameters that are needed for PAC file
                    # (beyond those set by the PCM file on open)
                    
                    if doBS:
                        codingParams.doBS = True
                    else:
                        codingParams.doBS = False
                    codingParams.nMDCTLines = nMDCTLines
                    codingParams.nScaleBits = nScaleBits
                    codingParams.nMantSizeBits = nMantSizeBits
                    codingParams.shortBlockSize = SHORTBLOCKSIZE
                    codingParams.longBlockSize = LONGBLOCKSIZE
                    #codingParams.targetBitsPerSample = targetBitsPerSample
                    # tell the PCM file how large the block size is
                    #codingParams.nSamplesPerBlock = codingParams.nMDCTLines
                    
                    #codingParams.longBlockSize = nMDCTLines * 2
                    #codingParams.nScaleBits = 4
                    #codingParams.nMantSizeBits = 4
                    codingParams.prevPE = 10
                    codingParams.blocksize = 0
                    codingParams.bitReservoir = 0
                   
                    #codingParams.shortBlockSize = SHORTBLOCKSIZE
                    # tell the PCM file how large the block size is
                    codingParams.nSamplesPerBlock = nMDCTLines
                    # SBR related stuff
                    codingParams.sbrCutoff = cutoff # Specified in Hz
                    codingParams.doSBR = doSBR # For toggling SBR algorithm
                    codingParams.nSpecEnvBits = nSpecEnvBits # Bits per band in spectral envelope
                    codingParams.specEnv  = np.zeros((codingParams.nChannels,int(24-codec.freqToBand(codingParams.sbrCutoff))))
        
                    codingParams.doCoupling = doCoupling
                    codingParams.nCouplingStart = 20

                else: # "Decode"
                    # set PCM parameters (the rest is same as set by PAC file on open)
                    codingParams.bitsPerSample = 16

                # open the output file
                outFile.OpenForWriting(codingParams) # (includes writing header)
                
                

                # Read the input file and pass its data to the output file to be written
                iBlock=0  # current block
                if Direction=="Encode": gauge=self.encodeGauge
                else: gauge=self.decodeGauge
                
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
                            transData = inFile.ReadTransientTestBlock(codingParams)
                            nextData = inFile.ReadDataBlock(codingParams)
                            newBlock = True
                            if not nextData: break # end of file
                            # detect Transient, set next window shape
                            # 0 = long long, 1 = short long, 2 = short short, 3 = long short
                        # Maybe add >2 channel transient detection
                        if(codingParams.blocksize == 3):
                            codingParams.blocksize = 2
                            codingParams.nMDCTLines = (SHORTBLOCKSIZE)/2
                            #K = codingParams.nMDCTLines*(2* float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                            # assumes stereo input file
                        elif (len(data[0]) != 0 and newBlock and (DetectTransient(transData[0],codingParams) or DetectTransient(transData[1], codingParams))):
                            if(codingParams.blocksize < 2):
                                codingParams.blocksize = 3
                                codingParams.nMDCTLines = (SHORTBLOCKSIZE + LONGBLOCKSIZE)/4
                                #K = codingParams.nMDCTLines*(1 + float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                            else:
                                codingParams.blocksize = 2;
                                codingParams.nMDCTLines = (SHORTBLOCKSIZE)/2
                                #K = codingParams.nMDCTLines*(2* float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                        elif (newBlock):
                            if(codingParams.blocksize < 2):
                                codingParams.blocksize = 0
                                codingParams.nMDCTLines = LONGBLOCKSIZE/2
                                #K = codingParams.nMDCTLines*2
                            else:
                                codingParams.blocksize = 1
                                codingParams.nMDCTLines = (SHORTBLOCKSIZE + LONGBLOCKSIZE)/4
                                #K = codingParams.nMDCTLines*(1 + float(SHORTBLOCKSIZE)/LONGBLOCKSIZE)
                        # get correct amount of sample data
                        if(codingParams.blocksize > 1):
                            for iCh in range(codingParams.nChannels):
                                data[iCh] = nextData [iCh][:SHORTBLOCKSIZE/2]
                                nextData[iCh] = nextData [iCh][SHORTBLOCKSIZE/2:]
                        else:
                            data = nextData
                        K = 2 * codingParams.nMDCTLines
                        # db print "priorBlock data set: ",len(codingParams.priorBlock[0])
                        sfBands = ScaleFactorBands(AssignMDCTLinesFromFreqLimits(K/2, codingParams.sampleRate))
                        codingParams.sfBands=sfBands
        
                        # set targetBitsPerSample
                        codingParams.targetBitsPerSample = (((data_rate/codingParams.sampleRate)*K)- \
                         (6+codingParams.sfBands.nBands*(codingParams.nScaleBits+codingParams.nMantSizeBits)+\
                         codingParams.nSpecEnvBits*len(codingParams.specEnv)))/K
                        # # boost bit budget for short blocks
                        # if(codingParams.blocksize==2):codingParams.targetBitsPerSample *= SHORTBLOCKBITBOOST
                    else: # decoding
                        data = inFile.ReadDataBlock(codingParams)
        
                    if not data:
                        break # we hit the end of the input file
                    # don't write the first PCM block (it corresponds to the half-block delay introduced by the MDCT)
                    if firstBlock and Direction == "Decode":
                        firstBlock = False
                        continue
                    outFile.WriteDataBlock(data,codingParams)
                    iBlock +=1
                    gauge.SetValue(100*iBlock/nBlocks)  # set new value
                    gauge.Refresh()     # make sure it knows to refresh
                    wx.GetApp().Yield(True)
                    
        
                # end loop over reading/writing the blocks
            

                # close the files
                inFile.Close(codingParams)
                outFile.Close(codingParams)
                gauge.SetValue(100)
            # end of loop over Encode/Decode

            # we're done - give user GUI control and tell them we're done
            self.goButton.Enable() # allow access again now
            self.goButton.SetLabel("Go")
            dlg = wx.MessageDialog(self, 'File has been encoded and then decoded!',
                                   'Done Coding',
                                   wx.OK | wx.ICON_INFORMATION
                                   )
            dlg.ShowModal()
            dlg.Destroy()
        # end codingpass for OK inputs,end of function

            
        
            



class MyApp(wx.App):
    def OnInit(self):
        wx.InitAllImageHandlers()
        PACCoderGUI = MyFrame(None, -1, "")
        PACCoderGUI.Center(wx.BOTH)
        self.SetTopWindow(PACCoderGUI)
        PACCoderGUI.Show()
        return 1

if __name__ == "__main__":
    app = MyApp(0)
    app.MainLoop()
