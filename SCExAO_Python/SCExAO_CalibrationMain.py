'''
#-----Header-----#

Defines a class that stores all information obtained from
SCExAO calibration. This information can be used to obtain the
SCExAO model parameters. SCExAO_CalibrationPlots.Py plots
the results obtained in this file.

#-----Header-----#
'''

#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import pickle
import Methods as Mt
import SCExAO_Model

#--/--Imports--/--#

#-----Class-----#

class SCExAO_Calibration():

    def __init__(self,PolFileList,RotationFile):

        self.ImageAngle = 27 #Angle with which the images seem to be rotated
        self.LeftX = 34 #Position in pixels of border on the left
        self.RightX = 166 #position in pixels of border on the right
        self.BottomY = 34 #Position in pixels of border on the top
        self.TopY = 166 #Position in pixels of border on the left
        self.MiddleX = 100 #Position in pixels of line that separates measurements
        self.PixelOffset = 5

        self.MaxTimeDifference = 5*60

        self.HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]

        self.ColorList = ["blue","lightblue","red","orange"]

        #File names of text files where variables are stored
        self.PolParamValueArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Python/PickleFiles/PicklePolParamValueArray.txt"
        self.PolImrArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Python/PickleFiles/PicklePolImrArray.txt"
        

    def RunCalibration(self,PolFileList,RotationFile):
        '''
        Summary:     
            Runs the whole calibration process. Combines all the methods found below.
        '''
        print("Reading files...")
        self.PolImageList,self.PolLambdaList,self.PolTimeList = ReadCalibrationFiles(PolFileList)
        self.RotationTimeList,self.RotationImrList,self.RotationHwpList = ReadRotationFile(RotationFile)

        print("Finding Imr and Hwp angles of calibration images...")
        self.PolImrList,self.PolHwpList,self.PolBadImageList = self.GetRotations(self.PolTimeList)
        self.PolImageList = self.PolImageList[self.PolBadImageList==False]
        self.PolLambdaList = self.PolLambdaList[self.PolBadImageList==False]

        print("Splitting calibration images...")
        self.PolImageListL,self.PolImageListR = self.SplitCalibrationImages(self.PolImageList)

        print("Creating double difference images...")
        self.PolDDImageArray,self.PolDSImageArray,self.PolImrArray = self.CreateHwpDoubleDifferenceImges(self.PolHwpList,self.PolImrList,self.PolImageListL,self.PolImageListR)

        print("Getting double difference value...")
        self.PolParamValueArray = self.GetDoubleDifferenceValue(self.PolDDImageArray,self.PolDSImageArray)

    def SplitCalibrationImages(self,ImageList):
        '''
        Summary:     
            Rotates the images and splits them in a left and right part.
        Input:
            ImageList:  List of all calibration images.
                Has dimensions (325,22,201,201) --> (N,wavelength,x,y)
        Output:
            ImageListL: Similar to ImageList but now only the left part
            ImageListR: Similar to ImageList but now only the right part
                ImageListL and ImageListR have the same dimensions
        '''
        RotatedImageList = ndimage.rotate(ImageList,self.ImageAngle,reshape=False,axes=(2,3))

        ImageListL = RotatedImageList[:,:,self.BottomY+self.PixelOffset:self.TopY-self.PixelOffset,self.LeftX+self.PixelOffset:self.MiddleX-self.PixelOffset]
        ImageListR = RotatedImageList[:,:,self.BottomY+self.PixelOffset:self.TopY-self.PixelOffset,self.MiddleX+self.PixelOffset:self.RightX-self.PixelOffset]
        return ImageListL,ImageListR

    def GetRotations(self,ImageTimeList):
        '''
        Summary:     
            Finds the Imr and Hwp angle of each image.
            Each image has a time in the header. We also
            know the Imr and Hwp angle over time. This function
            combines the two.
        Input:
            ImageTimeList: Time at which the images are taken.
                Is a list of timedelta's.

        Output:
            ImageImrList: Imr angle of each image.
            ImageHwpList: Hwp angle of each image.
            BadImageList: Boolean list that indicates which
                images have no Imr/Hwp angle to be found.
                Should be used to remove those images outside
                this function
        '''

        ImageImrList = [] #ImrAngle for each calibration image
        ImageHwpList = [] #HwpAngle for each calibration image
        BadImageList = [] #Boolean list of images without correct Imr,Hwp angles
        for i in range(len(ImageTimeList)):
            ImageTime = ImageTimeList[i]

            DeltaList = (self.RotationTimeList-ImageTime) / timedelta(seconds=1) #List of time differences in seconds

            TargetIndex = ArgMaxNegative(DeltaList)
            if(np.abs(DeltaList[TargetIndex]) <= self.MaxTimeDifference):
                BadImageList.append(False)
                ImageImrList.append(self.RotationImrList[TargetIndex])
                ImageHwpList.append(self.RotationHwpList[TargetIndex])
            else:
                BadImageList.append(True)

        return np.array(ImageImrList),np.array(ImageHwpList),np.array(BadImageList)

    def CreateHwpDoubleDifferenceImges(self,TotalHwpList,TotalImrList,ImageListL,ImageListR):
        '''
        Summary:     
            Creates double difference and sum images by
            combining images differing 45 degrees HWP angle.
        Input:
            TotalHwpList: List of Hwp angles per image.
            TotalImrList: List of Imr angles per image.
            ImageListL: List of left part of images.
            ImageListR: List of right part of images.

        Output:
            DDImageArray: Array of double difference images.
            DSImageArray: Array of double sum images.
            ImrArray: Array of Imr angles per of images in DDImageArray,DSImageArray
        '''

        DDImageArray = []
        DSImageArray = []
        ImrArray = []
        OldThetaImr = 0

        for HwpTarget in self.HwpTargetList:
            HwpPlusTarget = HwpTarget[0]
            HwpMinTarget = HwpTarget[1]
            ImrList = []
            DDImageList = []
            DSImageList = []
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalImrList[i] == TotalImrList[j]):
                            ThetaImr = TotalImrList[i]
                            if(ThetaImr < 0):
                                ThetaImr += 180

                            PlusDifference = ImageListL[j]-ImageListR[j]
                            MinDifference = ImageListL[i]-ImageListR[i]
                            PlusSum = ImageListL[j]+ImageListR[j]
                            MinSum = ImageListL[i]+ImageListR[i]
                            DDImage = 0.5*(PlusDifference - MinDifference) 
                            DSImage = 0.5*(PlusSum + MinSum)
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            ImrList.append(ThetaImr)

                            #There are normally 3 images found per derAngle
                            #There are only 2 at this specific spot
                            #Thus fills it up with the previous value so the array has right dimensions
                            if(ThetaImr == 112.5 and OldThetaImr == 112.5 and HwpPlusTarget == 33.75):
                                DDImageList.append(OldDDImage)
                                DSImageList.append(OldDSImage)
                                ImrList.append(OldThetaImr)

                            OldThetaImr = ThetaImr
                            OldDDImage = DDImage
                            OldDSImage = DSImage
                            break
                            
            DDImageArray.append(np.array(DDImageList))
            DSImageArray.append(np.array(DSImageList))
            ImrArray.append(np.array(ImrList))

        return np.array(DDImageArray),np.array(DSImageArray),np.array(ImrArray)

    #Uses aperatures to get a single value for the double differance
    def GetDoubleDifferenceValue(self,DDImageArray,DSImageArray):
        '''
        Summary:     
            Finds normalized parameter values from the double difference
            and double sum images. Currently just takes the median, should
            change to taking apertures.
        Input:
            DDImageArray: Array containing double difference images.
            DSImageArray: Array containing double sum images.
        Output:
            ParamValueArray: Array of normalized stokes parameters over Imr angle.
        '''

        #ParamValueArray = []
        #for i in range(len(DDImageArray)):
        #    ParamValueArray.append(np.median(DDImageArray[i],axis=(2,3)) / np.median(DSImageArray[i],axis=(2,3)))
        
        #return np.array(ParamValueArray)
        return np.median(DDImageArray,axis=(3,4)) / np.median(DSImageArray,axis=(3,4))


        #for i in range(len(self.ApertureXList)):
        #    ApertureX = self.ApertureXList[i]
        #    ApertureY = self.ApertureYList[i] 
        #    Shape = DDImageArray[0][0].shape
        #    Aperture = CreateAperture(Shape,ApertureX,ApertureY,self.ApertureSize)
        #    ParamValue = np.median(DDImageArray[:,:,Aperture==1],axis=2) / np.median(DSImageArray[:,:,Aperture==1],axis=2)
        #    ParamValueArray.append(ParamValue)

        #return np.array(ParamValueArray)


#--/--Class--/--#

#-----Functions-----#

#---ReadInFunctions---#

#Gets images and header data from fits files
def ReadCalibrationFiles(FileList):
    LambdaList = []
    ImageList = []
    TimeList = []
    for File in FileList:
        Header = File[0].header
        Image = File[1].data
        RawHeader = File[3].header #Not currently used, not sure what to do with this
        Lambda = Header['lam_min']*np.exp(np.arange(Image.shape[0])*Header['dloglam']) #This gets the wavelength...
        LambdaList.append(Lambda)
        ImageList.append(Image)
        Days = float(Header["UTC-Date"][-2:])

        TimeRow = Header["UTC-Time"].split(":")
        #Converts time to the timedelta data type
        TimeList.append(timedelta(hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2]),days=Days))
    
    return np.array(ImageList),np.array(LambdaList),np.array(TimeList)

#Reads the file with rotations for each date,time
def ReadRotationFile(RotationFile):

    TimeList = []
    ImrAngleList = []
    HwpAngleList = []

    for Row in RotationFile:
        RowList = Row.split(" ") 

        TimeRow = RowList[1].split(":")
        TimeList.append(timedelta(days=float(RowList[0][-2:]),hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2])))

        ImrAngleList.append(float(RowList[2]))
        HwpAngleList.append(float(RowList[3][:-1]))#Also removes the /n on the end with the [:-1]

    return np.array(TimeList),np.array(ImrAngleList),np.array(HwpAngleList)

#-/-ReadInFunctions-/-#

#---OtherFunctions---#
#Finds the index of the highest(closest to zero) negative number in a list
def ArgMaxNegative(List):
    List = List*(List<=0) - List*(List>0)*1E6
    return np.argmax(List)

#-/-OtherFunctions-/-#

#-----Parameters-----#

#Path to calibration files
PolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_pol_source/CRSA000"
UnpolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA000"
RotationPath = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/RotationsChanged.txt"

PolNumberList = np.arange(59565,59905)
UnpolNumberList = np.arange(59559,59565)

#--/--Parameters--/--#

#-----Main-----#

if __name__ == '__main__':
    #Get the file with rotations over time
    RotationFile = open(RotationPath, "r")

    if(True):
        UnpolFile = fits.open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA00059559_cube.fits")
        Header = UnpolFile[0].header
        print(repr(Header))

    if(False):
        #Get the polarized calibration images
        PolFileList = []
        for PolNumber in PolNumberList:
            PolPath = PolPrefix + str(PolNumber) + "_cube.fits"
            PolFile = fits.open(PolPath)
            PolFileList.append(PolFile)

        SCExAO_CalibrationObject = SCExAO_Calibration(PolFileList,RotationFile)
        SCExAO_CalibrationObject.RunCalibration(PolFileList,RotationFile)
        pickle.dump(SCExAO_CalibrationObject,open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","wb"))

#--/--Main--/--#



