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
        
    '''
    Summary:     
        Class contains all information about calibration data.
    static variables:
        RollVector: Vector in pixels indicating the difference in position of the left and right side of detector.
        ApertureCoordList: List of pixel positions of aperture centers
        ApertureLx: Horizontal length in pixels of apertures
        ApertureLy: Vertical length in pixels of apertures
        ApertureAngle: Angle of rotation in radians of the rectangular apertures
        MaxTimeDifference: If calibration image is taken with TimeDifference > MaxTimeDifference
            then image is discarded. TimeDifference is time between image taken and last Hwp/imr change.
        HwpTargetList: List of half-wave plate angle combinations with which to do the double difference
        ColorList: List of colors with which to plot stokes parameters
        PolParamValueArray_FileName: Name of text file where PolParamValueArray is stored with pickle
        PolImrArray_FileName: Name of text file where PolImrAray is stored with pickle
    
    local variables:
        PolImageList: List of all polarized calibration images. Has Dimensions (325,22,201,201) --> (ImageNumber,wavelength,X,Y)
        PolLambdaList: List of used wavelengths per image. Has Dimensions (325,22) --> (ImageNumber,Wavelength)
            Since it is the same for each image can be changed to Dimension (22) --> (Wavelength)
        PolTimeList: List of times at which polarized images are taken. Times are stored as timedelta's
            Has Dimensions (340) --> (ImageNumber)
        RotationTimeList: List of times at which Hwp/Imr angles are changed. Times are stored as timedelta's
            Has Dimensions (113) --> (RotationNumber)
        RotationImrList: List of Imr angles at times in RotationTimeList. Has Dimensions (113) --> (RotationNumber)
        RotationHwpList: List of Hwp angles at times in RotationTimeList. Has Dimensions (113) --> (RotationNumber)
        PolImrList: List of Imr angles for each image in PolImageList. Has Dimensions (325) --> (ImageNumber)
        PolHwpList: List of Hwp angles for each image in PolImageList. Has Dimensions (325) --> (ImageNumber)
        PolBadImageList: Boolean list indicating which images have no associated Hwp/imr angle.
             Has Dimensions (340) --> (ImageNumber)
        PolApertureListL: List of values obtained by taking the median over apertures.
            Only contains values for left side of image. Has Dimensions = (ApertureNumber,ImageNumber,wavelength)
        PolApertureListR: Same as PolApertureListL but for the right side. Has Dimensions = (ApertureNumber,ImageNumber,wavelength)
        PolDDArray: Array of double difference values. Has Dimensions (4,39,8,22) --> (HwpCombination,ImrAngle,ApertureNumber,Wavelength)
        PolDSArray: Array of double sum values. Has Dimensions (4,39,8,22) --> (HwpCombination,ImrAngle,ApertureNumber,Wavelength)
        PolParamValueArray: Array of normalized stokes parameters = PolDDArray / PolDSArray
            Has Dimensions (4,39,8,22) --> (HwpCombination,ImrAngle,ApertureNumber,Wavelength)
            
    '''

    RollVector = np.array([-60,-31])

    ApertureCoordList = np.array([(34,119),(58,131),(47,95),(72,107),(63,69),(86,81),(78,39),(103,52)])
    ApertureLx = 20
    ApertureLy = 20
    ApertureAngle=27*np.pi/180

    MaxTimeDifference = 5*60

    HwpTargetList = [(0,45),(11.25,56.25),(22.5,67.5),(33.75,78.75)]

    ColorList = ["blue","lightblue","red","orange"]

    PolParamValueArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Python/PickleFiles/PicklePolParamValueArray.txt"
    PolImrArray_FileName = "C:/Users/Gebruiker/Desktop/BRP/SCExAO_Python/PickleFiles/PicklePolImrArray.txt"
    
    
    def __init__(self):
        pass

        
    def RunCalibration(self,PolFileList,RotationFile):
        '''
        Summary:     
            Runs the whole calibration process. Combines all the methods found below.
        Input:
            PolFileList: List of .fit files containing polarized calibration images
            RotationFile: Text file containing Hwp/Imr angles over time
        '''

        print("Reading files...")
        self.PolImageList,self.PolLambdaList,self.PolTimeList = ReadCalibrationFiles(PolFileList)
        self.RotationTimeList,self.RotationImrList,self.RotationHwpList = ReadRotationFile(RotationFile)

        print("Finding Imr and Hwp angles of calibration images...")
        self.PolImrList,self.PolHwpList,self.PolBadImageList = self.GetRotations(self.PolTimeList)
        self.PolImageList = self.PolImageList[self.PolBadImageList==False]
        self.PolLambdaList = self.PolLambdaList[self.PolBadImageList==False]

        print("Splitting calibration images...")
        self.PolApertureListL,self.PolApertureListR = self.SplitCalibrationImages(self.PolImageList)

        print("Creating double difference images...")
        self.PolDDArray,self.PolDSArray,self.PolImrArray = self.CreateHwpDoubleDifferenceImages(self.PolHwpList,self.PolImrList,self.PolApertureListL,self.PolApertureListR)

        self.PolParamValueArray = self.PolDDArray/self.PolDSArray
        

    def SplitCalibrationImages(self,ImageList):
        '''
        Summary:     
            Rotates the images and splits them in a left and right part.
        Input:
            ImageList:  List of all calibration images.
                Has dimensions (325,22,201,201) --> (ImageNumber,wavelength,x,y)
        Output:
            LeftApertureList: Median values of the apertures on left side of detector
                Has dimensions (8,325,22) --> (ApertureNumber,ImageNumber,wavelength)
            RightApertureList: Median values of the apertures on right side of detector
                Has dimensions (8,325,22) --> (ApertureNumber,ImageNumber,wavelength)
        '''

        #RolledImageList contains calibration images moved by RollVector
        RolledImageList = np.roll(ImageList,self.RollVector[1],2)
        RolledImageList = np.roll(RolledImageList,self.RollVector[0],3)

        LeftApertureList = []
        RightApertureList = []

        for ApertureCoord in self.ApertureCoordList:
            
            #Create rectangular aperture
            Aperture = CreateAperture(ImageList[0][0].shape,ApertureCoord[0],ApertureCoord[1],self.ApertureLx,self.ApertureLy,-self.ApertureAngle)

            LeftApertureValues = np.median(ImageList[:,:,Aperture],axis=(2))
            RightApertureValues = np.median(RolledImageList[:,:,Aperture],axis=(2))
            LeftApertureList.append(LeftApertureValues)
            RightApertureList.append(RightApertureValues)

        return np.array(LeftApertureList),np.array(RightApertureList)

    def GetRotations(self,ImageTimeList):
        '''
        Summary:     
            Finds the Imr and Hwp angle of each image.
            Each image has a time in the header. We also
            know the Imr and Hwp angle over time. This function
            combines the two.
        Input:
            ImageTimeList: Time at which the images are taken.
                Is a list of timedelta's. Has dimensions (340) --> (ImageNumber)
                
        Output:
            ImageImrList: Imr angle of each image.
                Has Dimensions: (325) --> (ImageNumber)
            ImageHwpList: Hwp angle of each image.
                Has Dimensions: (325) --> (ImageNumber)
            BadImageList: Boolean list that indicates which
                images have no Imr/Hwp angle to be found.
                Should be used to remove those images outside
                this function
                Has Dimensions: (340) --> (ImageNumber)
        '''

        ImageImrList = [] 
        ImageHwpList = [] 
        BadImageList = [] 
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

    def CreateHwpDoubleDifferenceImages(self,TotalHwpList,TotalImrList,ApertureListL,ApertureListR):
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
            UsedIndexList = [] #Skip images that are already used
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalImrList[i] == TotalImrList[j] and not j in UsedIndexList):
                            UsedIndexList.append(j)
                            ThetaImr = TotalImrList[i]
                            if(ThetaImr < 0):
                                ThetaImr += 180

                            PlusDifference = ApertureListL[:,j,:]-ApertureListR[:,j,:]
                            MinDifference = ApertureListL[:,i,:]-ApertureListR[:,i,:]
                            PlusSum = ApertureListL[:,j,:]+ApertureListR[:,j,:]
                            MinSum = ApertureListL[:,i,:]+ApertureListR[:,i,:]
                            DDImage = 0.5*(PlusDifference - MinDifference) 
                            DSImage = 0.5*(PlusSum + MinSum)
                            DDImageList.append(DDImage)
                            DSImageList.append(DSImage)
                            ImrList.append(ThetaImr)

                            #There are normally 3 images found per derAngle
                            #There are only 2 at this specific spot
                            #Thus fills it up with the previous value so the array has consistent dimensions
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


#--/--Class--/--#

#-----Functions-----#

#---ReadInFunctions---#

def ReadCalibrationFiles(FileList):
    '''
    Summary:     
        Gets the calibration images and header information from the .fits files.
    Input:
        FileList: List of calibration .fits files. Has Dimensions (340) --> (ImageNumber)

    Output:
        ImageList: List of calibration images. Has Dimensions (325,22,201,201) --> (ImageNumber,wavelength,X,Y)
        LambdaList: List of used wavelengths per image. Has Dimensions (325,22) --> (ImageNumber,Wavelength)
        TimeList: List of times at which polarized images are taken. Times are stored as timedelta's
            Has Dimensions (340) --> (ImageNumber)
    '''
    LambdaList = []
    ImageList = []
    TimeList = []
    for File in FileList:
        Header = File[0].header
        RawHeader = File[3].header #Not currently used, not sure what to do with this
        
        Image = File[1].data
        ImageList.append(Image)

        Lambda = Header['lam_min']*np.exp(np.arange(Image.shape[0])*Header['dloglam']) #This gets the wavelength...somehow
        LambdaList.append(Lambda)

        Days = float(Header["UTC-Date"][-2:])
        TimeRow = Header["UTC-Time"].split(":")

        #Converts time to the timedelta data type
        TimeList.append(timedelta(hours=float(TimeRow[0]),minutes=float(TimeRow[1]),seconds=float(TimeRow[2]),days=Days))
    
    return np.array(ImageList),np.array(LambdaList),np.array(TimeList)

def ReadRotationFile(RotationFile):
    '''
    Summary:     
        Reads the HWP and Imr angles over time from a text file.
    Input:
        RotationFile: Text file containing HWP and Imr angle over time.

    Output:
        TimeList: List of times at which Hwp/Imr angles are changed. Times are stored as timedelta's
            Has Dimensions (113) --> (RotationNumber) 
        ImrAngleList: List of Imr angles at times in TimeList. Has Dimensions (113) --> (RotationNumber)
        HwpAngleList: List of HWP angles at times in TimeList. Has Dimensions (113) --> (RotationNumber)
    '''

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
def ArgMaxNegative(List):
    '''
    Summary:     
        Finds the index of the highest(closest to zero) negative number in a list
    Input:
        List: Any list of numbers

    Output:
        MaxNegativeIndex: Index of the negative number in List that is closest to 0.
    '''
    List = List*(List<=0) - List*(List>0)*1E6
    MaxNegativeIndex = np.argmax(List)
    return MaxNegativeIndex

def CreateAperture(Shape,x0,y0,Lx,Ly,Angle=0):
    '''
    Summary:     
        Creates a rectangular aperture array of 1's within the rectangle and 0's outside the rectangle.

    Input:
        Shape: Shape of the array
        x0: x coordinate of aperture centre
        y0: y coordinate of aperture centre
        Lx: Horizontal length in pixels
        Ly: Verical length in pixels
        Angle: Rotation of the aperture in radians 

    Output:
        Aperture: 2d aperture array.
    '''
    Aperture = np.ones(Shape,dtype=bool)
    for y in range(Shape[0]):
        for x in range(Shape[1]):
            X = (x-x0)*np.cos(Angle)-np.sin(Angle)*(y-y0)
            Y = (x-x0)*np.sin(Angle)+np.cos(Angle)*(y-y0)
            if(X >= 0.5*Lx or X<= -0.5*Lx or Y >= 0.5*Ly or Y<= -0.5*Ly ):
                Aperture[y,x] = False
    
    return Aperture

#-/-OtherFunctions-/-#

#-----UserInput-----#

#First part of the path to polarized source .fits files
PolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_pol_source/CRSA000"
UnpolPrefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA000"

#Absolute path to rotation text file
RotationPath = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/RotationsChanged.txt"

#Absolute path to a unpolarized source .fits file, not used
UnpolPath = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA00059563_cube.fits"

#Path to text file where calibration results are stored
PickleSavePath = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt"

#--/--UserInput--/--#

#-----Main-----#

if __name__ == '__main__':

    PolNumberList = np.arange(59565,59905)
    UnpolNumberList = np.arange(59559,59565)

    #Get the file with rotations over time
    RotationFile = open(RotationPath, "r")

    #Checks the unpolarized source images, which are not used
    if(False):

        UnpolFile = fits.open(UnpolPath)
        Header = UnpolFile[2].header
        print("UnpolarizedSourceHeader:\n"+repr(Header))

        plt.imshow(UnpolFile[1].data[4],vmin=100,vmax=160)
        plt.show()

    if(True):
        #Get the polarized calibration images
        PolFileList = []
        for PolNumber in PolNumberList:
            PolPath = PolPrefix + str(PolNumber) + "_cube.fits"
            PolFile = fits.open(PolPath)
            PolFileList.append(PolFile)

        SCExAO_CalibrationObject = SCExAO_Calibration()
        SCExAO_CalibrationObject.RunCalibration(PolFileList,RotationFile)
        
        #Saves the calibration results in a text file using pickle module.
        pickle.dump(SCExAO_CalibrationObject,open(PickleSavePath,"wb"))

#--/--Main--/--#



