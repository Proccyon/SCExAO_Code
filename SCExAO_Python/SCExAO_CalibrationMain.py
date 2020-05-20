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
        BottomMiddle: Pixel position of the bottom of the split of calibration image
        TopMiddle: Pixel position of the top of the split of calibration image
        CornerLeft: Pixel position of the left corner of the calibration image
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
        PolImageList: List of all polarized calibration images. Dim = (Wavelength,ImageId,X,Y)
         PolLambdaList: List of used wavelengths per image. Dim = (ImageId,Wavelength)
            Since it is the same for each image can be changed to Dim = (Wavelength)
         PolTimeList: List of times at which polarized images are taken. Times are stored as timedelta's
            Dim = (N)
        RotationTimeList: List of times at which Hwp/Imr angles are changed. Times are stored as timedelta's
            Dim = (RotationId)
        RotationImrList: List of Imr angles at times in RotationTimeList. Dim = (RotationId)
        RotationHwpList: List of Imr angles at times in RotationTimeList. Dim = (RotationId)
        PolImrList: List of Imr angles for each image in PolImageList. Dim = (ImageId)
        PolHwpList: List of Hwp angles for each image in PolImageList. Dim = (ImageId)
        PolBadImageList: Boolean list indicating which images have no associated Hwp/imr angle.
             Dim = (ImageId)
        PolApertureListL: List of values obtained by taking the median over apertures.
            Only contains values for left side of image. Dim = (Aperture,ImageId)
        PolApertureListR: Same as PolApertureListL but for the right side.
        ApertureImage: boolean image of apertures used. True = within aperture, False = outside aperture.
            Only used to illustrate apertures.
        PolDDArray: Array of double difference values. Dim = (HwpCombination,ImrAngle,Aperture,Wavelength)
        PolDSArray: Array of double sum values. Dim = (HwpCombination,ImrAngle,Aperture,Wavelength)
        PolParamValueArray: Array of normalized stokes parameters = PolDDArray / PolDSArray
            
    '''

    BottomMiddle = np.array([67,157])
    TopMiddle = np.array([130,41])
    CornerLeft = np.array([12,128])

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
        self.PolApertureListL,self.PolApertureListR,self.ApertureImage = self.SplitCalibrationImages(self.PolImageList)

        print("Creating double difference images...")
        self.PolDDArray,self.PolDSArray,self.PolImrArray = self.CreateHwpDoubleDifferenceImages(self.PolHwpList,self.PolImrList,self.PolApertureListL,self.PolApertureListR)

        self.PolParamValueArray = self.PolDDArray/self.PolDSArray
        

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

        RollVector = self.CornerLeft-self.BottomMiddle
        RolledImageList = np.roll(ImageList,RollVector[1],2)
        RolledImageList = np.roll(RolledImageList,RollVector[0],3)

        LeftApertureList = []
        RightApertureList = []
        FirstAperture = True

        for ApertureCoord in self.ApertureCoordList:
            Aperture = CreateAperture(ImageList[0][0].shape,ApertureCoord[0],ApertureCoord[1],self.ApertureLx,self.ApertureLy,self.ApertureAngle)
            if(FirstAperture):
                ApertureImage = Aperture
            else:
                ApertureImage = np.inverse(np.inverse(ApertureImage)*np.inverse(Aperture))

            LeftApertureValues = np.median(ImageList[:,:,Aperture],axis=(2))
            RightApertureValues = np.median(RolledImageList[:,:,Aperture],axis=(2))
            LeftApertureList.append(LeftApertureValues)
            RightApertureList.append(RightApertureValues)

        return np.array(LeftApertureList),np.array(RightApertureList), ApertureImage

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
            for i in range(len(TotalHwpList)):
                if(TotalHwpList[i] == HwpMinTarget):
                    for j in range(len(TotalHwpList)):
                        if(TotalHwpList[j] == HwpPlusTarget and TotalImrList[i] == TotalImrList[j]):
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

def CreateAperture(Shape,x0,y0,Lx,Ly,Angle=0):
    '''
    Summary:     
        Creates a rectangular aperture array of ones within the rectangle and 0's outside the circle.
    Input:
        Shape: Shape of the array
        x0: x coordinate of aperture centre
        y0: y coordinate of aperture centre
        Lx: Horizontal length in pixels
        Ly: Verical length in pixels
        Angle: Rotation of the aperture in radians 
    Output:
        Aperture: 2d aperture array
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

    if(False):
        UnpolFile = fits.open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA00059563_cube.fits")
        Header = UnpolFile[2].header
        #print(repr(Header))
        plt.imshow(UnpolFile[1].data[4],vmin=100,vmax=160)
        plt.show()
        #print(np.sum(.shape))

    if(True):
        #Get the polarized calibration images
        PolFileList = []
        for PolNumber in PolNumberList:
            PolPath = PolPrefix + str(PolNumber) + "_cube.fits"
            PolFile = fits.open(PolPath)
            PolFileList.append(PolFile)

        SCExAO_CalibrationObject = SCExAO_Calibration()
        SCExAO_CalibrationObject.RunCalibration(PolFileList,RotationFile)
        
        pickle.dump(SCExAO_CalibrationObject,open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","wb"))

#--/--Main--/--#



