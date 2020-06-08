#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import pickle
import Methods as Mt
import SCExAO_Model
import SCExAO_CalibrationMain
from SCExAO_CalibrationMain import SCExAO_Calibration

#--/--Imports--/--#

#-----PlotFunctions-----#

def PlotParamValues(self,LambdaNumber,Model=SCExAO_Model.BB_H,PlotModelCurve=False):
    
    fig1 = plt.figure()

    #---MainPlot---#

    frame1=fig1.add_axes((.13,.36,.77,.57))

    Lambda = self.PolLambdaList[0][LambdaNumber]

    plt.title("Stokes parameters against derotator angle (polarized source)($\lambda$="+str(int(Lambda))+"nm)") 
    plt.ylabel("Normalized Stokes parameter (%)")

    frame1.set_xticklabels([]) #Remove x-tic labels for the first frame          
    plt.yticks(np.arange(-120,120,20))
    plt.xticks(np.arange(45,135,7.5))

    plt.xlim(left=44,right=128.5)
    plt.ylim(bottom=-100,top=100)

    plt.axhline(y=0,color="black")

    ModelDerList = np.linspace(43*np.pi/180,130*np.pi/180,200)
    S_In = np.array([1,0,0,0])       

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    for i in range(len(self.HwpTargetList)):
        HwpPlusTarget = self.HwpTargetList[i][0]
        HwpMinTarget = self.HwpTargetList[i][1]

        #Loop through apertures
        for j in range(len(self.PolParamValueArray[0][0])):
            ParamValueList = 100*self.PolParamValueArray[i,:,j,LambdaNumber]
            plt.scatter(self.PolImrArray[i],ParamValueList,zorder=100,color=self.ColorList[i],s=18,edgecolors="black")

        if(PlotModelCurve):
            #Plots model curve    
            plt.plot(ModelDerList*180/np.pi,ModelParmValueArray[i]*100,color=self.ColorList[i],label=r"$\theta^+_{Hwp} = $"+str(HwpPlusTarget))


    plt.grid(linestyle="--")
    legend = plt.legend(fontsize=7,loc=1)
    legend.set_zorder(200)

    #-/-MainPlot-/-#

    #---ResidualsPlot---#
    if(PlotModelCurve):

        ResidualModelParmValueArray = 100*Model.FindParameterArray(self.PolImrArray[0]*np.pi/180,S_In,0,self.HwpTargetList,True,False)
        frame2=fig1.add_axes((.13,.1,.77,.23))

        plt.xticks(np.arange(45,135,7.5))
        #plt.yticks(np.arange(-5,7,2))
        plt.locator_params(axis='y', nbins=6)

        plt.xlim(left=44,right=128.5)
        #plt.ylim(ymin=-5.5,ymax=5.5)

        plt.xlabel("Der angle(degrees)")
        plt.ylabel("Residuals (%)")
        
        plt.axhline(y=0,color="black")
        
        for i in range(len(self.HwpTargetList)):
            #Loop through apertures
            for j in range(len(self.PolParamValueArray[0][0])):
                ParamValueList = 100*self.PolParamValueArray[i,:,j,LambdaNumber]
                ResidualValueList = ParamValueList - ResidualModelParmValueArray[i]
                plt.scatter(self.PolImrArray[i],ResidualValueList,zorder=100,color=self.ColorList[i],s=13,edgecolors="black",linewidth=1)

        plt.grid(linestyle="--")

    #-/-ResidualsPlot-/-#

    
def PlotPolarimetricEfficiency(self,LambdaNumber,Color,Model):

    Lambda = self.PolLambdaList[0][LambdaNumber]

    plt.title("Polarimetric efficiency over derotator angle at several wavelengths")           
    plt.xlabel(r"Derotator angle($^\circ$)")
    plt.ylabel("Degree of linear polarization(%)")

    plt.xlim(left=0,right=128.5)
    plt.ylim(bottom=0,top=100)
    plt.yticks(np.arange(0,100,10))
    plt.xticks(np.arange(0,135,15))

    ModelDerList = np.linspace(0*np.pi/180,130*np.pi/180,400)
    S_In = np.array([1,0,0,0])       

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    Model_PolDegree1 = np.sqrt(ModelParmValueArray[0,:]**2+ModelParmValueArray[2,:]**2)
    Model_PolDegree2 = np.sqrt(ModelParmValueArray[1,:]**2+ModelParmValueArray[3,:]**2)
    Model_PolDegree = 0.5*(Model_PolDegree1+Model_PolDegree2)

    plt.plot(ModelDerList*180/np.pi,Model_PolDegree*100,color=Color,label=r"$\lambda$="+str(int(Lambda))+"nm")

    for ApertureNumber in range(8):
        PolDegree1 = np.sqrt(self.PolParamValueArray[0,:,ApertureNumber,LambdaNumber]**2+self.PolParamValueArray[2,:,ApertureNumber,LambdaNumber]**2)
        PolDegree2 = np.sqrt(self.PolParamValueArray[1,:,ApertureNumber,LambdaNumber]**2+self.PolParamValueArray[3,:,ApertureNumber,LambdaNumber]**2)
        PolDegree = 0.5*(PolDegree1+PolDegree2)

        plt.scatter(self.PolImrArray[0],100*PolDegree,zorder=100,s=18,edgecolors="black",color=Color)
        
    plt.grid(linestyle="--")

def PlotMinimumEfficiency(self,ModelList):
    plt.figure()

    MinPolDegreeList = []
    ModelDerList = np.linspace(-40*np.pi/180,50*np.pi/180,100)
    S_In = np.array([1,0,0,0])

    for i in range(len(ModelList)):
        ModelParmValueArray = ModelList[i].FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
        Model_PolDegree1 = np.sqrt(ModelParmValueArray[0,:]**2+ModelParmValueArray[2,:]**2)
        Model_PolDegree2 = np.sqrt(ModelParmValueArray[1,:]**2+ModelParmValueArray[3,:]**2)
        Model_PolDegree = 0.5*(Model_PolDegree1+Model_PolDegree2)

        MinPolDegree = np.amin(Model_PolDegree)
        MinPolDegreeList.append(MinPolDegree)

    plt.scatter(self.PolLambdaList[0],np.array(MinPolDegreeList)*100,color="black",zorder=10)

    plt.xlabel(r"$\lambda$(nm)")
    plt.ylabel("Polarimetric efficiency(%)")

    plt.xlim(xmin=np.amin(self.PolLambdaList[0])-50,xmax=np.amax(self.PolLambdaList[0])+50)
    plt.ylim(ymin=0,ymax=100)
    plt.yticks(np.arange(0,110,10))

    plt.grid(linestyle="--")



def PlotApertures(self,ImageNumber,LambdaNumber):
    fig = plt.figure()
    
    i=0
    for ApertureCoord in self.ApertureCoordList:
        i+=1
        Aperture_X_List,Aperture_Y_List = CreateApertureContour(ApertureCoord[0],ApertureCoord[1],self.ApertureLx,self.ApertureLy,self.ApertureAngle)
        
        plt.plot(Aperture_X_List,Aperture_Y_List,color="red")
        plt.plot(Aperture_X_List-self.RollVector[0],Aperture_Y_List-self.RollVector[1],color="red")

        plt.text(ApertureCoord[0]-3,ApertureCoord[1]+3,i,fontsize=12,color="black")
        plt.text(ApertureCoord[0]-self.RollVector[0]-3,ApertureCoord[1]-self.RollVector[1]+3,i,fontsize=12,color="black")

    plt.imshow(self.PolImageList[LambdaNumber,ImageNumber],vmin=400,vmax=1800,cmap="gist_gray")
    print(self.PolImrList[0])
    print(self.PolHwpList[0])
    plt.xlabel("x(pixels)")
    plt.ylabel("y(pixles)")
    cbar = plt.colorbar()
    cbar.set_label('Counts')

    plt.show()

    #plt.imshow(LeftApertures)
    #plt.figure()
    #plt.imshow(self.ApertureImage)

    #plt.show()


def CreateApertureContour(x0,y0,Lx,Ly,Angle):
    
    XList = []
    YList = []

    D_List = [[-1,-1],[-1,1],[1,1],[1,-1],[-1,-1]]

    for D in D_List:
        dx = D[0]
        dy = D[1]

        x = 0.5*dx*Lx
        y = 0.5*dy*Ly

        X = x*np.cos(Angle)-np.sin(Angle)*y+x0
        Y = x*np.sin(Angle)+np.cos(Angle)*y+y0

        XList.append(X)
        YList.append(Y)

    return np.array(XList),np.array(YList)


#--/--PlotFunctions-----#

#-----OtherFunctions----#

def CreateContours(Image):
    NewImage = np.zeros(Image.shape)

    for x0 in range(len(Image)):
        for y0 in range(len(Image[0])):
            if(Image[x0,y0]==False):
                continue

            NewPixelValue = False
            for dx in range(-1,2):
                for dy in range(-1,2):
                    if(not Image[x0+dx,y0+dy]):
                        NewPixelValue = True
            
            NewImage[x0,y0] = NewPixelValue

    return NewImage


#--/--OtherFunctions--/--#
#-----SetFunctions-----#
SCExAO_CalibrationMain.SCExAO_Calibration.PlotParamValues = PlotParamValues
SCExAO_CalibrationMain.SCExAO_Calibration.PlotPolarimetricEfficiency = PlotPolarimetricEfficiency
SCExAO_CalibrationMain.SCExAO_Calibration.PlotApertures = PlotApertures
SCExAO_CalibrationMain.SCExAO_Calibration.PlotMinimumEfficiency = PlotMinimumEfficiency
#--/--SetFunctions--/--#

#-----Main-----#

if __name__ == '__main__':
    SCExAO = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    Model = SCExAO_Model.IdealModel

    #SCExAO.PlotPolarizationDegree([0],["red","blue","green"])

    #for i in range(1):
    #    print(i)
    #    SCExAO.PlotParamValues(i,Model,True)

    #plt.show()



#--/--Main--/--#