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

    Lambda = self.LambdaList[0][LambdaNumber]

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
        for j in range(len(self.ParamValueArray[0][0])):
            ParamValueList = 100*self.ParamValueArray[i,:,j,LambdaNumber]
            plt.scatter(self.ImrArray[i],ParamValueList,zorder=100,color=self.ColorList[i],s=18,edgecolors="black")

        if(PlotModelCurve):
            #Plots model curve    
            plt.plot(ModelDerList*180/np.pi,ModelParmValueArray[i]*100,color=self.ColorList[i],label=r"$\theta^+_{Hwp} = $"+str(HwpPlusTarget))


    plt.grid(linestyle="--")
    legend = plt.legend(fontsize=7,loc=1)
    legend.set_zorder(200)

    #-/-MainPlot-/-#

    #---ResidualsPlot---#
    if(PlotModelCurve):

        ResidualModelParmValueArray = 100*Model.FindParameterArray(self.ImrArray[0]*np.pi/180,S_In,0,self.HwpTargetList,True,False)
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
            for j in range(len(self.ParamValueArray[0][0])):
                ParamValueList = 100*self.ParamValueArray[i,:,j,LambdaNumber]
                ResidualValueList = ParamValueList - ResidualModelParmValueArray[i]
                plt.scatter(self.ImrArray[i],ResidualValueList,zorder=100,color=self.ColorList[i],s=13,edgecolors="black",linewidth=1)

        plt.grid(linestyle="--")

    #-/-ResidualsPlot-/-#

    
def PlotPolarimetricEfficiency(self,LambdaNumber,Color,Model):

    Lambda = self.LambdaList[0][LambdaNumber]

    plt.title("Polarimetric efficiency over derotator angle for several wavelengths")   
    plt.xlabel(r"Derotator angle($^\circ$)")
    plt.ylabel("Polarimetric efficiency(%)")

    plt.xlim(left=0,right=180)
    plt.ylim(bottom=0,top=100)
    plt.yticks(np.arange(0,110,10))
    plt.xticks(np.arange(0,195,15))

    ModelDerList = np.linspace(0*np.pi/180,180*np.pi/180,400)
    S_In = np.array([1,0,0,0])       

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    Model_PolDegree1 = np.sqrt(ModelParmValueArray[0,:]**2+ModelParmValueArray[2,:]**2)
    Model_PolDegree2 = np.sqrt(ModelParmValueArray[1,:]**2+ModelParmValueArray[3,:]**2)
    Model_PolDegree = 0.5*(Model_PolDegree1+Model_PolDegree2)

    plt.plot(ModelDerList*180/np.pi,Model_PolDegree*100,color=Color,label=r"$\lambda$="+str(int(Lambda))+"nm")

    for ApertureNumber in range(8):
        PolDegree1 = np.sqrt(self.ParamValueArray[0,:,ApertureNumber,LambdaNumber]**2+self.ParamValueArray[2,:,ApertureNumber,LambdaNumber]**2)
        PolDegree2 = np.sqrt(self.ParamValueArray[1,:,ApertureNumber,LambdaNumber]**2+self.ParamValueArray[3,:,ApertureNumber,LambdaNumber]**2)
        PolDegree = 0.5*(PolDegree1+PolDegree2)

        plt.scatter(self.ImrArray[0],100*PolDegree,zorder=100,s=18,color=Color,edgecolors="black")

    plt.grid(linestyle="--")


def PlotPolarimetricEfficiencyDiff(self,LambdaNumber,Model):

    Lambda = self.LambdaList[0][LambdaNumber]

    #plt.title("Polarimetric efficiency over derotator angle for several wavelengths")   
    #plt.title("Polarimetric efficiency over derotator angle for $\lambda=$"+str(int(Lambda))+"nm")  
    plt.xlabel(r"Derotator angle($^\circ$)")
    plt.ylabel("Polarimetric efficiency(%)")

    plt.xlim(left=0,right=180)
    plt.ylim(bottom=0,top=100)
    plt.yticks(np.arange(0,110,10))
    plt.xticks(np.arange(0,195,15))

    ModelDerList = np.linspace(0*np.pi/180,180*np.pi/180,400)
    S_In = np.array([1,0,0,0])       

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    Model_PolDegree1 = np.sqrt(ModelParmValueArray[0,:]**2+ModelParmValueArray[2,:]**2)
    Model_PolDegree2 = np.sqrt(ModelParmValueArray[1,:]**2+ModelParmValueArray[3,:]**2)
    Model_PolDegree = 0.5*(Model_PolDegree1+Model_PolDegree2)

    plt.plot(ModelDerList*180/np.pi,Model_PolDegree1*100,color="red",label="$DoLP$ of $x(0^\circ)$ and $x(22.5^\circ)$")
    plt.plot(ModelDerList*180/np.pi,Model_PolDegree2*100,color="blue",label="$DoLP$ of $x(11.25^\circ)$ and $x(33.75^\circ)$")
    
    #Title too big
    #plt.title("Polarimetric efficiency calculated using different Stokes parameter combinations")

    for ApertureNumber in range(8):
        PolDegree1 = np.sqrt(self.ParamValueArray[0,:,ApertureNumber,LambdaNumber]**2+self.ParamValueArray[2,:,ApertureNumber,LambdaNumber]**2)
        PolDegree2 = np.sqrt(self.ParamValueArray[1,:,ApertureNumber,LambdaNumber]**2+self.ParamValueArray[3,:,ApertureNumber,LambdaNumber]**2)
        PolDegree = 0.5*(PolDegree1+PolDegree2)

        #plt.scatter(self.ImrArray[0],100*PolDegree,zorder=100,s=18,color=Color,edgecolors="black")

        plt.scatter(self.ImrArray[0],100*PolDegree1,zorder=100,s=18,edgecolors="black",color="red")
        plt.scatter(self.ImrArray[0],100*PolDegree2,zorder=100,s=18,edgecolors="black",color="blue")
        
    plt.grid(linestyle="--")

def PlotMinimumEfficiency(self,ModelList):
    plt.figure()

    MinPolDegreeList = []
    ModelDerList = np.append(np.linspace(35*np.pi/180,55*np.pi/180,200),np.linspace(125*np.pi/180,145*np.pi/180,200))

    S_In = np.array([1,0,0,0])

    for i in range(len(ModelList)):
        ModelParmValueArray = ModelList[i].FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
        Model_PolDegree1 = np.sqrt(ModelParmValueArray[0,:]**2+ModelParmValueArray[2,:]**2)
        Model_PolDegree2 = np.sqrt(ModelParmValueArray[1,:]**2+ModelParmValueArray[3,:]**2)
        Model_PolDegree = 0.5*(Model_PolDegree1+Model_PolDegree2)

        MinPolDegree = np.amin(Model_PolDegree)
        MinPolDegreeList.append(MinPolDegree)

    plt.scatter(self.LambdaList[0],np.array(MinPolDegreeList)*100,color="black",zorder=10)

    plt.title("Minimum polarimetric efficiency over wavelength")
    plt.xlabel(r"$\lambda$(nm)")
    plt.ylabel("Minimum polarimetric efficiency(%)")

    plt.xlim(xmin=np.amin(self.LambdaList[0])-50,xmax=np.amax(self.LambdaList[0])+50)
    plt.ylim(ymin=0,ymax=100)
    plt.yticks(np.arange(0,110,10))

    plt.grid(linestyle="--")

def Plot_AOLP_Offset(self,LambdaNumber,Model,Color="black"):

    Lambda = self.LambdaList[0][LambdaNumber]

    plt.title("AOLP offset over derotator angle for several wavelengths")   
    plt.xlabel(r"Derotator angle($^\circ$)")
    plt.ylabel("AOLP offset($^\circ$)")

    plt.yticks(np.arange(-90,105,15))
    plt.xticks(np.arange(0,195,15))
    plt.xlim(left=0,right=180)
    plt.ylim(-90,90)


    ModelDerList = np.linspace(0*np.pi/180,180*np.pi/180,1000)
    S_In = np.array([1,0,0,0])       

    IdealModel = SCExAO_Model.MatrixModel("",0,180,0,0,180,0,1,0,0,0)

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    IdealParmValueArrayContinuous = IdealModel.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
    IdealParmValueArrayDiscrete = IdealModel.FindParameterArray(self.ImrArray[0]*np.pi/180,S_In,0,self.HwpTargetList,True,False)

    ModelAOLP_List = []
    ModelAOLP_List2 = []

    for i in range(len(ModelDerList)):
        IdealModelAOLP = CalculateAOLP(IdealParmValueArrayContinuous[0][i],IdealParmValueArrayContinuous[2][i])
        RealModelAOLP = CalculateAOLP(ModelParmValueArray[0][i],ModelParmValueArray[2][i])
        ModelAOLP_List.append(ModulateAngle(RealModelAOLP-IdealModelAOLP))

    #Stole this from SO, removes discontinuity
    pos = np.where(np.abs(np.diff(ModelAOLP_List)) >= 50)[0]+1
    ModelDerList = np.insert(ModelDerList, pos, np.nan)
    ModelAOLP_List = np.insert(ModelAOLP_List, pos, np.nan)

    plt.plot(ModelDerList*180/np.pi,ModelAOLP_List,color=Color,label=r"$\lambda$="+str(int(Lambda))+"nm")
    plt.axhline(y=0,color="black")

    for ApertureNumber in range(8):

        MeasuredAOLP_List = []

        for i in range(len(self.ImrArray[0])): 
            MeasuredAOLP = CalculateAOLP(self.ParamValueArray[0,i,ApertureNumber,LambdaNumber],self.ParamValueArray[2,i,ApertureNumber,LambdaNumber])
            IdealModelAOLP_Discrete = CalculateAOLP(IdealParmValueArrayDiscrete[0][i],IdealParmValueArrayDiscrete[2][i])
            MeasuredAOLP_List.append(ModulateAngle(MeasuredAOLP-IdealModelAOLP_Discrete))


        plt.scatter(self.ImrArray[0],MeasuredAOLP_List,zorder=100,s=18,edgecolors="black",color=Color)


    plt.grid(linestyle="--")


def Plot_Diff_AOLP_Offset(self,LambdaNumber,Model,Color="black"):

    Lambda = self.LambdaList[0][LambdaNumber]
        
    plt.xlabel(r"Derotator angle($^\circ$)")
    plt.ylabel("AOLP offset($^\circ$)")

    plt.yticks(np.arange(-90,105,7.5))
    plt.xticks(np.arange(0,195,15))
    plt.xlim(left=0,right=180)
    plt.ylim(-30,30)

    ModelDerList = np.linspace(0*np.pi/180,180*np.pi/180,1000)
    S_In = np.array([1,0,0,0])       

    IdealModel = SCExAO_Model.MatrixModel("",0,180,0,0,180,0,1,0,0,0)

    #Stokes parameters as predicted by the model
    ModelParmValueArray = Model.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)

    IdealParmValueArrayContinuous = IdealModel.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
    IdealParmValueArrayDiscrete = IdealModel.FindParameterArray(self.ImrArray[0]*np.pi/180,S_In,0,self.HwpTargetList,True,False)

    ModelAOLP_List1 = []
    ModelAOLP_List2 = []

    for i in range(len(ModelDerList)):
        IdealModelAOLP1 = CalculateAOLP(IdealParmValueArrayContinuous[0][i],IdealParmValueArrayContinuous[2][i])
        RealModelAOLP1 = CalculateAOLP(ModelParmValueArray[0][i],ModelParmValueArray[2][i])
        ModelAOLP_List1.append(ModulateAngle(RealModelAOLP1-IdealModelAOLP1))

        IdealModelAOLP2 = CalculateAOLP(IdealParmValueArrayContinuous[1][i],IdealParmValueArrayContinuous[3][i])
        RealModelAOLP2 = CalculateAOLP(ModelParmValueArray[1][i],ModelParmValueArray[3][i])
        ModelAOLP_List2.append(ModulateAngle(RealModelAOLP2-IdealModelAOLP2))


    plt.plot(ModelDerList[1:]*180/np.pi,ModelAOLP_List1[1:],color="red",label="$AoLP$ of $x(0^\circ)$ and $x(22.5^\circ)$")
    plt.plot(ModelDerList[1:]*180/np.pi,ModelAOLP_List2[1:],color="blue",label="$AoLP$ of $x(11.25^\circ)$ and $x(33.75^\circ)$")

    plt.axhline(y=0,color="black")

    for ApertureNumber in range(8):

        MeasuredAOLP_List1 = []
        MeasuredAOLP_List2 = []

        for i in range(len(self.ImrArray[0])): 
            MeasuredAOLP1 = CalculateAOLP(self.ParamValueArray[0,i,ApertureNumber,LambdaNumber],self.ParamValueArray[2,i,ApertureNumber,LambdaNumber])
            IdealModelAOLP_Discrete1 = CalculateAOLP(IdealParmValueArrayDiscrete[0][i],IdealParmValueArrayDiscrete[2][i])
            MeasuredAOLP_List1.append(ModulateAngle(MeasuredAOLP1-IdealModelAOLP_Discrete1))

            MeasuredAOLP2 = CalculateAOLP(self.ParamValueArray[1,i,ApertureNumber,LambdaNumber],self.ParamValueArray[3,i,ApertureNumber,LambdaNumber])
            IdealModelAOLP_Discrete2 = CalculateAOLP(IdealParmValueArrayDiscrete[1][i],IdealParmValueArrayDiscrete[3][i])
            MeasuredAOLP_List2.append(ModulateAngle(MeasuredAOLP2-IdealModelAOLP_Discrete2))

        plt.scatter(self.ImrArray[0],MeasuredAOLP_List1,zorder=100,s=18,edgecolors="black",color="red")
        plt.scatter(self.ImrArray[0],MeasuredAOLP_List2,zorder=100,s=18,edgecolors="black",color="blue")
            
    plt.grid(linestyle="--")

def Plot_Max_AOLP_Offset(self,ModelList):

    plt.figure()

    plt.title("Maximum AOLP offset over wavelength")           
    plt.xlabel(r"$\lambda$(nm)")
    plt.ylabel("Maximum AOLP offset($^\circ$)")

    plt.xlim(xmin=np.amin(self.LambdaList[0])-50,xmax=np.amax(self.LambdaList[0])+50)
    plt.ylim(bottom=0,top=95)
    plt.yticks(np.arange(0,100,10))
    #plt.xticks(np.arange())

    ModelDerList = np.linspace(15*np.pi/180,165*np.pi/180,1000)
    S_In = np.array([1,0,0,0])       

    IdealModel = SCExAO_Model.MatrixModel("",0,180,0,0,180,0,1,0,0,0)
    IdealParmValueArrayContinuous = IdealModel.FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
    MaxAOLP_List = []

    for i in range(len(ModelList)):

        #Stokes parameters as predicted by the model
        ModelParmValueArray = ModelList[i].FindParameterArray(ModelDerList,S_In,0,self.HwpTargetList,True,False)
        ModelAOLP_List = []

        for i in range(len(ModelDerList)):
            IdealModelAOLP = CalculateAOLP(IdealParmValueArrayContinuous[0][i],IdealParmValueArrayContinuous[2][i])
            RealModelAOLP = CalculateAOLP(ModelParmValueArray[0][i],ModelParmValueArray[2][i])
            ModelAOLP_List.append(ModulateAngle(RealModelAOLP-IdealModelAOLP))

        MaxAOLP_List.append(np.amax(np.absolute(ModelAOLP_List)))

    plt.scatter(self.LambdaList[0],MaxAOLP_List,color="black",s=20,zorder=10)

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
    
    plt.xlabel("x(pixels)")
    plt.ylabel("y(pixles)")
    cbar = plt.colorbar()
    cbar.set_label('Counts')

    plt.show()


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


def ModulateAngle(Angle):
    while(Angle <-90 or Angle > 90):
        if(Angle < -90):
            Angle += 180
        if(Angle > 90):
            Angle -= 180
    return Angle

def CalculateAOLP(Q,U):
    Delta = 0
    if(Q <= 0):
        Delta = 90

    return 0.5*np.arctan(U/Q)*180/np.pi+Delta# - 2*ThetaDerList[i] 

#--/--OtherFunctions--/--#
#-----SetFunctions-----#
SCExAO_CalibrationMain.SCExAO_Calibration.PlotParamValues = PlotParamValues
SCExAO_CalibrationMain.SCExAO_Calibration.PlotPolarimetricEfficiency = PlotPolarimetricEfficiency
SCExAO_CalibrationMain.SCExAO_Calibration.PlotApertures = PlotApertures
SCExAO_CalibrationMain.SCExAO_Calibration.PlotMinimumEfficiency = PlotMinimumEfficiency
SCExAO_CalibrationMain.SCExAO_Calibration.Plot_AOLP_Offset = Plot_AOLP_Offset
SCExAO_CalibrationMain.SCExAO_Calibration.Plot_Max_AOLP_Offset = Plot_Max_AOLP_Offset
SCExAO_CalibrationMain.SCExAO_Calibration.PlotPolarimetricEfficiencyDiff = PlotPolarimetricEfficiencyDiff
SCExAO_CalibrationMain.SCExAO_Calibration.Plot_Diff_AOLP_Offset = Plot_Diff_AOLP_Offset
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