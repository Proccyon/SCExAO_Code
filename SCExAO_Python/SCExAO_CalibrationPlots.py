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

    
def PlotPolarizationDegree(self,LambdaNumber):
    plt.figure()
    plt.ylabel("Degree of linear polarization(%)")
    plt.xticks(np.arange(45,135,7.5))

    Lambda = self.PolLambdaList[0][LambdaNumber]  
    plt.title("Pol degree vs Imr angle (polarizer)(Lambda="+str(int(Lambda))+"nm)")           
    plt.xlabel("Imr angle(degrees)")
    
    plt.yticks(np.arange(0,100,10))
    plt.xlim(left=44,right=128.5)
    plt.ylim(bottom=0,top=100)

    PolDegree = np.sqrt(self.PolParamValueArray[0][:,LambdaNumber]**2+self.PolParamValueArray[2][:,LambdaNumber]**2)
    #PolDegree = np.sqrt(self.PolParamValueArray[1][:,LambdaNumber]**2+self.PolParamValueArray[3][:,LambdaNumber]**2)

    plt.scatter(self.PolImrArray[0],100*PolDegree,zorder=100,s=18,edgecolors="black")
        
    plt.grid(linestyle="--")


#--/--PlotFunctions-----#

#-----SetFunctions-----#
SCExAO_CalibrationMain.SCExAO_Calibration.PlotParamValues = PlotParamValues
SCExAO_CalibrationMain.SCExAO_Calibration.PlotPolarizationDegree = PlotPolarizationDegree
#--/--SetFunctions--/--#

#-----Main-----#

if __name__ == '__main__':
    SCExAO = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    Model = SCExAO_Model.IdealModel

    for i in range(1):
        print(i)
        SCExAO.PlotParamValues(i,Model,True)

    plt.show()



#--/--Main--/--#