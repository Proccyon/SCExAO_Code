'''
#-----Header-----#

The goal of this file is to fit a model curve
to the data found in SCExAO_CalibrationMain.py.
This way the model parameters are obtained.
Currently only polarized calibration images are fitted.

#-----Header-----#
'''

#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize
import SCExAO_Model
import SCExAO_CalibrationPlots
from SCExAO_CalibrationMain import SCExAO_Calibration

#--/--Imports--/--#

#-----Functions-----#
def FindTotalParameterList(FittedParameterList,GuessParameterList,DoFitList):
    TotalParameterList = []
    j = 0
    for i in range(len(GuessParameterList)):
        if(DoFitList[i]):
            TotalParameterList.append(FittedParameterList[j])
            j+=1
        else:
            TotalParameterList.append(GuessParameterList[i])

    return np.array(TotalParameterList)


def MinimizeFunction(FittedParameterList,GuessParameterList,DerList,ParamValueArray,UsePolarizer,DerMethod,Bounds,DoFitList):
    
    '''
    Summary:     
        This is the function used as an argument in scipy.optimize.minimize.
        It returns the chi-squared value of given parameters compared to the data. 
    Input:
        FittedParameterList: Fitted parameter values found by scipy.optimize minimize.
                                This only includes parameters that are actually fitted.
        GuessParameterList: List of guess parameter values for all parameters.
                                If FittedParameterList does not include a parameter, use value from this list.
        DerList: Derotator angles (degree). These are hwp angles if DerMethod == True
        ParamvalueArray: List of stokes Q and U,etc. values over der and hwp angle as measured
        UsePolarizer: If calibration polarizer is used.
        DerMethod: If true double difference is done using differing der angles.
        Bounds: List of upper and lower bounds of all parameters.
        DoFitList: Boolean list, same length as StandardParameterList. Indicates for every parameter if it should be fitted. 
            
    Output:
        ChiSquared: Difference between FittedParamValueArray and ParamValueArray squared
        '''

    #Creates a list of all model parameters, even ones that are not currently fitted
    TotalParameterList = FindTotalParameterList(FittedParameterList,GuessParameterList,DoFitList)

    #If parameters are outside of bounds, return high chi squared value
    if(np.sum((TotalParameterList < Bounds[:,0])) + np.sum((TotalParameterList > Bounds[:,1])) > 0):
        return 1E12

    q_in = TotalParameterList[-2]
    u_in = TotalParameterList[-1]
    FittedModel = SCExAO_Model.MatrixModel("",*TotalParameterList[:-2],0,0)

    if(UsePolarizer):
        S_In = [1,0,0,0]
    else:
        S_In = [1,q_in,u_in,0]

    FittedParamValueArray = FittedModel.FindParameterArray(DerList*np.pi/180,S_In,UsePolarizer=UsePolarizer,DerMethod=DerMethod)
    
    ChiSquared = np.sqrt(np.sum((FittedParamValueArray-ParamValueArray)**2))
    return ChiSquared

def DoFit(Args):
    '''
    Summary:     
        Finds irdis model parameters by fitting the matrix model using the calibration data.
        Uses scipy.optimize.minimize.
    Input:
        Args: List of all constant variables required in MinimizeFunction.
            
    Output:
        TotalParameterList: List of all fitted irdis model parameters
        ChiSquared: ChiSquared value of best fit
    '''

    GuessParameterList = Args[0]
    DoFitList = Args[6]
    Results = scipy.optimize.minimize(MinimizeFunction,GuessParameterList[DoFitList],args=Args,method="Nelder-Mead")
    FittedParameterList = Results["x"]
    TotalParameterList = FindTotalParameterList(FittedParameterList,GuessParameterList,DoFitList)
    ChiSquared = MinimizeFunction(FittedParameterList,*Args)
    return TotalParameterList, ChiSquared

def PrintParameters(ParameterList):
    print("E_Hwp="+str(ParameterList[0]))
    print("R_Hwp="+str(ParameterList[1]))
    print("Delta_Hwp="+str(ParameterList[2]))
    print("E_Der="+str(ParameterList[3]))
    print("R_Der="+str(ParameterList[4]))
    print("DeltaDer="+str(ParameterList[5]))
    print("d="+str(ParameterList[6]))
    print("DeltaCal="+str(ParameterList[7]))
    print("q_in="+str(ParameterList[8]*100)+"%")
    print("u_in="+str(ParameterList[9]*100)+"%")

def SaveParameters(File,ParameterList):
    File.write("E_Hwp="+str(ParameterList[0])+"\n")
    File.write("R_Hwp="+str(ParameterList[1])+"\n")
    File.write("Delta_Hwp="+str(ParameterList[2])+"\n")
    File.write("E_Der="+str(ParameterList[3])+"\n")
    File.write("R_Der="+str(ParameterList[4])+"\n")
    File.write("DeltaDer="+str(ParameterList[5])+"\n")
    File.write("d="+str(ParameterList[6])+"\n")
    File.write("DeltaCal="+str(ParameterList[7])+"\n")
    File.write("q_in="+str(ParameterList[8]*100)+"%\n")
    File.write("u_in="+str(ParameterList[9]*100)+"%")

def ProcessLine(Line):
    Line = Line.replace("\n","")
    Line = Line.split("=")[1]
    return Line

def PlotModelParameter(ParameterList,WavelengthList,ParameterName,Unit="",Title=""):
    plt.figure()

    if(Title==""):
        plt.title("Fitted "+ParameterName+" over wavelength")
    else:
        plt.title(Title)

    plt.xlabel("λ(nm)")
    plt.ylabel(ParameterName+Unit)

    plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)

    plt.scatter(WavelengthList,ParameterList,color="black",s=20,zorder=100)

    plt.grid(linestyle="--")

#--/--Functions--/--#

#-----Main-----#

if __name__ == '__main__':
    
    Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_results/Fits/PolFit"
    
    RunFit = False
    PlotModelParameters = True

    SCExAO_Cal = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0])
    Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(20,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])

    PolParamValueArray = SCExAO_Cal.PolParamValueArray
    ReshapedPolArray = np.swapaxes(np.swapaxes(PolParamValueArray,0,2),1,2)
    PolDerList = SCExAO_Cal.PolImrArray[0]
    PolDoFitList = np.array([False,True,True,False,True,True,True,True,False,False])

    if(RunFit):
        ModelParameterArray = []
        for i in range(0,12):
            PolArgs = (GuessParameterList,PolDerList,ReshapedPolArray[i],True,False,Bounds,PolDoFitList)
            ModelParameterList,ChiSquared = DoFit(PolArgs)
            ModelParameterArray.append(ModelParameterList)
            FittedModel=SCExAO_Model.MatrixModel("",*ModelParameterList[:-2],0,0)
            SCExAO_Cal.PlotParamValues(i,FittedModel,True)
            
            Wavelength = int(SCExAO_Cal.PolLambdaList[0][i])
            SaveFile = open(Prefix+str(Wavelength)+"/"+str(Wavelength)+"Parameters2.txt","w+")
            SaveParameters(SaveFile,ModelParameterList)
            SaveFile.close()

            plt.savefig(Prefix+str(Wavelength)+"/"+str(Wavelength)+"Fit2.png")
            print(str(i)+" is Done")

    if(PlotModelParameters):

        R_Hwp_List = []
        Delta_Hwp_List = []
        R_Der_List = []
        Delta_Der_List = []
        d_List = []
        Delta_Cal_List = []
        WavelengthList = SCExAO_Cal.PolLambdaList[0]

        for i in range(len(WavelengthList)):
            Wavelength = int(WavelengthList[i])
            SaveFile = open(Prefix+str(Wavelength)+"/"+str(Wavelength)+"Parameters.txt","r+")
            LineArray = SaveFile.readlines()
        
            R_Hwp_List.append(float(ProcessLine(LineArray[1])))
            Delta_Hwp_List.append(float(ProcessLine(LineArray[2])))
            R_Der_List.append(float(ProcessLine(LineArray[4])))
            Delta_Der_List.append(float(ProcessLine(LineArray[5])))
            d_List.append(float(ProcessLine(LineArray[6])))
            Delta_Cal_List.append(float(ProcessLine(LineArray[7])))

            SaveFile.close()

        PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength")
        PlotModelParameter(R_Der_List,WavelengthList,"$\Delta_{der}$","($^{\circ}$)","Fitted derotator retardance over wavelength")
        PlotModelParameter(Delta_Hwp_List,WavelengthList,"$\delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate offset over wavelength")
        PlotModelParameter(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Fitted derotator offset over wavelength")
        PlotModelParameter(d_List,WavelengthList,"d","","Fitted polarizer diattenuation over wavelength")
        PlotModelParameter(Delta_Cal_List,WavelengthList,"$\delta_{Cal}$","","Fitted calibration polarizer offset over wavelength")
        plt.show()

#--/--Main--/--#