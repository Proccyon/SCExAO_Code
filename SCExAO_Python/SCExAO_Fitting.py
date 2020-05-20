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
import hwp_halle
#--/--Imports--/--#

#-----Functions-----#

#---StokesFitFunctions---#

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
    
    ChiSquared = 0
    for i in range(len(ParamValueArray)):
        ChiSquared += np.sqrt(np.sum((FittedParamValueArray-ParamValueArray[i])**2))
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

def PlotModelParameter(ParameterList,WavelengthList,ParameterName,Unit="",Title="",SavePlot=False,SaveParameterName=""):
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

    if(SavePlot):
        plt.savefig(CreateModelPlotPath(Prefix,CalibrationNumber,SaveParameterName))

def CreateFitPlotPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitPlot{}.png".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)
    
def CreateFitParametersPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitParameters{}.txt".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateModelPlotPath(Prefix,CalibrationNumber,ModelParameter):
    return "{}Calibration{}/ModelParameterPlots/{}Plot{}.png".format(Prefix,CalibrationNumber,ModelParameter,CalibrationNumber)

#-/-StokesFitFunctions-/-#

#---RetardanceFitFunctions---#

#Formula to find refractive index from refractiveindex.com
def RefractiveIndexFormula(Wavelength,A,B,C,D,E,F,G):
    NewWavelength = Wavelength / 1000 #nanometer to micrometer
    Part1 = A*NewWavelength**2 / (NewWavelength**2-B)
    Part2 = C*NewWavelength**2 / (NewWavelength**2-D)
    Part3 = E*NewWavelength**2 / (NewWavelength**2-F)
    return np.sqrt(Part1+Part2+Part3+G+1)

def CalculateRetardance(Wavelength,d1,d2,n1_0,n1_e,n2_0,n2_e,CrossAxes=True,SwitchMaterials=False):
    if(CrossAxes):
        if(SwitchMaterials):
            return (2*np.pi*1E9/Wavelength)*(d2*(n2_e-n2_0)-d1*(n1_e-n1_0))
        else:
            return (2*np.pi*1E9/Wavelength)*(d1*(n1_e-n1_0)-d2*(n2_e-n2_0))
    else:
        return (2*np.pi*1E9/Wavelength)*(d1*(n1_e-n1_0)+d2*(n2_e-n2_0))


def RetardanceMinimizeFunction(ParameterList,Wavelength,RealRetardance,n1_0,n1_e,n2_0,n2_e,CrossAxes,SwitchMaterials,Bounds):
    d1 = ParameterList[0]
    d2 = ParameterList[1]

    #Return high value if out of bounds
    if(d1<Bounds[0][0] or d1>Bounds[0][1] or d2<Bounds[1][0] or d2>Bounds[1][1]):
        return 1E9

    FittedRetardance = CalculateRetardance(Wavelength,d1,d2,n1_0,n1_e,n2_0,n2_e,CrossAxes,SwitchMaterials)
    return np.sqrt(np.sum((FittedRetardance - RealRetardance)**2))

def DoRetardanceFit(RetardanceArgs,GuessParameterList):

    Results = scipy.optimize.minimize(RetardanceMinimizeFunction,GuessParameterList,args=RetardanceArgs,method="Nelder-Mead")
    FittedParameters = Results["x"]
    return FittedParameters

#-/-RetardanceFitFunctions-/-#

#--/--Functions--/--#

#-----Main-----#

if __name__ == '__main__':
    

    #---InputParameters---#

    CalibrationNumber = 5
    PlotCalibrationNumber = 2
    Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_results/PolarizedCalibration/"
    
    RunFit = False
    PlotModelParameters = False
    InsertOldOffsets = False
    RunRetardanceFit = True

    SCExAO_Cal = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    GuessParameterList = np.array([0,180,0,0,190,0,1,0,0,0],dtype=float)
    Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(180,250),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])

    #GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0])
    #Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(50,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])
    
    PolDoFitList = np.array([False,True,False,False,True,False,True,False,False,False])

    #-/-InputParameters-/-#

    #---PreviousModelParameters---#

    R_Hwp_List = []
    Delta_Hwp_List = []
    R_Der_List = []
    Delta_Der_List = []
    d_List = []
    Delta_Cal_List = []
    WavelengthList = SCExAO_Cal.PolLambdaList[0]

    for i in range(len(WavelengthList)):
        Wavelength = int(WavelengthList[i])
        SaveFile = open(CreateFitParametersPath(Prefix,PlotCalibrationNumber,Wavelength),"r+")
        LineArray = SaveFile.readlines()
        
        R_Hwp_List.append(float(ProcessLine(LineArray[1])))
        Delta_Hwp_List.append(float(ProcessLine(LineArray[2])))
        R_Der_List.append(float(ProcessLine(LineArray[4])))
        Delta_Der_List.append(float(ProcessLine(LineArray[5])))
        d_List.append(float(ProcessLine(LineArray[6])))
        Delta_Cal_List.append(float(ProcessLine(LineArray[7])))

        SaveFile.close()

    #-/-PreviousModelParameters-/-#

    #---DoFit---#

    if(InsertOldOffsets):

        GuessParameterList[2] = np.average(Delta_Hwp_List)
        GuessParameterList[5] = np.average(Delta_Der_List)
        GuessParameterList[7] = np.average(Delta_Cal_List)

    PolParamValueArray = SCExAO_Cal.PolParamValueArray
    
    ReshapedPolArray = np.swapaxes(PolParamValueArray,0,3)
    ReshapedPolArray = np.swapaxes(ReshapedPolArray,1,2)
    ReshapedPolArray = np.swapaxes(ReshapedPolArray,2,3)

    PolDerList = SCExAO_Cal.PolImrArray[0]

    if(RunFit):
        ModelParameterArray = []
        for i in range(0,22):

            if(i>3):
                Bounds[4] = (50,180)
                GuessParameterList[4] = 90

            PolArgs = (GuessParameterList,PolDerList,ReshapedPolArray[i],True,False,Bounds,PolDoFitList)
            ModelParameterList,ChiSquared = DoFit(PolArgs)
            ModelParameterArray.append(ModelParameterList)
            FittedModel=SCExAO_Model.MatrixModel("",*ModelParameterList[:-2],0,0)
            SCExAO_Cal.PlotParamValues(i,FittedModel,True)
            
            Wavelength = int(SCExAO_Cal.PolLambdaList[0][i])
            SaveFile = open(CreateFitParametersPath(Prefix,CalibrationNumber,Wavelength),"w+")
            #OldString = Prefix+str(Wavelength)+"/"+str(Wavelength)+ParameterFileName+".txt"
            SaveParameters(SaveFile,ModelParameterList)
            SaveFile.close()

            plt.savefig(CreateFitPlotPath(Prefix,CalibrationNumber,Wavelength))
            #OldString = Prefix+str(Wavelength)+"/"+str(Wavelength)+PlotFileName+".png"
            print(str(i)+" is Done")

    #-/-DoFit-/-#

    #---DoModelPlots---#

    if(PlotModelParameters):

        PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength",True,"R_Hwp")
        HalleRetardance = hwp_halle.HalleRetardance(WavelengthList)
        plt.plot(WavelengthList,HalleRetardance,label="HalleRetardance")
        plt.legend()

        PlotModelParameter(R_Der_List,WavelengthList,"$\Delta_{der}$","($^{\circ}$)","Fitted derotator retardance over wavelength",True,"R_Der")
        PlotModelParameter(Delta_Hwp_List,WavelengthList,"$\delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate offset over wavelength",True,"Delta_Hwp")
        PlotModelParameter(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Fitted derotator offset over wavelength",True,"Delta_Der")
        PlotModelParameter(d_List,WavelengthList,"d","","Fitted polarizer diattenuation over wavelength",True,"d")
        PlotModelParameter(Delta_Cal_List,WavelengthList,"$\delta_{Cal}$","","Fitted calibration polarizer offset over wavelength",True,"Delta_Cal")
        plt.show()

    #-/-DoModelPlots-/-#


if(RunRetardanceFit):
    #NewWavelength = np.linspace(1000,2500,100)
    n_0_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.07044083,1.00585997E-2,1.10202242,100,0.28604141)
    n_e_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.09509924,1.02101864E-2,1.15662475,100,0.28851804)
    n_0_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.48755108,0.04338408**2,0.39875031,0.09461442**2,2.3120353,23.793604**2,0)
    n_e_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.41344023,0.03684262**2,0.50497499,0.09076162**2,2.4904862,23.771995**2,0)

    n_0_quartz = n_0_quartz_function(WavelengthList)
    n_e_quartz = n_e_quartz_function(WavelengthList)
    n_0_MgF2 = n_0_MgF2_function(WavelengthList)
    n_e_MgF2 = n_e_MgF2_function(WavelengthList)

    ThicknessBounds = [[1E-6,1E-1],[1E-6,1E-1]]
    ThicknessGuessList = [0.0001,0.0001]
    CrossAxes = True
    SwitchMaterials = True

    RetardanceArgs = (WavelengthList,R_Hwp_List,n_0_quartz,n_e_quartz,n_0_MgF2,n_e_MgF2,CrossAxes,SwitchMaterials,ThicknessBounds)

    Fitted_d_quartz,Fitted_d_MgF2 = DoRetardanceFit(RetardanceArgs,ThicknessGuessList)

    Wavelength_Wide = np.linspace(1000,2500,400)

    n_0_quartz_Wide = n_0_quartz_function(Wavelength_Wide)
    n_e_quartz_Wide = n_e_quartz_function(Wavelength_Wide)
    n_0_MgF2_Wide = n_0_MgF2_function(Wavelength_Wide)
    n_e_MgF2_Wide = n_e_MgF2_function(Wavelength_Wide)

    FittedRetardance = CalculateRetardance(Wavelength_Wide,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz_Wide,n_e_quartz_Wide,n_0_MgF2_Wide,n_e_MgF2_Wide,CrossAxes,SwitchMaterials)

    PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fit of quartz and $MgF_2$ plates to measured hwp retardance",True,"R_Hwp")
    plt.plot(Wavelength_Wide,FittedRetardance,label="Fitted retardance")
    plt.legend()
    plt.show()

    print("d_quartz="+str(Fitted_d_quartz*1000)+"mm")
    print("d_MgF2="+str(Fitted_d_MgF2*1000)+"mm")

#--/--Main--/--#