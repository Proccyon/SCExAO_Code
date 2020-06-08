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
        ChiSquared += np.sum((FittedParamValueArray-ParamValueArray[i])**2)

    print(ChiSquared)
    return ChiSquared

def DoFit(Args,OptimizeMethod="Nelder-Mead"):
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
    Results = scipy.optimize.minimize(MinimizeFunction,GuessParameterList[DoFitList],args=Args,method=OptimizeMethod)
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

def SaveParameters(File,ParameterList,ChiSquared):
    File.write("E_Hwp="+str(ParameterList[0])+"\n")
    File.write("R_Hwp="+str(ParameterList[1])+"\n")
    File.write("Delta_Hwp="+str(ParameterList[2])+"\n")
    File.write("E_Der="+str(ParameterList[3])+"\n")
    File.write("R_Der="+str(ParameterList[4])+"\n")
    File.write("DeltaDer="+str(ParameterList[5])+"\n")
    File.write("d="+str(ParameterList[6])+"\n")
    File.write("DeltaCal="+str(ParameterList[7])+"\n")
    File.write("q_in="+str(ParameterList[8]*100)+"%\n")
    File.write("u_in="+str(ParameterList[9]*100)+"%\n")
    File.write("ChiSquared="+str(ChiSquared))

def ProcessLine(Line):
    Line = Line.replace("\n","")
    Line = Line.split("=")[1]
    return Line

def PlotModelParameter(ParameterList,WavelengthList,ParameterName,Unit="",Title="",SavePlot=False,SaveParameterName="",ymin=None,ymax=None,NewFigure=True):
    if(NewFigure):
        plt.figure()

    if(Title==""):
        plt.title("Fitted "+ParameterName+" over wavelength")
    else:
        plt.title(Title)

    plt.xlabel("λ(nm)")
    plt.ylabel(ParameterName+Unit)

    plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)
    if(not ymin==None and not ymax==None):
        plt.ylim(ymin,ymax)

    plt.scatter(WavelengthList,ParameterList,color="black",s=20,zorder=100)

    plt.grid(linestyle="--")

    if(SavePlot):
        plt.savefig(CreateModelPlotPath(Prefix,PlotCalibrationNumber,SaveParameterName))

def CreateFitPlotPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitPlot{}.png".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateEffPlotPath(Prefix,CalibrationNumber):
    return "{}Calibration{}/EffPlot{}.png".format(Prefix,CalibrationNumber,CalibrationNumber)

def CreateMinEffPath(Prefix,CalibrationNumber):
    return "{}Calibration{}/MinEffPlot{}.png".format(Prefix,CalibrationNumber,CalibrationNumber)

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

def DoRetardanceFit(RetardanceArgs,GuessParameterList,OptimizeMethod="Nelder-Mead"):

    Results = scipy.optimize.minimize(RetardanceMinimizeFunction,GuessParameterList,args=RetardanceArgs,method=OptimizeMethod)
    FittedParameters = Results["x"]
    return FittedParameters

def GetPolyfitY(x,Coefficients):
    y=np.zeros(x.shape)
    
    N = len(Coefficients)
    for i in range(N):
        y += Coefficients[i]*(x**(N-i-1))
    
    return y

#-/-RetardanceFitFunctions-/-#

#--/--Functions--/--#

#-----Main-----#

if __name__ == '__main__':
    

    #---InputParameters---#

    CalibrationNumber = 3
    PlotCalibrationNumber = 6
    Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_results/PolarizedCalibration2/"
    
    RunFit = False
    PlotModelParameters = False
    PlotStokesParameters = False
    PlotEffDiagram = False
    RunRetardanceFit = False
    RunDerotatorFit = True

    InsertOldOffsets = True
    FitOffsets = False
    OptimizeMethod = "Powell"

    EfficiencyColors = ["red","green","blue","cyan"]
    EfficiencyWaveNumbers = [5,10,15,20]

    SCExAO_Cal = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    GuessParameterList = np.array([0,180,0,0,190,0,1,0,0,0],dtype=float)
    Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(180,250),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])

    #GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0])
    #Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(50,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])
    
    if(FitOffsets):
        PolDoFitList = np.array([False,True,True,False,True,True,True,True,False,False])
    else:
        PolDoFitList = np.array([False,True,False,False,True,True,True,False,False,False])

    #-/-InputParameters-/-#

    #---PreviousModelParameters---#

    R_Hwp_List = []
    Delta_Hwp_List = []
    R_Der_List = []
    Delta_Der_List = []
    d_List = []
    Delta_Cal_List = []
    ChiSquaredList = []
    WavelengthList = SCExAO_Cal.PolLambdaList[0]
    FittedModelList = []

    for i in range(len(WavelengthList)):
        Wavelength = int(WavelengthList[i])
        SaveFile = open(CreateFitParametersPath(Prefix,PlotCalibrationNumber,Wavelength),"r+")
        LineArray = SaveFile.readlines()
        
        R_Hwp = float(ProcessLine(LineArray[1]))
        Delta_Hwp = float(ProcessLine(LineArray[2]))
        R_Der = float(ProcessLine(LineArray[4]))
        Delta_Der = float(ProcessLine(LineArray[5]))
        d = float(ProcessLine(LineArray[6]))
        Delta_Cal = float(ProcessLine(LineArray[7]))
        ChiSquared = float(ProcessLine(LineArray[10]))

        R_Hwp_List.append(R_Hwp)
        Delta_Hwp_List.append(Delta_Hwp)
        R_Der_List.append(R_Der)
        Delta_Der_List.append(Delta_Der)
        d_List.append(d)
        Delta_Cal_List.append(Delta_Cal)
        ChiSquaredList.append(ChiSquared)

        FittedModel=SCExAO_Model.MatrixModel("",0,R_Hwp,Delta_Hwp,0,R_Der,Delta_Der,d,Delta_Cal,0,180)
        FittedModelList.append(FittedModel)

        SaveFile.close()

    R_Hwp_List = np.array(R_Hwp_List)
    Delta_Hwp_List = np.array(Delta_Hwp_List)
    R_Der_List = np.array(R_Der_List)
    Delta_Der_List = np.array(Delta_Der_List)
    d_List = np.array(d_List)
    Delta_Cal_List = np.array(Delta_Cal_List)
    ChiSquaredList = np.array(ChiSquaredList)

    #-/-PreviousModelParameters-/-#

    #---DoFit---#

    if(InsertOldOffsets):

        GuessParameterList[2] = np.average(Delta_Hwp_List[-9:])
        #GuessParameterList[5] = np.average(Delta_Der_List)
        GuessParameterList[7] = np.average(Delta_Cal_List[-9:])

        print("AverageDeltaHwp="+str(GuessParameterList[2]))
        print("AverageDeltaCal="+str(GuessParameterList[7]))

    PolParamValueArray = SCExAO_Cal.PolParamValueArray
    
    ReshapedPolArray = np.swapaxes(PolParamValueArray,0,3)
    ReshapedPolArray = np.swapaxes(ReshapedPolArray,1,2)
    ReshapedPolArray = np.swapaxes(ReshapedPolArray,2,3)

    PolDerList = SCExAO_Cal.PolImrArray[0]

    if(RunFit):
        ColorIndex=0
        ModelParameterArray = []
        for i in range(0,22):

            if(i>3):
                Bounds[4] = (50,180)
                GuessParameterList[4] = 90

            PolArgs = (GuessParameterList,PolDerList,ReshapedPolArray[i],True,False,Bounds,PolDoFitList)
            ModelParameterList,ChiSquared = DoFit(PolArgs,OptimizeMethod)
            
            ModelParameterArray.append(ModelParameterList)
            #FittedModel=SCExAO_Model.MatrixModel("",*ModelParameterList[:-2],0,0)

            Wavelength = int(SCExAO_Cal.PolLambdaList[0][i])
            SaveFile = open(CreateFitParametersPath(Prefix,CalibrationNumber,Wavelength),"w+")

            SaveParameters(SaveFile,ModelParameterList,ChiSquared)
            SaveFile.close()

            print(str(i)+" is Done")
    
    #-/-DoFit-/-#

    #---DoModelPlots---#

    if(PlotModelParameters):

        PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength",True,"R_Hwp")
        HalleRetardance = hwp_halle.HalleRetardance(WavelengthList)
        plt.plot(WavelengthList,HalleRetardance,label="HalleRetardance")
        plt.legend()

        PlotModelParameter(R_Der_List,WavelengthList,"$\Delta_{der}$","($^{\circ}$)","Fitted derotator retardance over wavelength",True,"R_Der")
        PlotModelParameter(Delta_Hwp_List,WavelengthList,"$\delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate offset over wavelength",True,"Delta_Hwp",-2,3)
        PlotModelParameter(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Fitted derotator offset over wavelength",True,"Delta_Der")
        PlotModelParameter(d_List,WavelengthList,"d","","Fitted polarizer diattenuation over wavelength",True,"d")
        PlotModelParameter(Delta_Cal_List,WavelengthList,"$\delta_{Cal}$","","Fitted calibration polarizer offset over wavelength",True,"Delta_Cal",-2,3)
        PlotModelParameter(ChiSquaredList,WavelengthList,"$\chi^2$","","Sum of squared residuals of fits over wavelength",True,"ChiSquared")

        plt.figure()
        plt.ylim(-2,3)
        plt.scatter(WavelengthList,Delta_Cal_List-2*Delta_Hwp_List)

        plt.show()

    #-/-DoModelPlots-/-#

    if(PlotStokesParameters):
        for i in range(22):
            SCExAO_Cal.PlotParamValues(i,FittedModelList[i],True)
            plt.savefig(CreateFitPlotPath(Prefix,PlotCalibrationNumber,int(WavelengthList[i])))

        plt.show()

    if(PlotEffDiagram):
        
        plt.figure()
        ColorIndex=0

        for i in range(22):
            if(i in EfficiencyWaveNumbers):
                SCExAO_Cal.PlotPolarimetricEfficiency(i,EfficiencyColors[ColorIndex],FittedModelList[i])
                ColorIndex+=1

        legend = plt.legend(loc=4,fontsize=7)
        legend.set_zorder(200)
        plt.savefig(CreateEffPlotPath(Prefix,PlotCalibrationNumber))

        SCExAO_Cal.PlotMinimumEfficiency(FittedModelList)  
        plt.savefig(CreateMinEffPath(Prefix,PlotCalibrationNumber))

        plt.show()


    if(RunRetardanceFit):

        BadIndex = [4,5]

        n_0_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.07044083,1.00585997E-2,1.10202242,100,0.28604141)
        n_e_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.09509924,1.02101864E-2,1.15662475,100,0.28851804)
        n_0_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.48755108,0.04338408**2,0.39875031,0.09461442**2,2.3120353,23.793604**2,0)
        n_e_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.41344023,0.03684262**2,0.50497499,0.09076162**2,2.4904862,23.771995**2,0)

        n_0_quartz = n_0_quartz_function(np.delete(WavelengthList,BadIndex))
        n_e_quartz = n_e_quartz_function(np.delete(WavelengthList,BadIndex))
        n_0_MgF2 = n_0_MgF2_function(np.delete(WavelengthList,BadIndex))
        n_e_MgF2 = n_e_MgF2_function(np.delete(WavelengthList,BadIndex))

        ThicknessBounds = [[1E-6,2E-1],[1E-6,2E-1]]
        ThicknessGuessList = [1E-3,1E-3]
        CrossAxes = True
        SwitchMaterials = True
        
        RetardanceArgs = (np.delete(WavelengthList,BadIndex),np.delete(R_Hwp_List,BadIndex)*np.pi/180,n_0_quartz,n_e_quartz,n_0_MgF2,n_e_MgF2,CrossAxes,SwitchMaterials,ThicknessBounds)

        Fitted_d_quartz,Fitted_d_MgF2 = DoRetardanceFit(RetardanceArgs,ThicknessGuessList,OptimizeMethod)

        Wavelength_Wide = np.linspace(1000,2500,400)

        n_0_quartz_Wide = n_0_quartz_function(Wavelength_Wide)
        n_e_quartz_Wide = n_e_quartz_function(Wavelength_Wide)
        n_0_MgF2_Wide = n_0_MgF2_function(Wavelength_Wide)
        n_e_MgF2_Wide = n_e_MgF2_function(Wavelength_Wide)

        FittedRetardance = (180/np.pi)*CalculateRetardance(Wavelength_Wide,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz_Wide,n_e_quartz_Wide,n_0_MgF2_Wide,n_e_MgF2_Wide,CrossAxes,SwitchMaterials)

        PlotModelParameter(np.delete(R_Hwp_List,BadIndex),np.delete(WavelengthList,BadIndex),"$\Delta_{Hwp}$","($^{\circ}$)","Fit of quartz and $MgF_2$ plates to measured hwp retardance",False,"R_Hwp")
        plt.plot(Wavelength_Wide,FittedRetardance,label="Fitted retardance")
        plt.legend()
        plt.show()

        print("d_quartz="+str(Fitted_d_quartz*1000)+"mm")
        print("d_MgF2="+str(Fitted_d_MgF2*1000)+"mm")

    if(RunDerotatorFit):
        
        fig1 = plt.figure()
        frame1=fig1.add_axes((.13,.36,.77,.57))
        frame1.set_xticklabels([])

        PolyfitDegree = 4

        NewWavelengthList = np.delete(WavelengthList,4)
        New_R_Der_List = np.delete(R_Der_List,4)

        PolyfitCoefficients = np.polyfit(NewWavelengthList,New_R_Der_List,PolyfitDegree)

        Wavelength_Wide = np.linspace(1000,2500,400)
        R_Der_Fit = GetPolyfitY(Wavelength_Wide,PolyfitCoefficients)
        
        PlotModelParameter(New_R_Der_List,NewWavelengthList,"$\Delta_{Der}$","($^{\circ}$)","Polynomial fit(deg="+str(PolyfitDegree)+") on derotator retardance",False,"R_Der",NewFigure=False)
        plt.scatter(WavelengthList[4],R_Der_List[4],color="red",s=20,label="outliers")
        plt.plot(Wavelength_Wide,R_Der_Fit,label="Polynomial fit")
        plt.legend()

        frame2=fig1.add_axes((.13,.1,.77,.23))

        plt.axhline(y=0,color="black")
        
        R_Der_Residuals = R_Der_List - GetPolyfitY(WavelengthList,PolyfitCoefficients)

        plt.scatter(np.delete(WavelengthList,4),np.delete(R_Der_Residuals,4),color="black",s=20,zorder=20)
        plt.scatter(WavelengthList[4],R_Der_Residuals[4],color="red",s=20,zorder=20)

        plt.grid(linestyle="--")
        plt.yticks(np.arange(-10,15,5))
        plt.ylim(-11,11)
        plt.xlabel("λ(nm)")
        plt.ylabel("Residuals($^{\circ}$)")
        plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)

        plt.show()

#--/--Main--/--#