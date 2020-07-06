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


def GetChiSquared(Model,ParamValueArray,DerList,S_In,UsePolarizer,DerMethod):

    FittedParamValueArray = Model.FindParameterArray(DerList*np.pi/180,S_In,UsePolarizer=UsePolarizer,DerMethod=DerMethod)

    ChiSquared = 0
    for i in range(len(ParamValueArray)):
        ChiSquared += np.sum((FittedParamValueArray-ParamValueArray[i])**2)

    return ChiSquared


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

    #FittedParamValueArray = FittedModel.FindParameterArray(DerList*np.pi/180,S_In,UsePolarizer=UsePolarizer,DerMethod=DerMethod)
    
    #ChiSquared = 0
    #for i in range(len(ParamValueArray)):
    #    ChiSquared += np.sum((FittedParamValueArray-ParamValueArray[i])**2)

    ChiSquared = GetChiSquared(FittedModel,ParamValueArray,DerList,S_In,UsePolarizer,DerMethod)

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
    print(Results)
    FittedParameterList = Results["x"]
    ErrorList = GetParameterErrors(Results)

    TotalParameterList = FindTotalParameterList(FittedParameterList,GuessParameterList,DoFitList)
    TotalErrorList = FindTotalParameterList(ErrorList,np.zeros(GuessParameterList.shape),DoFitList)

    ChiSquared = MinimizeFunction(FittedParameterList,*Args)
    return TotalParameterList,TotalErrorList, ChiSquared


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

def SaveParameters(File,ParameterList,ErrorList,ChiSquared):
    File.write("E_Hwp="+str(ParameterList[0])+"$\pm$"+str(ErrorList[0])+"\n")
    File.write("R_Hwp="+str(ParameterList[1])+"$\pm$"+str(ErrorList[1])+"\n")
    File.write("Delta_Hwp="+str(ParameterList[2])+"$\pm$"+str(ErrorList[2])+"\n")
    File.write("E_Der="+str(ParameterList[3])+"$\pm$"+str(ErrorList[3])+"\n")
    File.write("R_Der="+str(ParameterList[4])+"$\pm$"+str(ErrorList[4])+"\n")
    File.write("DeltaDer="+str(ParameterList[5])+"$\pm$"+str(ErrorList[5])+"\n")
    File.write("d="+str(ParameterList[6])+"$\pm$"+str(ErrorList[6])+"\n")
    File.write("DeltaCal="+str(ParameterList[7])+"$\pm$"+str(ErrorList[7])+"\n")
    File.write("q_in="+str(ParameterList[8]*100)+"$\pm$"+str(ErrorList[8])+"%\n")
    File.write("u_in="+str(ParameterList[9]*100)+"$\pm$"+str(ErrorList[9])+"%\n")
    File.write("ChiSquared="+str(ChiSquared))

def ProcessLine(Line):
    Line = Line.replace("\n","")
    Line = Line.split("=")[1]
    return Line

def PlotModelParameter(ParameterList,WavelengthList,ParameterName,Unit="",Title="",SavePlot=False,SaveParameterName="",ymin=None,ymax=None,NewFigure=True,DataColor="black",MarkerStyle="o",ScatterSize=20,DataLabel=""):
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

    plt.scatter(WavelengthList,ParameterList,color=DataColor,s=ScatterSize,zorder=100,label=DataLabel,alpha=1,marker=MarkerStyle,linewidths=1)

    plt.grid(linestyle="--")
    if(DataLabel != ""):
        legend = plt.legend(fontsize=7)
        legend.set_zorder(200)

    if(SavePlot):
        plt.savefig(CreateModelPlotPath(Prefix,PlotCalibrationNumber,SaveParameterName))

def PlotModelParameterFit(ParameterList,WavelengthList,ParameterName,Unit="",Title="",SavePlot=False,SaveParameterName="",ymin=None,ymax=None,yResMin=None,yResMax=None,PolyfitDegree=4,OutlierIndices=[]):

        fig1 = plt.figure()
        frame1=fig1.add_axes((.13,.36,.77,.57))
        frame1.set_xticklabels([])

        NewWavelengthList = np.delete(WavelengthList,OutlierIndices)
        NewParameterList = np.delete(ParameterList,OutlierIndices)

        PolyfitCoefficients = np.polyfit(NewWavelengthList,NewParameterList,PolyfitDegree)

        Wavelength_Wide = np.linspace(1000,2500,400)
        Parameter_Fit = GetPolyfitY(Wavelength_Wide,PolyfitCoefficients)

        PlotModelParameter(NewParameterList,NewWavelengthList,ParameterName,Unit,Title,False,SaveParameterName,ymin,ymax,False)
        if(len(OutlierIndices)>0):
            plt.scatter(WavelengthList[OutlierIndices],ParameterList[OutlierIndices],color="red",s=20,label="outliers")

        plt.plot(Wavelength_Wide,Parameter_Fit,label="Polynomial fit")
        plt.legend(fontsize=8)

        plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)

        frame2=fig1.add_axes((.13,.1,.77,.23))

        plt.axhline(y=0,color="black")
        
        ParameterResiduals = ParameterList - GetPolyfitY(WavelengthList,PolyfitCoefficients)

        plt.scatter(np.delete(WavelengthList,OutlierIndices),np.delete(ParameterResiduals,OutlierIndices),color="black",s=20,zorder=20)
        plt.scatter(WavelengthList[OutlierIndices],ParameterResiduals[OutlierIndices],color="red",s=20,zorder=20)

        plt.grid(linestyle="--")
        plt.yticks(np.linspace(yResMin,yResMax,5))
        plt.ylim(yResMin-0.1*(yResMax-yResMin),yResMax+0.1*(yResMax-yResMin))
        plt.xlabel("λ(nm)")
        plt.ylabel("Residuals"+Unit)
        plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)

        if(SavePlot):
            plt.savefig(CreateModelFitPlotPath(Prefix,PlotCalibrationNumber,SaveParameterName))

        return PolyfitCoefficients


def CreateFitPlotPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitPlot{}.png".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateSingleEffPlotPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}SingleEffPlot{}.png".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateSingleAolpPlotPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}SingleAolpPlot{}.png".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateEffPlotPath(Prefix,CalibrationNumber):
    return "{}Calibration{}/EffPlot{}.png".format(Prefix,CalibrationNumber,CalibrationNumber)

def CreateMinEffPath(Prefix,CalibrationNumber):
    return "{}Calibration{}/MinEffPlot{}.png".format(Prefix,CalibrationNumber,CalibrationNumber)

def CreateFitParametersPath(Prefix,CalibrationNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitParameters{}.txt".format(Prefix,CalibrationNumber,Wavelength,Wavelength,CalibrationNumber)

def CreateModelPlotPath(Prefix,CalibrationNumber,ModelParameter):
    return "{}Calibration{}/ModelParameterPlots/{}Plot{}.png".format(Prefix,CalibrationNumber,ModelParameter,CalibrationNumber)

def CreateModelFitPlotPath(Prefix,CalibrationNumber,ModelParameter):
    return "{}Calibration{}/ModelParameterPlots/{}Plot{}Polyfit.png".format(Prefix,CalibrationNumber,ModelParameter,CalibrationNumber)

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

def GetModelParameters(WavelengthList,CalNumber,Prefix):

    R_Hwp_List = []
    Delta_Hwp_List = []
    R_Der_List = []
    Delta_Der_List = []
    d_List = []
    Delta_Cal_List = []
    ChiSquaredList = []
    FittedModelList = []

    for i in range(len(WavelengthList)):
        Wavelength = int(WavelengthList[i])
        SaveFile = open(CreateFitParametersPath(Prefix,CalNumber,Wavelength),"r+")
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

    return np.array(R_Hwp_List),np.array(Delta_Hwp_List),np.array(R_Der_List),np.array(Delta_Der_List),np.array(d_List),np.array(Delta_Cal_List),np.array(ChiSquaredList),np.array(FittedModelList)

#-/-RetardanceFitFunctions-/-#


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

def GetParameterErrors(FitResults):
    #ftol = 2.220446049250313e-09
    #tmp_i = np.zeros(len(FitResults.x))

    #ErrorList = []
    #for i in range(len(FitResults.x)):
    #    tmp_i[i] = 1.0
    #    hess_inv_i = FitResults.hess_inv(tmp_i)[i]
    #    uncertainty_i = np.sqrt(max(1, abs(FitResults.fun)) * ftol * hess_inv_i)
    #    tmp_i[i] = 0.0
    #    print('x^{0} = {1:12.4e} ± {2:.1e}'.format(i, FitResults.x[i], uncertainty_i))
    #    ErrorList.append(uncertainty_i)

    #return ErrorList

    return np.diag(FitResults.hess_inv.todense())
#--/--Functions--/--#

#-----Main-----#

if __name__ == '__main__':
    

    #---InputParameters---#

    CalibrationNumber = 9
    PlotCalibrationNumber = 8
    PlotCalibrationNumbers = [3,5,6]
    MarkerStyles = [">","<","o"]
    Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_results/PolarizedCalibration2/"
    
    RunFit = False
    PlotModelParameters = False
    PlotModelParametersCombined = False
    PlotStokesParameters = False
    PlotEffDiagram = False
    Plot_AOLP_Diagram = True
    RunRetardanceFit = False
    RunDerotatorFit = False
    FindSmoothChiSquared = False
    TestPlot=False

    InsertOldOffsets = True
    FitOffsets = False
    OptimizeMethod = "Powell"

    #EfficiencyColors = ["red","green","blue","cyan"]
    EfficiencyColors = ["red","green","blue","cyan"]#,"darkorange","blueviolet","dodgerblue","magenta"]#]
    #EfficiencyWaveNumbers = [4,9,14,19]
    #EfficiencyWaveNumbers = [0,1,2,3,4,5,6]
    #EfficiencyWaveNumbers = [7,8,9,10,11,12,13]
    EfficiencyWaveNumbers = [14,15,16,17,18,19,20,21]
    AOLP_WaveNumbers = [3,7,9,19]

    SCExAO_Cal = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))

    GuessParameterList = np.array([0,180,0,0,190,0,1,0,0,0],dtype=float)
    Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(180,300),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])

    #GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0])
    #Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(50,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])
    
    if(FitOffsets):
        PolDoFitList = np.array([False,True,True,False,True,True,True,True,False,False])
    else:
        PolDoFitList = np.array([False,True,True,False,True,False,True,False,False,False])

    #-/-InputParameters-/-#

    #---PreviousModelParameters---#

    WavelengthList = SCExAO_Cal.PolLambdaList[0]
    R_Hwp_List,Delta_Hwp_List,R_Der_List,Delta_Der_List,d_List,Delta_Cal_List,ChiSquaredList,FittedModelList = GetModelParameters(WavelengthList,PlotCalibrationNumber,Prefix)

    #-/-PreviousModelParameters-/-#

    #---DoFit---#

    if(InsertOldOffsets):

        #GuessParameterList[2] = np.average(Delta_Hwp_List[-9:])
        GuessParameterList[5] = np.average(Delta_Der_List)
        GuessParameterList[7] = np.average(Delta_Cal_List[-9:])

        #print("AverageDeltaHwp="+str(GuessParameterList[2]))
        print("AverageDeltaDer="+str(GuessParameterList[5]))
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
                Bounds[4] = (30,180)
                GuessParameterList[4] = 90

            PolArgs = (GuessParameterList,PolDerList,ReshapedPolArray[i],True,False,Bounds,PolDoFitList)
            ModelParameterList,ErrorList,ChiSquared = DoFit(PolArgs,OptimizeMethod)
            
            ModelParameterArray.append(ModelParameterList)
            #FittedModel=SCExAO_Model.MatrixModel("",*ModelParameterList[:-2],0,0)

            Wavelength = int(SCExAO_Cal.PolLambdaList[0][i])
            SaveFile = open(CreateFitParametersPath(Prefix,CalibrationNumber,Wavelength),"w+")

            SaveParameters(SaveFile,ModelParameterList,ErrorList,ChiSquared)
            SaveFile.close()

            print(str(i)+" is Done")
    
    #-/-DoFit-/-#

    #---DoModelPlots---#

    if(PlotModelParameters):

        
        #HalleRetardance = hwp_halle.HalleRetardance(WavelengthList)
        #plt.plot(WavelengthList,HalleRetardance,label="HalleRetardance")
        #plt.legend()

        PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength",True,"R_Hwp",165,185)
        PlotModelParameter(R_Der_List,WavelengthList,"$\Delta_{der}$","($^{\circ}$)","Fitted derotator retardance over wavelength",True,"R_Der",45,260)
        PlotModelParameter(Delta_Hwp_List,WavelengthList,"$\delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate offset over wavelength",True,"Delta_Hwp")
        PlotModelParameter(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Fitted derotator offset over wavelength",True,"Delta_Der")
        plt.axhline(y=GuessParameterList[5],linestyle="--",color="blue",label="Average $\delta_{der}$")
        plt.legend(fontsize=8)
        PlotModelParameter(d_List,WavelengthList,"$\epsilon_{cal}$","","Fitted polarizer diattenuation over wavelength",True,"d")
        PlotModelParameter(Delta_Cal_List,WavelengthList,"$\delta_{Cal}$","","Fitted calibration polarizer offset over wavelength",True,"Delta_Cal")
        plt.axhline(y=GuessParameterList[7],linestyle="--",color="blue",label="Average $\delta_{Cal}$")
        plt.legend(fontsize=8)
        PlotModelParameter(ChiSquaredList,WavelengthList,"$\chi^2$","","Sum of squared residuals of fits over wavelength",True,"ChiSquared")

        #plt.figure()
        #plt.ylim(-2,3)
        #plt.scatter(WavelengthList,Delta_Cal_List-2*Delta_Hwp_List)

        plt.show()

    if(PlotModelParametersCombined):

        #PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength",True,"R_Hwp")
        #HalleRetardance = hwp_halle.HalleRetardance(WavelengthList)
        #plt.plot(WavelengthList,HalleRetardance,label="HalleRetardance")
        #plt.legend()
        DataColors = ["blue","goldenrod","red"]
        MarkerStyles = ["o","o","o"]
        ScatterSizes=[80,35,10]
        DataLabels=["Fit1","Fit2","Fit3"]

        for i in range(len(PlotCalibrationNumbers)):

            PlotCalNumber = PlotCalibrationNumbers[i]

            R_Hwp_List,Delta_Hwp_List,R_Der_List,Delta_Der_List,d_List,Delta_Cal_List,ChiSquaredList,FittedModelList = GetModelParameters(WavelengthList,PlotCalNumber,Prefix)
            
            plt.figure(0)
            PlotModelParameter(R_Der_List,WavelengthList,"$\Delta_{der}$","($^{\circ}$)","Fitted derotator retardance over wavelength",False,"R_Der",NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(1)
            PlotModelParameter(Delta_Hwp_List,WavelengthList,"$\delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate offset over wavelength",False,"Delta_Hwp",-4.5,4.5,NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(2)
            PlotModelParameter(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Fitted derotator offset over wavelength",False,"Delta_Der",NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(3)
            PlotModelParameter(d_List,WavelengthList,"$\epsilon_{\mathmr{cal}}$","","Fitted polarizer diattenuation over wavelength",False,"d",NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(4)
            PlotModelParameter(Delta_Cal_List,WavelengthList,"$\delta_{Cal}$","($^{\circ}$)","Fitted calibration polarizer offset over wavelength",False,"Delta_Cal",-4.5,4.5,NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(5)
            PlotModelParameter(ChiSquaredList,WavelengthList,"$\chi^2$","","Sum of squared residuals of fits over wavelength",False,"ChiSquared",NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(6)
            PlotModelParameter(Delta_Cal_List-2*Delta_Hwp_List,WavelengthList,"$\delta_{Cal}$$-2\delta_{Hwp}$","($^{\circ}$)","HWP and polarizer offset difference over wavelength",False,"Delta_Diff",-4.5,4.5,NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])
            plt.figure(7)
            PlotModelParameter(R_Hwp_List,WavelengthList,"$\Delta_{Hwp}$","($^{\circ}$)","Fitted Half-wave plate retardance over wavelength",False,"R_Hwp",NewFigure=False,DataColor=DataColors[i],MarkerStyle=MarkerStyles[i],ScatterSize=ScatterSizes[i],DataLabel=DataLabels[i])

        #plt.figure()
        #plt.ylim(-2,3)
        #plt.scatter(WavelengthList,Delta_Cal_List-2*Delta_Hwp_List)

        plt.show()

    #-/-DoModelPlots-/-#

    if(PlotStokesParameters):
        for i in range(22):
            SCExAO_Cal.PlotParamValues(i,FittedModelList[i],True)
            plt.savefig(CreateFitPlotPath(Prefix,PlotCalibrationNumber,int(WavelengthList[i])))

        plt.show()

    if(PlotEffDiagram):
        
        ColorIndex=0

        for i in range(22):
            #if(i in EfficiencyWaveNumbers):
            #    plt.figure(1)
            #    SCExAO_Cal.PlotPolarimetricEfficiency(i,EfficiencyColors[ColorIndex],FittedModelList[i])
            #    ColorIndex+=1

            plt.figure()
            SCExAO_Cal.PlotPolarimetricEfficiency(i,"dodgerblue",FittedModelList[i])
            plt.savefig(CreateSingleEffPlotPath(Prefix,PlotCalibrationNumber,int(WavelengthList[i])))
            
            
        legend = plt.legend(fontsize=7)
        legend.set_zorder(200)
        
        #plt.savefig(CreateEffPlotPath(Prefix,PlotCalibrationNumber))

        #SCExAO_Cal.PlotMinimumEfficiency(FittedModelList)  
        #plt.savefig(CreateMinEffPath(Prefix,PlotCalibrationNumber))

        #SCExAO_Cal.PlotPolarimetricEfficiency(9,"black",FittedModelList[9])
        #plt.legend(fontsize=8)

        plt.show()


    if(Plot_AOLP_Diagram):

        #plt.figure()
        #for i in range(len(AOLP_WaveNumbers)):
        #    SCExAO_Cal.Plot_AOLP_Offset(AOLP_WaveNumbers[i],FittedModelList[AOLP_WaveNumbers[i]],EfficiencyColors[i])

        #plt.legend(fontsize=8)
        
        #SCExAO_Cal.Plot_Max_AOLP_Offset(FittedModelList)

        #plt.figure()
        #SCExAO_Cal.Plot_AOLP_Offset(8,FittedModelList[8],"black")
        #plt.legend(fontsize=9)

        for i in range(22):
            plt.figure()
            SCExAO_Cal.Plot_AOLP_Offset(i,FittedModelList[i],"red")
            plt.savefig(CreateSingleAolpPlotPath(Prefix,PlotCalibrationNumber,int(WavelengthList[i])))
            
        #plt.show()

    if(RunRetardanceFit):

        #BadIndex = [4,5]
        BadIndex=[0,1,2,3,4]

        n_0_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.07044083,1.00585997E-2,1.10202242,100,0.28604141)
        n_e_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.09509924,1.02101864E-2,1.15662475,100,0.28851804)
        n_0_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.48755108,0.04338408**2,0.39875031,0.09461442**2,2.3120353,23.793604**2,0)
        n_e_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.41344023,0.03684262**2,0.50497499,0.09076162**2,2.4904862,23.771995**2,0)

        n_0_quartz = n_0_quartz_function(WavelengthList)
        n_e_quartz = n_e_quartz_function(WavelengthList)
        n_0_MgF2 = n_0_MgF2_function(WavelengthList)
        n_e_MgF2 = n_e_MgF2_function(WavelengthList)

        ThicknessBounds = [[1E-6,2E-1],[1E-6,2E-1]]
        ThicknessGuessList = [1.6E-3,1.25E-3]
        CrossAxes = True
        SwitchMaterials = True
        
        RetardanceArgs = (np.delete(WavelengthList,BadIndex),np.delete(R_Hwp_List,BadIndex)*np.pi/180,np.delete(n_0_quartz,BadIndex),np.delete(n_e_quartz,BadIndex),np.delete(n_0_MgF2,BadIndex),np.delete(n_e_MgF2,BadIndex),CrossAxes,SwitchMaterials,ThicknessBounds)

        Fitted_d_quartz,Fitted_d_MgF2 = DoRetardanceFit(RetardanceArgs,ThicknessGuessList,OptimizeMethod)

        print("d_quartz="+str(Fitted_d_quartz*1000)+"mm")
        print("d_MgF2="+str(Fitted_d_MgF2*1000)+"mm")

        Wavelength_Wide = np.linspace(1000,2500,400)

        n_0_quartz_Wide = n_0_quartz_function(Wavelength_Wide)
        n_e_quartz_Wide = n_e_quartz_function(Wavelength_Wide)
        n_0_MgF2_Wide = n_0_MgF2_function(Wavelength_Wide)
        n_e_MgF2_Wide = n_e_MgF2_function(Wavelength_Wide)

        FittedRetardance = (180/np.pi)*CalculateRetardance(Wavelength_Wide,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz_Wide,n_e_quartz_Wide,n_0_MgF2_Wide,n_e_MgF2_Wide,CrossAxes,SwitchMaterials)
        FittedRetardanceDiscrete = (180/np.pi)*CalculateRetardance(WavelengthList,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz,n_e_quartz,n_0_MgF2,n_e_MgF2,CrossAxes,SwitchMaterials)

        fig1 = plt.figure()
        frame1=fig1.add_axes((.13,.36,.77,.57))
        frame1.set_xticklabels([])

        PlotModelParameter(np.delete(R_Hwp_List,BadIndex),np.delete(WavelengthList,BadIndex),"$\Delta_{Hwp}$","($^{\circ}$)","Fit of quartz and $\mathrm{MgF_2}$ plates to measured HWP retardance",False,"R_Hwp",165,185,NewFigure=False)
        plt.scatter(WavelengthList[BadIndex],R_Hwp_List[BadIndex],color="red",s=20,label="outliers")
        plt.plot(Wavelength_Wide,FittedRetardance,label="Fitted retardance")

        plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)
        
        plt.legend(fontsize=8,loc=4)
        
        frame2=fig1.add_axes((.13,.1,.77,.23))

        plt.scatter(np.delete(WavelengthList,BadIndex),np.delete(R_Hwp_List-FittedRetardanceDiscrete,BadIndex),color="black",s=20,zorder=10)
        plt.scatter(WavelengthList[BadIndex],(R_Hwp_List-FittedRetardanceDiscrete)[BadIndex],color="red",s=20,zorder=10)
        plt.axhline(y=0,color="black")
        
        plt.xlabel("λ(nm)")
        plt.ylabel("Residuals($^{\circ}$)")

        plt.ylim(-1,1)
        plt.xlim(xmin=np.amin(WavelengthList)-50,xmax=np.amax(WavelengthList)+50)
        plt.yticks([-2,-1,0,1,2])

        plt.grid(linestyle="--")

        plt.show()

    if(RunDerotatorFit):
        
        R_Der_Degree = 4
        Delta_Hwp_Degree = 6
        R_Der_Coefficients = PlotModelParameterFit(R_Der_List,WavelengthList,"$\Delta_{Der}$","($^{\circ}$)","Polynomial fit(deg="+str(R_Der_Degree)+") on derotator retardance",True,"R_Der",45,260,-10,10,R_Der_Degree,[4])
        #PlotModelParameterFit(Delta_Der_List,WavelengthList,"$\delta_{der}$","($^{\circ}$)","Polynomial fit(deg="+str(PolyfitDegree)+") on derotator offset",True,"Delta_Der",-1,0.1,-0.1,0.1,PolyfitDegree,[])
        Delta_Hwp_Coefficients = PlotModelParameterFit(Delta_Hwp_List,WavelengthList,"$\delta_{HWP}$","($^{\circ}$)","Polynomial fit(deg="+str(Delta_Hwp_Degree)+") on HWP offset",True,"Delta_Hwp",-1,0,-0.2,0.2,Delta_Hwp_Degree,[0])

    if(FindSmoothChiSquared):
        R_Der_List_Smooth = GetPolyfitY(WavelengthList,R_Der_Coefficients)
        Delta_Hwp_List_Smooth = GetPolyfitY(WavelengthList,Delta_Hwp_Coefficients)
        Smooth_ChiSquared_List = []

        for i in range(len(R_Der_List_Smooth)):
            SmoothModel = SCExAO_Model.MatrixModel("",0,FittedRetardanceDiscrete[i],Delta_Hwp_List_Smooth[i],0,R_Der_List_Smooth[i],Delta_Der_List[i],d_List[i],Delta_Cal_List[i],0,0)
            SmoothChiSquared = GetChiSquared(SmoothModel,ReshapedPolArray[i],PolDerList,np.array([1,0,0,0]),True,False)
            
            Smooth_ChiSquared_List.append(SmoothChiSquared)

        print("R_Der: "+str(R_Der_List_Smooth))
        print("Delta_Hwp: "+str(Delta_Hwp_List_Smooth))
        print("R_Hwp: "+str(FittedRetardanceDiscrete))
        PlotModelParameter(ChiSquaredList,WavelengthList,"$\chi^2$","","Sum of squared residuals of fits over wavelength",False,"ChiSquared",DataColor="blue",DataLabel="$\chi^2$ of fit")
        PlotModelParameter(Smooth_ChiSquared_List,WavelengthList,"$\chi^2$","","Sum of squared residuals of smoothened model over wavelength",False,"ChiSquared",NewFigure=False,ymin=0,ymax=2,DataColor="darkorange",DataLabel="$\chi^2$ of smoothened parameters")
        plt.show()



    if(TestPlot):
        #IdealModel = SCExAO_Model.MatrixModel("",-0.0003,170.7,-0.6,-0.003,99.32,0.5,0.9955,-1.5,0,0,45)
        IdealModel = SCExAO_Model.MatrixModel("",0,180,0,0,-10,0,1,0,0,0)
        #IdealParmValueArray = IdealModel.FindParameterArray(PolDerList,np.array([1,0,0,0]),0,UsePolarizer=True)
        #SCExAO_Cal.PlotParamValues(0,IdealModel,True)
        #SCExAO_Cal.PlotPolarimetricEfficiency(4,"black",IdealModel)
        SCExAO_Cal.Plot_AOLP_Offset(8,IdealModel,"black")
        #plt.show()
        #SCExAO_Cal.Plot_AOLP_Offset(0,IdealModel,"black")


        plt.show()



#--/--Main--/--#