'''
#-----Header-----#

The goal of this file is to fit a model curve
to the data found in SCExAO_CalibrationMain.py.
This way the model parameters are obtained.
Currently only polarized calibration data is fitted.

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


#-----FittingClass-----#

class SCExAO_Fitting:


    Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_results/PolarizedCalibration3/"

    def __init__(self,CalibrationObject_Pol):
        self.CalibrationObject_Pol = CalibrationObject_Pol
        
        self.WavelengthList = CalibrationObject_Pol.LambdaList[0]
        
        PolParamValueArray = CalibrationObject_Pol.ParamValueArray
        PolParamValueArray = np.swapaxes(PolParamValueArray,0,3)
        PolParamValueArray = np.swapaxes(PolParamValueArray,1,2)
        PolParamValueArray = np.swapaxes(PolParamValueArray,2,3)
        self.PolParamValueArray = PolParamValueArray

        self.PolDerList = CalibrationObject_Pol.ImrArray[0]

        
    def DoCompleteFit(self,SaveNumber):
 
        '''
        Summary:
            Determines the free parameters of the Mueller matrix model for all wavelengths.
            Used polarized source measurements.     
            Fits Hwp Retardance, Hwp offset angle, derotator retardance, derotator offset angle,
            calibration polarizer diattenuation and calibration polarizer offset angle.
        Input:
            SaveNumber: Determines where the data is saved/retreived. 
                See the SCExAO_Results file --> Calibration1 means SaveNumber = 1
        Output:
            Saves the fitted model parameters in folders SCExAO_Results/Calibration{SaveNumber}/Fit{Wavelength}/
        '''

        OptimizeMethod = "Powell"
        GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0],dtype=float)
        Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(30,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])
        DoFitList = np.array([False,True,True,False,True,True,True,True,False,False])

        DoAllWavelengthFit(SaveNumber,self.Prefix,CreateFitParametersPath,self.WavelengthList,GuessParameterList,self.PolDerList,self.PolParamValueArray,True,False,Bounds,DoFitList,True,OptimizeMethod)

    def DoFinalFit(self,SaveNumber,CompleteFitSaveNumber):

        '''
        Summary:
            Determines the free parameters of the Mueller matrix model for all wavelengths.
            Uses polarized source measurements.     
            Fits Hwp Retardance, Hwp offset angle, derotator retardance and calibration polarizer diattenuation.
            Derotator offset and polarizer offset is set to average values found in CompleteFit.

        Input:
            SaveNumber: Determines where the data is saved/retreived. 
                See the SCExAO_Results file --> Calibration1 means SaveNumber = 1
            CompleteFitSaveNumber: SaveNumber under which the complete fit results are stored

        Output:
            Saves the fitted model parameters in folders SCExAO_Results/Calibration{SaveNumber}/Fit{Wavelength}/
        '''

        OptimizeMethod = "Powell"
        GuessParameterList = np.array([0,180,0,0,90,0,1,0,0,0],dtype=float)
        Bounds = np.array([(-0.1,0.1),(150,210),(-10,10),(-0.1,0.1),(30,180),(-10,10),(0.8,1),(-10,10),(-0.1,0.1),(-0.1,0.1)])
        DoFitList = np.array([False,True,True,False,True,False,True,False,False,False])

        CompleteFitParameters = GetModelParameters(self.WavelengthList,CompleteFitSaveNumber,self.Prefix)
        CompleteFit_Delta_Der_List = CompleteFitParameters[3]
        CompleteFit_Delta_Cal_List = CompleteFitParameters[5]

        GuessParameterList[5] = np.average(CompleteFit_Delta_Der_List)
        GuessParameterList[7] = np.average(CompleteFit_Delta_Cal_List[-9:])

        DoAllWavelengthFit(SaveNumber,self.Prefix,CreateFitParametersPath,self.WavelengthList,GuessParameterList,self.PolDerList,self.PolParamValueArray,True,False,Bounds,DoFitList,True,OptimizeMethod)


    def PlotModelParameter(self,ParameterList,SavePlot,SaveNumber=1,SaveParameterName="",BadIndex=[],ParameterName="",Unit="",Title="",ymin=None,ymax=None,NewFigure=True,DataColor="black",MarkerStyle="o",ScatterSize=20,DataLabel=""):
        
        '''
        Summary:
            Plots a fitted model parameters over wavelength.
        Input:
            ParameterList: List of model parameter values over wavelength.
                Has dimensions: (22) --> (Wavelength)
            SavePlot: whether or not the plot is automatically saved.
            SaveNumber

            SaveNumber: Determines where the data is saved/retreived. 
                See the SCExAO_Results file --> Calibration1 means SaveNumber = 1
            
            SaveParameterName: Used in saving the .png file of the plot.
            BadIndex: At the wavelength indices the data points are shown in red to indicate outliers.
            ParameterName: Name of the parameter, used in labels etc.
            Unit: Unit of the model parameter.
            Title: Title of the plot.
            ymin: Lower limit of plot.
            ymax: upper limit of plot.
            NewFigure: Whether or not plt.figure is used.
            DataColor: Color of data points.
            MarkerStyle: Style of data points.
            ScatterSize: Size of data points.
            DataLabel: Label of data points, no label is used if set to ""  

        '''

        if(NewFigure):
            plt.figure()

        if(Title==""):
            plt.title("Fitted "+ParameterName+" over wavelength")
        else:
            plt.title(Title)

        plt.xlabel("λ(nm)")
        plt.ylabel(ParameterName+Unit)

        plt.xlim(xmin=np.amin(self.WavelengthList)-50,xmax=np.amax(self.WavelengthList)+50)
        if(not ymin==None and not ymax==None):
            plt.ylim(ymin,ymax)

        plt.scatter(np.delete(self.WavelengthList,BadIndex),np.delete(ParameterList,BadIndex),color=DataColor,s=ScatterSize,zorder=100,label=DataLabel,alpha=1,marker=MarkerStyle,linewidths=1)
        if(len(BadIndex) > 0):
            plt.scatter(self.WavelengthList[BadIndex],ParameterList[BadIndex],color="black",s=ScatterSize,zorder=100,label="Outliers",alpha=1,marker=MarkerStyle,linewidths=1)

        plt.grid(linestyle="--")
        if(DataLabel != ""):
            legend = plt.legend(fontsize=7)
            legend.set_zorder(200)

        if(SavePlot):
            plt.savefig(CreateModelPlotPath(self.Prefix,SaveNumber,SaveParameterName))

    def PlotAllModelParameters(self,SaveNumber,SavePlots=True):

        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        
        self.PlotModelParameter(FitParameters[0],SavePlots,SaveNumber,"R_Hwp",ParameterName="$\Delta_{Hwp}$",Unit="($^{\circ}$)",Title="Fitted Half-wave plate retardance over wavelength",ymin=165,ymax=185)
        self.PlotModelParameter(FitParameters[1],SavePlots,SaveNumber,"Delta_Hwp",ParameterName="$\delta_{Hwp}$",Unit="($^{\circ}$)",Title="Fitted Half-wave plate offset over wavelength")
        self.PlotModelParameter(FitParameters[2],SavePlots,SaveNumber,"R_Der",ParameterName="$\Delta_{der}$",Unit="($^{\circ}$)",Title="Fitted derotator retardance over wavelength",ymin=45,ymax=260)

        self.PlotModelParameter(FitParameters[3],SavePlots,SaveNumber,"Delta_Der",ParameterName="$\delta_{der}$",Unit="($^{\circ}$)",Title="Fitted derotator offset over wavelength")
        Delta_Der_Average = np.average(FitParameters[3])
        plt.axhline(y=Delta_Der_Average,linestyle="--",color="blue",label="Average $\delta_{der}$")
        plt.legend(fontsize=8)

        self.PlotModelParameter(FitParameters[4],SavePlots,SaveNumber,"d",ParameterName="$\epsilon_{cal}$",Unit="",Title="Fitted polarizer diattenuation over wavelength")
        
        self.PlotModelParameter(FitParameters[5],SavePlots,SaveNumber,"Delta_Cal",ParameterName="$\delta_{Cal}$",Unit="($^{\circ}$)",Title="Fitted calibration polarizer offset over wavelength")
        Delta_Cal_Average = np.average(FitParameters[5][-9:])
        plt.axhline(y=Delta_Cal_Average,linestyle="--",color="blue",label="Average $\delta_{Cal}$")
        plt.legend(fontsize=8)

        self.PlotModelParameter(FitParameters[6],SavePlots,SaveNumber,"ChiSquared",ParameterName="$\chi^2$",Unit="",Title="Sum of squared residuals of fits over wavelength")

    def PlotStokesParameters(self,SaveNumber):

        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        FittedModelList = FitParameters[7]

        for i in range(22):
            self.CalibrationObject_Pol.PlotParamValues(i,FittedModelList[i],True)
            plt.savefig(CreateFitPlotPath(self.Prefix,SaveNumber,int(self.WavelengthList[i])))

        plt.show()

    def PlotEffDiagram(self,SaveNumber,WaveNumbers,Colors):
        
        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        FittedModelList = FitParameters[7]

        ColorIndex = 0

        for i in range(22):
            if(i in WaveNumbers):
                plt.figure(1)
                self.CalibrationObject_Pol.PlotPolarimetricEfficiency(i,Colors[ColorIndex],FittedModelList[i])
                ColorIndex+=1


    def Plot_Min_Eff_Diagram(self,SaveNumber,SaveFile):

        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        FittedModelList = FitParameters[7]

        SCExAO_Cal.PlotMinimumEfficiency(FittedModelList)
        if(SaveFile):
            plt.savefig(CreateMinEffPath(self.Prefix,SaveNumber))

    def PlotAoLP_Diagram(self,SaveNumber,WaveNumbers,Colors):
        
        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        FittedModelList = FitParameters[7]

        plt.figure()
        for i in range(len(WaveNumbers)):
            self.CalibrationObject_Pol.Plot_AOLP_Offset(WaveNumbers[i],FittedModelList[WaveNumbers[i]],Colors[i])

        plt.legend(fontsize=8)
        
        plt.show()

    def Plot_Max_AOLP_Offset(self,SaveNumber):

        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        FittedModelList = FitParameters[7]

        self.CalibrationObject_Pol.Plot_Max_AOLP_Offset(FittedModelList)

    def Do_HWP_Retardance_Fit(self,SaveNumber,BadIndex,OptimizeMethod="Powell"):

        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        R_Hwp_Measured = FitParameters[0]

        n_0_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.07044083,1.00585997E-2,1.10202242,100,0.28604141)
        n_e_quartz_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0,0,1.09509924,1.02101864E-2,1.15662475,100,0.28851804)
        n_0_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.48755108,0.04338408**2,0.39875031,0.09461442**2,2.3120353,23.793604**2,0)
        n_e_MgF2_function = lambda Wavelength : RefractiveIndexFormula(Wavelength,0.41344023,0.03684262**2,0.50497499,0.09076162**2,2.4904862,23.771995**2,0)

        n_0_quartz = n_0_quartz_function(self.WavelengthList)
        n_e_quartz = n_e_quartz_function(self.WavelengthList)
        n_0_MgF2 = n_0_MgF2_function(self.WavelengthList)
        n_e_MgF2 = n_e_MgF2_function(self.WavelengthList)

        ThicknessBounds = [[1E-6,2E-1],[1E-6,2E-1]]
        ThicknessGuessList = [1.6E-3,1.25E-3]
        CrossAxes = True
        SwitchMaterials = True

        RetardanceArgs = (np.delete(self.WavelengthList,BadIndex),np.delete(R_Hwp_Measured,BadIndex)*np.pi/180,np.delete(n_0_quartz,BadIndex),np.delete(n_e_quartz,BadIndex),np.delete(n_0_MgF2,BadIndex),np.delete(n_e_MgF2,BadIndex),CrossAxes,SwitchMaterials,ThicknessBounds)

        Results = scipy.optimize.minimize(RetardanceMinimizeFunction,ThicknessGuessList,args=RetardanceArgs,method=OptimizeMethod)
        Fitted_d_quartz,Fitted_d_MgF2 = Results["x"]

        SaveFile = open(CreateHwpThicknessesPath(self.Prefix,SaveNumber),"w+")
        SaveThicknesses(SaveFile,Fitted_d_quartz,Fitted_d_MgF2)

        print("d_quartz="+str(Fitted_d_quartz*1000)+"mm")
        print("d_MgF2="+str(Fitted_d_MgF2*1000)+"mm")

        WavelengthList_Wide = np.linspace(1000,2500,400)

        n_0_quartz_Wide = n_0_quartz_function(WavelengthList_Wide)
        n_e_quartz_Wide = n_e_quartz_function(WavelengthList_Wide)
        n_0_MgF2_Wide = n_0_MgF2_function(WavelengthList_Wide)
        n_e_MgF2_Wide = n_e_MgF2_function(WavelengthList_Wide)

        R_Hwp_Fit_Continuous = (180/np.pi)*CalculateHwpRetardance(WavelengthList_Wide,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz_Wide,n_e_quartz_Wide,n_0_MgF2_Wide,n_e_MgF2_Wide,CrossAxes,SwitchMaterials)
        R_Hwp_Fit_Discrete = (180/np.pi)*CalculateHwpRetardance(self.WavelengthList,Fitted_d_quartz,Fitted_d_MgF2,n_0_quartz,n_e_quartz,n_0_MgF2,n_e_MgF2,CrossAxes,SwitchMaterials)

        fig1 = plt.figure()
        frame1=fig1.add_axes((.13,.36,.77,.57))
        frame1.set_xticklabels([])

        self.PlotModelParameter(R_Hwp_Measured,False,ParameterName="$\Delta_{Hwp}$",Unit="($^{\circ}$)",Title="Fit of quartz and $\mathrm{MgF_2}$ plates to measured HWP retardance",NewFigure=False,ymin=165,ymax=185)
        plt.plot(WavelengthList_Wide,R_Hwp_Fit_Continuous,label="Fitted retardance")

        plt.xlim(xmin=np.amin(self.WavelengthList)-50,xmax=np.amax(self.WavelengthList)+50)
        
        plt.legend(fontsize=8,loc=4)
        
        frame2=fig1.add_axes((.13,.1,.77,.23))

        R_Hwp_Residuals = R_Hwp_Measured-R_Hwp_Fit_Discrete

        plt.scatter(np.delete(self.WavelengthList,BadIndex),np.delete(R_Hwp_Residuals,BadIndex),color="black",s=20,zorder=10)
        plt.scatter(self.WavelengthList[BadIndex],(R_Hwp_Residuals)[BadIndex],color="red",s=20,zorder=10)
        plt.axhline(y=0,color="black")
        
        plt.xlabel(r"$\lambda$(nm)")
        plt.ylabel("Residuals($^{\circ}$)")

        plt.ylim(-1,1)
        plt.xlim(xmin=np.amin(self.WavelengthList)-50,xmax=np.amax(self.WavelengthList)+50)
        plt.yticks([-2,-1,0,1,2])

        plt.grid(linestyle="--")

        SavePath = CreateHwpRetardanceFitPlotPath(self.Prefix,SaveNumber)
        plt.savefig(SavePath)


    def Do_Parameter_Polyfit(self,SaveNumber,ParameterList,FitDegree,ParameterName,Unit=r"($^\circ$)",BadIndex=[],yResMin=-10,yResMax=10):

        NewWavelengthList = np.delete(self.WavelengthList,BadIndex)
        NewParameterList = np.delete(ParameterList,BadIndex)

        PolyfitCoefficients = np.polyfit(NewWavelengthList,NewParameterList,FitDegree)

        SaveFile = open(CreateSmoothenedParameterConstantsPath(self.Prefix,SaveNumber,ParameterName),"w+")
        SaveSmootheningConstants(SaveFile,PolyfitCoefficients)

        fig1 = plt.figure()
        frame1=fig1.add_axes((.13,.36,.77,.57))
        frame1.set_xticklabels([])

        Wavelength_Wide = np.linspace(1000,2500,400)
        Parameter_Fit = GetPolyfitY(Wavelength_Wide,PolyfitCoefficients)

        Title = "Polynomial fit(deg={}) on {}".format(FitDegree,ParameterName).replace("_"," ")

        self.PlotModelParameter(NewParameterList,False,BadIndex=BadIndex,ParameterName=ParameterName,Unit=r"($^\circ$)",Title=Title)
   
        plt.plot(Wavelength_Wide,Parameter_Fit,label="Polynomial fit")
        plt.legend(fontsize=8)

        plt.xlim(xmin=np.amin(self.WavelengthList)-50,xmax=np.amax(self.WavelengthList)+50)

        frame2=fig1.add_axes((.13,.1,.77,.23))

        plt.axhline(y=0,color="black")
        
        ParameterResiduals = ParameterList - GetPolyfitY(self.WavelengthList,PolyfitCoefficients)

        plt.scatter(np.delete(self.WavelengthList,BadIndex),np.delete(ParameterResiduals,BadIndex),color="black",s=20,zorder=20)
        plt.scatter(self.WavelengthList[BadIndex],ParameterResiduals[BadIndex],color="red",s=20,zorder=20)

        plt.grid(linestyle="--")
        plt.yticks(np.linspace(yResMin,yResMax,5))
        plt.ylim(yResMin-0.1*(yResMax-yResMin),yResMax+0.1*(yResMax-yResMin))
        plt.xlabel(r"$\lambda$(nm)")
        plt.ylabel("Residuals"+Unit)
        plt.xlim(xmin=np.amin(self.WavelengthList)-50,xmax=np.amax(self.WavelengthList)+50)

        #plt.savefig(CreateModelFitPlotPath(self.Prefix,SaveNumber,SaveParameterName))

    def Do_Derotator_Retardance_Polyfit(self,SaveNumber):
        FitParameters = GetModelParameters(self.WavelengthList,SaveNumber,self.Prefix)
        R_Der_List = FitParameters[2]

        self.Do_Parameter_Polyfit(SaveNumber,R_Der_List,4,"Derotator_Retardance")


#--/--FittingClass--/--#


#-----Functions-----#


#---FittingFunctions---#

def DoAllWavelengthFit(SaveNumber,Prefix,PathFunction,WavelengthList,GuessParameterList,DerList,ParamValueArray,IsPolarized,DerMethod,Bounds,DoFitList,ForceDerotatorRetardance=True,OptimizeMethod="Powell"):
        
    Old_R_Der_Bounds = Bounds[4].copy()
    Old_R_Der_Guess = GuessParameterList[4]

    for i in range(0,22):

        if(ForceDerotatorRetardance):
            
            if(i<=3):
                Bounds[4] = (180,300)
                GuessParameterList[4] = 190
                
            else:
                Bounds[4] = Old_R_Der_Bounds
                GuessParameterList[4] = Old_R_Der_Guess

        ModelParameterList,ChiSquared = DoSingleWavelengthFit(GuessParameterList,DerList,ParamValueArray[i],IsPolarized,DerMethod,Bounds,DoFitList,OptimizeMethod)

        Wavelength = int(WavelengthList[i])
        SaveFile = open(PathFunction(Prefix,SaveNumber,Wavelength),"w+")

        SaveParameters(SaveFile,ModelParameterList,ChiSquared)
        SaveFile.close()

        print(str(i)+" is Done")

def DoSingleWavelengthFit(GuessParameterList,DerList,ParamValueArray,IsPolarized,DerMethod,Bounds,DoFitList,OptimizeMethod="Powell"):
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

    Args = (GuessParameterList,DerList,ParamValueArray,IsPolarized,DerMethod,Bounds,DoFitList)

    Results = scipy.optimize.minimize(MinimizeFunction,GuessParameterList[DoFitList],args=Args,method=OptimizeMethod)

    FittedParameterList = Results["x"]

    TotalParameterList = FindTotalParameterList(FittedParameterList,GuessParameterList,DoFitList)

    ChiSquared = MinimizeFunction(FittedParameterList,*Args)
    return TotalParameterList, ChiSquared

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

    ChiSquared = GetChiSquared(FittedModel,ParamValueArray,DerList,S_In,UsePolarizer,DerMethod)

    return ChiSquared


#---FittingFunctions---#



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


#---SavingFunctions---#

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

def SaveThicknesses(File,d_quartz,d_MgF2):
    File.write("d_quartz="+str(d_quartz*1000)+"mm\n")
    File.write("d_MgF2="+str(d_MgF2*1000)+"mm\n")

def SaveSmootheningConstants(File,ConstantList):
    
    for i in range(len(ConstantList)):
        File.write("Constant"+str(i+1)+"="+str(ConstantList[i])+"\n")

def ProcessLine(Line):
    Line = Line.replace("\n","")
    Line = Line.split("=")[1]
    return Line

#-/-SavingFunctions-/-#

#---PathFunctions---#

def CreateFitPlotPath(Prefix,SaveNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitPlot{}.png".format(Prefix,SaveNumber,Wavelength,Wavelength,SaveNumber)

def CreateSingleEffPlotPath(Prefix,SaveNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}SingleEffPlot{}.png".format(Prefix,SaveNumber,Wavelength,Wavelength,SaveNumber)

def CreateSingleAolpPlotPath(Prefix,SaveNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}SingleAolpPlot{}.png".format(Prefix,SaveNumber,Wavelength,Wavelength,SaveNumber)

def CreateEffPlotPath(Prefix,SaveNumber):
    return "{}Calibration{}/EffPlot{}.png".format(Prefix,SaveNumber,SaveNumber)

def CreateMinEffPath(Prefix,SaveNumber):
    return "{}Calibration{}/MinEffPlot{}.png".format(Prefix,SaveNumber,SaveNumber)

def CreateFitParametersPath(Prefix,SaveNumber,Wavelength):
    return "{}Calibration{}/ModelFits/Fit{}/{}FitParameters{}.txt".format(Prefix,SaveNumber,Wavelength,Wavelength,SaveNumber)

def CreateHwpThicknessesPath(Prefix,SaveNumber):
    return "{}Calibration{}/SmoothenedParameters/HwpRetardance/HwpThicknesses{}.txt".format(Prefix,SaveNumber,SaveNumber)

def CreateHwpRetardanceFitPlotPath(Prefix,SaveNumber):
    return "{}Calibration{}/SmoothenedParameters/HwpRetardance/HwpRetardanceFitPlot{}.png".format(Prefix,SaveNumber,SaveNumber)

def CreateSmoothenedParameterConstantsPath(Prefix,SaveNumber,ParameterName):
    return "{}Calibration{}/SmoothenedParameters/{}/{}Constants{}.txt".format(Prefix,SaveNumber,ParameterName,ParameterName,SaveNumber)

def CreateModelPlotPath(Prefix,SaveNumber,ModelParameter):
    return "{}Calibration{}/ModelParameterPlots/{}Plot{}.png".format(Prefix,SaveNumber,ModelParameter,SaveNumber)

def CreateModelFitPlotPath(Prefix,SaveNumber,ModelParameter):
    return "{}Calibration{}/ModelParameterPlots/{}Plot{}Polyfit.png".format(Prefix,SaveNumber,ModelParameter,SaveNumber)

#-/-PathFunctions-/-#

#-/-StokesFitFunctions-/-#

#---RetardanceFitFunctions---#

#Formula to find refractive index from refractiveindex.com
def RefractiveIndexFormula(Wavelength,A,B,C,D,E,F,G):
    NewWavelength = Wavelength / 1000 #nanometer to micrometer
    Part1 = A*NewWavelength**2 / (NewWavelength**2-B)
    Part2 = C*NewWavelength**2 / (NewWavelength**2-D)
    Part3 = E*NewWavelength**2 / (NewWavelength**2-F)
    return np.sqrt(Part1+Part2+Part3+G+1)

def CalculateHwpRetardance(Wavelength,d1,d2,n1_0,n1_e,n2_0,n2_e,CrossAxes=True,SwitchMaterials=False):
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

    FittedRetardance = CalculateHwpRetardance(Wavelength,d1,d2,n1_0,n1_e,n2_0,n2_e,CrossAxes,SwitchMaterials)
    return np.sqrt(np.sum((FittedRetardance - RealRetardance)**2))

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


#--/--Functions--/--#

#-----UserInput-----#

if __name__ == '__main__':
    

    SCExAO_CalibrationObject_Pol = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))
    SCExAO_Fitting_Object = SCExAO_Fitting(SCExAO_CalibrationObject_Pol)

    #SCExAO_Fitting_Object.DoCompleteFit(1)

    SCExAO_Fitting_Object.PlotAllModelParameters(1)

    #SCExAO_Fitting_Object.PlotStokesParameters(1)

    #SCExAO_Fitting_Object.PlotAoLP_Diagram(1,[1,2,3,4],["red","blue","cyan","green"])
    
    #SCExAO_Fitting_Object.Plot_Max_AOLP_Offset(1)
    
    #SCExAO_Fitting_Object.Do_HWP_Retardance_Fit(1,[4,5],"Powell")

    #SCExAO_Fitting_Object.Do_Derotator_Retardance_Polyfit(1)

    plt.show()


#--/--UserInput--/--#