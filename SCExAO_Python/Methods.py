'''
#-----Header-----#

#This file contains methods for creating and applying
#Mueller matrices that represent optical components.

#-----Header-----#
'''

#-----Imports-----#
import numpy as np
#--/--Imports--/--#

#-----Matrices-----#

def Polarizer(Direction):

    '''
    Summary:     
        Returns Mueller matrix of an ideal polarizer with transmitting axis in +-Q direction

    Input:
        Direction: Boolean indicating polarization direction.
            Direction == True, --> axis in +Q direction
            Direction == False, --> axis in -Q direction
            
    Output:
        PolarizerMatrix: Mueller matrix of polarizer.
    '''

    #DirectionConstant is either -1 or 1
    DirectionConstant = Direction*2-1
    PolarizerMatrix = np.zeros((4,4))
    PolarizerMatrix[0,0] = 1
    PolarizerMatrix[1,1] = 1
    PolarizerMatrix[0,1] = 1*DirectionConstant
    PolarizerMatrix[1,0] = 1*DirectionConstant

    return 0.5*PolarizerMatrix
   
def Retarder(Delta):
    '''
    Summary:     
        Returns Mueller matrix of an ideal retarder with fast axis in +Q direction.

    Input:
        Delta: Retardance of retarder in radians.

    Output:
        RetarderMatrix: Mueller matrix of retarder.
    '''
    RetarderMatrix = np.zeros((4,4))
    RetarderMatrix[0,0] = 1
    RetarderMatrix[1,1] = 1
    RetarderMatrix[2,2] = np.cos(Delta)
    RetarderMatrix[3,2] = -np.sin(Delta)
    RetarderMatrix[2,3] = np.sin(Delta)
    RetarderMatrix[3,3] = np.cos(Delta)

    return RetarderMatrix
      
def RotationMatrix(Alpha):
    '''
    Summary:     
        Returns a rotation matrix. Use this to simulate a rotated component. 

    Input:
        Alpha: Angle of rotation in radians.

    Output:
        RotationMatrix: Rotation Mueller matrix.
    '''
    RotationMatrix = np.zeros((4,4)) 
    RotationMatrix[0,0] = 1
    RotationMatrix[3,3] = 1
    RotationMatrix[1,1] = np.cos(2*Alpha)
    RotationMatrix[2,1] = np.sin(-2*Alpha)
    RotationMatrix[1,2] = np.sin(2*Alpha)
    RotationMatrix[2,2] = np.cos(2*Alpha)

    return RotationMatrix
    
def IdentityMatrix():
    '''
    Summary:     
        Returns the 4x4 identity matrix (1's on diagonal, 0 otherwise).

    Output:
        IdentityMatrix: Rotation Mueller matrix.
    '''
    IdentityMatrix = np.zeros((4,4))
    IdentityMatrix[0,0] = 1
    IdentityMatrix[1,1] = 1
    IdentityMatrix[2,2] = 1
    IdentityMatrix[3,3] = 1
    return IdentityMatrix

#Matrix used in Holstein et al
#e = diattenuation
#R = retardance    
def ComMatrix(e,R):
    '''
    Summary:     
        Returns Mueller matrix of a component with a given diattenuation and retardance.

    Input:
        e: Diattenuation of component. Ranges from -1 to 1.
        R: Retardance of component in radians. 

    Output:
        ComMatrix: Mueller matrix of component.
    '''
    ComMatrix = np.zeros((4,4))
    ComMatrix[0,0] = 1
    ComMatrix[1,1] = 1
    ComMatrix[0,1] = e
    ComMatrix[1,0] = e
    ComMatrix[2,2] = np.sqrt(1-e**2)*np.cos(R)     
    ComMatrix[3,3] = np.sqrt(1-e**2)*np.cos(R)
    ComMatrix[3,2] = -np.sqrt(1-e**2)*np.sin(R)
    ComMatrix[2,3] = np.sqrt(1-e**2)*np.sin(R)
    return ComMatrix      
    
#--/--Matrices--/--#

#-----Definitions-----#

def PolDegree(S):
    '''
    Summary:     
        Returns the degree of polarization of a Stokes vector.

    Input:
        S: Stokes vector, preferably a numpy array.

    Output:
        PolarizationDegree: Degree of polarization of stokes vector.
    '''
    return np.sqrt(S[1]**2+S[2]**2+S[3]**2)/S[0]

def LinPolDegree(S):
    '''
    Summary:     
        Returns the degree of linear polarization (DoLP) of a Stokes vector.

    Input:
        S: Stokes vector, preferably a numpy array.

    Output:
        DoLP: Degree of linear polarization (DoLP) of stokes vector.
    '''
    return np.sqrt(S[1]**2+S[2]**2)/S[0]
    
def PolAngle(S):
    '''
    Summary:     
        Returns the angle of linear polarization (AoLP) of a Stokes vector.

    Input:
        S: Stokes vector, preferably a numpy array.

    Output:
        AoLP: Angle of linear polarization (AoLP) of stokes vector.
    '''
    return 0.5*np.arctan(S[2]/S[1])
    

#--/--Definitions--/--#

#-----OtherMethods-----#


def ApplyRotation(MuellerMatrix,Alpha):
    '''
    Summary:     
        Rotates a given Mueller matrix model using rotation matrices.

    Input:
        MuellerMatrix: Mueller matrix of an optical component.
        Alpha: Rotation angle of the component in radians.

    Output:
        RotatedMatrix: Matrix in the new reference frame.
    '''
    MatrixMin = RotationMatrix(-Alpha)
    MatrixPlus = RotationMatrix(Alpha)
    return np.dot(np.dot(MatrixMin,MuellerMatrix),MatrixPlus)            
       
                                                                                                                                                                                                                                  
#--/--OtherMethods--/--#    

#-----FresnelEquations-----#

#Fresnel equations are not used.

def FresnelRs(Na,Nb,Theta):
    Numerator = Na*np.cos(Theta)-np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    Denominator = Na*np.cos(Theta)+np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    return Numerator/Denominator

def FresnelRp(Na,Nb,Theta):
    Numerator = -np.cos(Theta)*Nb**2+Na*np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    Denominator = np.cos(Theta)*Nb**2+Na*np.sqrt(Nb**2-(Na**2)*np.sin(Theta)**2)
    return Numerator/Denominator
    
def FresnelTs(Na,Nb,Theta):
    return 1-FresnelRs(Na,Nb,Theta)
    
def FresnelTp(Na,Nb,Theta):
    return 1-FresnelRp(Na,Nb,Theta)

#--/--FresnelEquations--/--#

