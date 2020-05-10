import Methods as Mt
import numpy as np

def Round(Matrix):
    return np.round(Matrix,5)


Theta_Der = 0
Theta_Hwp = 22.5
R_Der = 90

M_Der = Mt.ComMatrix(0,R_Der*np.pi/180)
M_Hwp = Mt.ComMatrix(0,np.pi)

print("Der")
print(Round(Mt.ApplyRotation(M_Der,Theta_Der*np.pi/180)))
print("Hwp")
print(Round(Mt.ApplyRotation(M_Hwp,Theta_Hwp*np.pi/180)))
