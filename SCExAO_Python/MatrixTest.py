import Methods as Mt
import numpy as np

def Round(Matrix):
    return np.round(Matrix,5)


M_Hwp = Mt.ComMatrix(0,np.pi)
M_Hwp_Q = Mt.ApplyRotation(M_Hwp,0)
M_Hwp_U = Mt.ApplyRotation(M_Hwp,(1/8)*np.pi)

M_Pol_Plus = Mt.Polarizer(True)
M_Pol_Min = Mt.Polarizer(False)

M_SD_Q = np.dot(M_Pol_Plus,M_Hwp_Q) - np.dot(M_Pol_Min,M_Hwp_Q)
M_SD_U = np.dot(M_Pol_Plus,M_Hwp_U) - np.dot(M_Pol_Min,M_Hwp_U)

print(np.round(M_SD_Q,5))
print(np.round(M_SD_U,5))

print(np.invert(np.ones((2,3), dtype=bool)))

#heta_Der = 0
#Theta_Hwp = 22.5
#R_Der = 90

#M_Der = Mt.ComMatrix(0,R_Der*np.pi/180)


#print("Der")
#print(Round(Mt.ApplyRotation(M_Der,Theta_Der*np.pi/180)))
#print("Hwp")
#print(Round(Mt.ApplyRotation(M_Hwp,Theta_Hwp*np.pi/180)))
