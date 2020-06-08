import Methods as Mt
import numpy as np
import matplotlib.pyplot as plt

def Round(Matrix):
    return np.round(Matrix,5)

def CalcI(Theta_Hwp,PositivePol,Q_IP=0,U_IP=0):
    
    M_Hwp = Mt.ApplyRotation(Mt.ComMatrix(0,np.pi),Theta_Hwp)
    M_Pol = Mt.Polarizer(PositivePol)
    M_IP = np.zeros((4,4))
    M_IP[1,0]=Q_IP
    M_IP[2,0]=U_IP
    M_IP[0,0] = 1
    M_IP[1,1] = 1
    M_IP[2,2] = 1
    M_IP[3,3] = 1

    return np.linalg.multi_dot([M_Pol,M_IP,M_Hwp])


Theta_Hwp_Plus = 11.25*np.pi/180
Theta_Hwp_Min = Theta_Hwp_Plus+45*np.pi/180
IP_Q_List = np.linspace(0,0,1)
IP_U_List = np.linspace(0,0,1)
Measured_IP_Plus_List = []
Measured_IP_Min_List = []

for i in range(len(IP_Q_List)):
    IP_Q = IP_Q_List[i]
    IP_U = IP_U_List[i]

    X_Plus = Round(CalcI(Theta_Hwp_Plus,True,IP_Q,IP_U)-CalcI(Theta_Hwp_Plus,False,IP_Q,IP_U))
    X_Min = Round(CalcI(Theta_Hwp_Min,True,IP_Q,IP_U)-CalcI(Theta_Hwp_Min,False,IP_Q,IP_U))
    Measured_IP_Plus_List.append(X_Plus[0,0])
    Measured_IP_Min_List.append(X_Min[0,0])

#print(X_Plus)
#print(X_Min)
#plt.plot(IP_Q_List,Measured_IP_Plus_List)
#plt.plot(IP_Q_List,Measured_IP_Min_List)
#plt.show()

A = np.arange(10)
print(A[:3:])


