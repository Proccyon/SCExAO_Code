import Methods as Mt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as ani

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



R_HWP =  170*np.pi/180
M_HWP_Q1 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),0*np.pi/180)
M_HWP_Q2 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),45*np.pi/180)
M_HWP_U1 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),22.5*np.pi/180)
M_HWP_U2 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),67.5*np.pi/180)

M_HWP_X1 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),11.25*np.pi/180)
M_HWP_X2 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),56.25*np.pi/180)
M_HWP_Y1 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),33.75*np.pi/180)
M_HWP_Y2 = Mt.ApplyRotation(Mt.ComMatrix(0,R_HWP),78.75*np.pi/180)

M_Der1 = Mt.ApplyRotation(Mt.ComMatrix(0,90*np.pi/180),40*np.pi/180)
M_Der2 = Mt.ApplyRotation(Mt.ComMatrix(0,90*np.pi/180),140*np.pi/180)

M_HWP_Q = M_HWP_Q1-M_HWP_Q2
M_HWP_U = M_HWP_U1-M_HWP_U2
M_HWP_X = M_HWP_X1-M_HWP_X2
M_HWP_Y = M_HWP_Y1-M_HWP_Y2

##print(Round(np.dot(M_HWP_Q,[1,1,0,0])))
#print(Round(np.dot(M_HWP_U,[1,1,0,0])))
#print("")
#print(Round(np.dot(M_HWP_Q,[1,0,1,0])))
#print(Round(np.dot(M_HWP_U,[1,0,1,0])))
#print("M_HWP_Q")

#print("M_HWP_U")

#print(Round(np.dot(M_HWP_X,[1,0,1,0])))
#print(Round(np.dot(M_HWP_Y,[1,0,1,0])))
#print("")
#print(Round(np.dot(M_HWP_X,[1,1,0,0])))
#print(Round(np.dot(M_HWP_Y,[1,1,0,0])))

#print("M_Der_1")
#print(Round(M_Der1))
#print("M_Der_2")
#print(Round(M_Der2))
#print(Round(np.dot(M_Der1,M_HWP)))
#print(Round(np.dot(M_Der2,M_HWP)))

Fig1 = plt.figure()
plt.plot([1,2,3],[3,4,5])
Fig2 = plt.figure()
plt.plot([1,2,3],[3,4,6])

Writer = ani.FFMpegWriter(1)
Writer.setup(Fig1,"C:/Users/Gebruiker/Desktop/BRP/TestMovie.mp4", 300)