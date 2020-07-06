from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

#Data
x = np.arange(1,10,0.2)
ynoise = x*np.random.rand(len(x)) 
#Noise; noise is scaled by x, in order to it be noticable on a x-squared function
ydata = x**2 + ynoise #Noisy data

#Model
Fofx = lambda x,a,b,c: a*x**2+b*x+c
#Best fit parameters
p, cov = curve_fit(Fofx,x,ydata)

#PLOT
fig1 = plt.figure(1)
#Plot Data-model
frame1=fig1.add_axes((.1,.33,.8,.6))
#xstart, ystart, xend, yend [units are fraction of the image frame, from bottom left corner]

plt.title("Residual showcase")
plt.ylabel("Chickens")
plt.plot(x,ydata,'.b') #Noisy data
plt.plot(x,Fofx(x,*p),'-r') #Best fit model
frame1.set_xticklabels([]) #Remove x-tic labels for the first frame
plt.grid()

#Residual plot
difference = Fofx(x,*p) - ydata
frame2=fig1.add_axes((.1,.1,.8,.2))
plt.xlabel("Eggs")
plt.ylabel("Residuals")        
plt.scatter(x,difference,s=6)
plt.grid()

plt.show()