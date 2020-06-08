'''
#-----Header-----#

This file is the SCExAO equivalent of irdap.

#-----Header-----#
'''

#-----Imports-----#

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits
from scipy import ndimage, misc
from datetime import timedelta
import pickle
import Methods as Mt
import SCExAO_Model

#--/--Imports--/--#

def CreatePath(Prefix,ImageNumber):
    return Prefix+str(ImageNumber)+"_cube.fits"

Prefix = "C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/PolarizedStars/polarization_standard_stars/2020-01-31/HD_283809/CRSA0005"
#C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/Calibration/cal_data_instrumental_pol_model/cal_data_unpol_source/CRSA00059563_cube.fits
#C:/Users/Gebruiker/Desktop/BRP/SCExAO/SCExAO_Data/PolarizedStars/polarization_standard_stars/2019-12-16/BD_59389/CRSA00057848_cube.fits

for i in range(8982,9175):
    File = fits.open(CreatePath(Prefix,i))
    Header = File[1].header

    print(repr(Header))
