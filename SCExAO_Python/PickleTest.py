from SCExAO_CalibrationMain import SCExAO_Calibration
import pickle
import numpy as np


class A:
    def __init__(self,Chickens):
        self.Chickens = Chickens
        #self.Files = open("C:/Users/Gebruiker/Desktop/PickleTestFile.txt", "r")
        self.Array = np.ones((1,2,3))

    def SomeFunc(self):
        Monkeys =  lambda a : a+1
        self.Monkeys = Monkeys(1)


def SomeOtherFunc():
    print("yep")



SCExAO = pickle.load(open("C:/Users/Gebruiker/Desktop/BRP/SCExAO/PickleFiles/PickleSCExAOClass.txt","rb"))
print(SCExAO.PolDDImageArray)
