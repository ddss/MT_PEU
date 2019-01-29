# -*- coding: utf-8 -*-

from threading import Thread
from numpy import concatenate, exp
from sys import exc_info


def Modelo(param,x,args):

    # x1 = self.x[:,0:1]
    # x2 = self.x[:,1:]
    #
    # alpha1 = self.param[0]
    # beta1  = self.param[1]
    # alpha2 = self.param[2]
    # beta2  = self.param[3]
    #
    # y1 = alpha1*x1/(1+beta1*x1)
    # y2 = alpha2*(x2**beta2)
    #
    # y1 = concatenate((y1,y2),axis=1)
    #
    # self.result = y1

    tipo = args[0][0]

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y1 = [exp(-(ko*10**17)*tempo*exp(-E/T)),exp(-tempo*exp(ko-E/T)),exp(-ko*tempo*exp(-E*(1/T-1./630.)))]

    return y1[tipo]
