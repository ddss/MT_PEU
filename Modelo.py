# -*- coding: utf-8 -*-

from threading import Thread
from numpy import concatenate, exp

class Modelo(Thread):
    result = 0
    def __init__(self,param,x,args):
        Thread.__init__(self)
        self.param = param
        self.x     = x

        self.args  = args

    def run(self):
        
        x = self.x

        alpha = self.param[0]
        beta  = self.param[1]
        
        y1 = alpha*(x**beta)
        #y1 = alpha*x/(1+beta*x)


        #tempo = self.x[:,0:1]
        #T     = self.x[:,1:2]
        
        #ko = self.param[0]
        #E  = self.param[1]
        #gama  = self.param[2]
        #eta   = self.param[3]
        #y1 = exp(-(ko*10**17)*tempo*exp(-E/T))
        #y1 = exp(-tempo*exp(ko-E/T)) 
        # y1 = exp(-ko*tempo*exp(-E*(1/T-1./630.)))
        #y1 = alpha*x/(1+beta*x)

        self.result = y1