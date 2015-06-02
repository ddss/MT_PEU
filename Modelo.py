# -*- coding: utf-8 -*-

from threading import Thread
from numpy import concatenate, exp
from sys import exc_info


class Modelo(Thread):
    result = 0
    def __init__(self,param,x,args,**kwargs):
        Thread.__init__(self)
        self.param  = param
        self.x      = x
        
        self.args  = args

        # LIDAR COM EXCEPTIONS THREAD
        self.bucket = kwargs.get('bucket')


    def runEquacoes(self):
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

        tipo = self.args[0][0]

        tempo = self.x[:,0:1]
        T     = self.x[:,1:2]

        ko = self.param[0]
        E  = self.param[1]

        y1 = [exp(-(ko*10**17)*tempo*exp(-E/T)),exp(-tempo*exp(ko-E/T)),exp(-ko*tempo*exp(-E*(1/T-1./630.)))]

        self.result = y1[tipo]

    def run(self):
    
        if self.bucket == None:
            self.runEquacoes()
        else:
            try:
                self.runEquacoes()
            except:
                self.bucket.put(exc_info())