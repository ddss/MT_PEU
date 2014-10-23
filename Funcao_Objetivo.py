# -*- coding: utf-8 -*-

from subrotinas import matriz2vetor

from time import ctime, time, sleep
from numpy import random, size, linalg
from scipy import transpose, dot, concatenate, matrix
from threading import Thread
from Modelo import Modelo
import sys

class WLSRecon(Thread):
    result = 0
    def __init__(self,p,argumentos):
        Thread.__init__(self)
        self.param = p
        self.y     = argumentos[0]
        self.x     = argumentos[1]
        self.Vy    = argumentos[2]
        self.Vx    = argumentos[3]
        self.args  = argumentos[4]
 
        
    def run(self):

        NP = self.args[-1]                
        p  = self.param[0:NP]
        xm = self.param[NP:]
        ym = Modelo(p,xm,self.args[:-1])
        ym.start()
        ym.join()
        #print '-------------'
        #print ym
        #print '-------------'

        dy = self.y - ym.result
        dx = self.x - transpose(matrix(xm))
  

        self.result =  float(dot(dot(transpose(dy),linalg.inv(self.Vy)),dy)) + float(dot(dot(transpose(dx),linalg.inv(self.Vx)),dx)) 


class WLS(Thread):
    result = 0
    def __init__(self,p,argumentos):
        Thread.__init__(self)

        self.param  = p
        
        self.y      = argumentos[0]
        self.x      = argumentos[1]
        self.Vy     = argumentos[2]
        self.Vx     = argumentos[3]        
        self.simb_x = argumentos[4]
        self.simb_y = argumentos[5]
        
        self.args   = argumentos[6]
        
    def run(self):

        ym = Modelo(self.param,self.x,self.args)
        ym.start()
        ym.join()
        
        ym = matriz2vetor(ym.result)

        #print '-------------'
        #print ym
        #print '-------------'

        d     = self.y - ym
        self.result =  float(dot(dot(transpose(d),linalg.inv(self.Vy)),d))