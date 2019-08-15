from MT_PEU import EstimacaoNaoLinear
from numpy import exp

def Modelo (param, x, args):

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y = exp(-(ko*10**17)*tempo*exp(-E/T))


    return y

Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], simbolos_y=[r'y'], simbolos_param=['ko','E'])

y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]

temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

constante = 10**17

uy = [1]*41; uxtempo = [1]*41; uxtemperatura = [1]*41

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy))
Estime.setConjunto(tipo='estimacao')


def modelocasadi(ko,E,tempo,T):
    return exp(-(ko*10**17)*tempo*exp(-E/T))

parameters, var_indep = Estime.casadivariables()
Modelo_cas = modelocasadi(parameters[0],parameters[1],var_indep[0],var_indep[1])

print(Estime.otimiza_cas(Estimativa_inicial = [0.5,25000],Modelo_cas = Modelo_cas))







