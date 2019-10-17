'''This example is about N,N-diethyl-formamide (liq) and was taken from the article 'Vapour pressures and enthalpies
 of vaporisation of alkyl formamides' '''

from MT_PEU import EstimacaoNaoLinear
from numpy import exp, log

def Model(param, x,*args):

    a, b = param[0], param[1]
    T    = x[0]

    return exp(a / 8.31446 + b / (8.31446 * T) - (65.6 / 8.31446) * log( T / 298.15))

Estimation = EstimacaoNaoLinear(Model, simbolos_x=['T'], simbolos_y=['P'], simbolos_param=['a','b'], projeto='Formamide' )

'''Data definiton'''
P = [75.1, 96.1, 117.3, 167.7, 169, 211, 248.2, 326.5, 464.6, 627.4, 709.4,
     853.4, 1093, 1478.8, 1732.8, 1947.5, 2460.1, 2893.3, 3208.9, 4163.7]
T = [288.8, 292.2, 295.1, 300.5, 300.7, 303.7, 305.7, 310.5, 315.7, 320.8, 323.2,
     325.8, 330.9, 335.9, 338.5, 340.9, 346, 348.5, 350.9, 356.1]

uP  = [1.9, 2.4, 3.0, 4.2, 4.3, 5.3, 6.2, 8.2, 11.6, 15.7, 17.8, 21.4, 27.4, 37.0, 43.3, 48.7, 61.5, 72.4, 80.2, 104.1]
uxT = [1]*len(T)

Estimation.setDados(0,(T,uxT))

Estimation.setDados(1,(P,uP))

Estimation.setConjunto(tipo='estimacao')

Estimation.optimize(initial_estimative = [200, -70000])

Estimation.incertezaParametros(metodoIncerteza='SensibilidadeModelo')

Estimation.predicao()

Estimation.analiseResiduos()

etapas = ['predicao', 'grandezas-calculadas', 'analiseResiduos', 'regiaoAbrangencia']

Estimation.graficos(etapas)

