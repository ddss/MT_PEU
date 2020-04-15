from MT_PEU import EstimacaoNaoLinear
from numpy import exp

def Model (param,x, *args):

    ko, E = param[0], param[1]
    tempo, T = x[:,0], x[:,1]

    return exp(-tempo*exp(ko-E/T))

Estime = EstimacaoNaoLinear(Model, simbolos_x=['t','T'], unidades_x=['s','K'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko','E'], unidades_param=['adm','K'], Folder='Teste1')

#Tempo
tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]


#Temperatura
temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

uxtempo = [1]*41
uxtemperatura = [1]*41
uy = [1]*41

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy))

Estime.setConjunto()

Estime.optimize(initial_estimative=[18, 20000.000])

Estime.incertezaParametros(metodoIncerteza='2InvHessiana',objectiveFunctionMapping=True)

Estime.predicao()
Estime.analiseResiduos()

etapas = ['grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos()