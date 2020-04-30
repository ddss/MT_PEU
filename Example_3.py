from MT_PEU import EstimacaoNaoLinear
from numpy import exp


def Model(param, x, *args):
    ko, E = param[0], param[1]
    tempo, T = x[:,0], x[:,1]

    return exp(-ko * tempo * exp(-E * (1 / T - 1. / 630.)))


Estime = EstimacaoNaoLinear(Model, simbolos_x=[r't', 'Tau'], unidades_x=['s', 'K'], label_latex_x=[r'$t$', '$T$'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko', 'E'], unidades_param=['adm', 'K'],
                            label_latex_param=[r'$k_o$', r'$E$'],
                            Folder='Exemplo3')

# Tempo
tempo = [120.0, 60.0, 120.0, 60.0, 30.0, 15.0, 45.1, 90.0, 150.0, 60.0, 60.0, 30.0, 150.0, 90.4, 120.0, 60.0, 60.0,
         60.0, 30.0,
         45.1, 30.0, 45.0, 15.0, 90.0, 25.0, 60.1, 60.0, 30.0, 60.0]

# Temperatura
temperatura = [600.0, 612.0, 612.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0, 620.0,
               620.0, 620.0,
               620.0, 631.0, 631.0, 631.0, 631.0, 639.0, 639.0, 639.0, 639.0, 639.0, 639.0, 639.0]

y = [0.9, 0.886, 0.791, 0.787, 0.877, 0.938, 0.827, 0.696, 0.582, 0.795, 0.790, 0.883, 0.576, 0.715, 0.673, 0.802,
     0.804, 0.804, 0.764,
     0.688, 0.802, 0.695, 0.808, 0.309, 0.689, 0.437, 0.425, 0.659, 0.449]

uxtempo = [0.001] * 29
uxtemperatura = [0.05] * 29
uy1 = [1] * 29

Estime.setDados(0, (tempo, uxtempo), (temperatura, uxtemperatura))
Estime.setDados(1, (y, uy1))

Estime.setConjunto(tipo='estimacao')
Estime.optimize(initial_estimative=[0.005, 20000.000])
Estime.incertezaParametros(metodoIncerteza='Geral')

Estime.predicao()
Estime.analiseResiduos()
#Estime.graficos()

# =================================================================================
# IX - OPCIONAL: PREDIÇÃO
# =================================================================================

u"""
Caso o usuário deseje, será possível fazer a mesma analise anterior com os dados de predição. 
O procedimento a seguir é semelhante, diferindo apenas no parametro de setConjunto, que ao invés de ser tipo = 'estimacao'
passa a ser tipo='predicao'. Se faz necessário pelo no mínimo 4 dados para cada variável de predição. 

"""

tempo = [60.0,120.0,60.0,60.0,60.0,60.0,60.0,30.0,30.0,90.0,60.0,30.0]

temperatura = [600.0,612.0,612.0,620.0,620.0,620.0,620.0,639.0,639.0,620.0,620.0,631.0]

y = [0.949,0.785,0.890,0.782,0.800,0.802,0.799,0.655,0.638,0.712,0.794,0.717]

uxtempo = [0.2]*12
uxtemperatura = [0.2]*12
uy1 = [0.2]*12

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy1))
Estime.graficos()

Estime.setConjunto(tipo='predicao')

Estime.predicao()
Estime.analiseResiduos()
Estime.graficos()

