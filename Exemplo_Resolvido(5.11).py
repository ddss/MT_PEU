# -*- coding: utf-8 -*-
"""
Exemplo de uso do MT_PEU

EXEMPLO (5.11) - Retirado do livro Schwaab e Pinto (2007, p. 323) que trata sobre a estimação de parâmetros do modelo cinético:


"""

from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import exp

# =================================================================================
# PARTE II - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

u"""
O modelo é definido na forma de uma subrotina ((def) do python) e representa a equação acima,onde: 
y1 é a fração que resta do reagente, t é o tempo, T é a temperatura, por fim k0 e E são os parâmetros a serem estimados.

"""

def Modelo (param, x, args):

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y = exp(-(ko*10**17)*tempo*exp(-E/T))


    return y


# =================================================================================
# PARTE II - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

u"""

Iniciar a classe para realizar a estimação:

"""
#Cria o objeto que realiza a estimação
Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], unidades_x=['s','K'], label_latex_x=[r'$t$','$T$'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko','E'], unidades_param=['unid1','unid2'],label_latex_param=[r'$k_o$',r'$E$'],
                            projeto='Test0')

# =================================================================================
# PARTE II - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

u"""

Os dados experimentais da variável dependente (y) e das variáveis independentes (t e T)
são disponibilizados em Schwaab e Pinto (2007, p.324), e apresentados abaixo na forma de listas:

"""
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

u"""

A plataforma MT_PEU necessita da incerteza dos dados experimentais, e quando está informação não é disponibilizada,
o programa assume valor 1 para todos os dados (ux1, ux2, uy1):

"""

uy = [1]*41

uxtempo = [1]*41

uxtemperatura = [1]*41

u"""

Inclusão de dados experimentais na estimação:
Inclui os dados experimentais nesse objeto (setDados), onde a opção 0 é para a grandeza dependente,
e a opção 1 é para a grandeza independente.

"""

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))

Estime.setDados(1,(y,uy))


u"""

Define que os dados experimentais previamente inseridos serão utilizados como um conjunto de dados para o qual os parâmetros serão estimados:

"""

Estime.setConjunto(tipo='estimacao')

# =================================================================================
# PARTE IV - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

u"""

Otimização será realizada utilizando o algoritmo escolhido pelo usuário (disponiveis: Nelder-Mead, Powell, BFGS, L-BFGS-B, CG),
caso não seja escolhido o algoritmo a ser utilizado, por padrão a otimização será realizada utilizando o Nelder-Mead,
com estimativa inicial em [0,03 , 20000,00], para k0 e E respectivamente.

"""
Estime.otimiza(estimativa_inicial= [0.005, 20000.000],algoritmo='Nelder-Mead')

# =================================================================================
# PARTE IV - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=True)

# =================================================================================
# PARTE IV - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

Estime.predicao()
Estime.analiseResiduos()

# =================================================================================
# PARTE IV - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)
Estime.relatorio(export_y=True,export_cov_y=True)

u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

"""