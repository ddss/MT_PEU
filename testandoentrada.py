# -*- coding: utf-8 -*-
"""
Exemplos de validação

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""
 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from Modelo import Modelo
from MT_PEU import EstimacaoNaoLinear
from MT_PEU_Linear import EstimacaoLinear

##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS NÃO LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE II - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

# ---------------------------------------------------------------------------------
# Exemplo validação: Exemplo resolvido 5.11, 5.12, 5.13 (capítulo 5) (Análise de Dados experimentais I)
# ---------------------------------------------------------------------------------
tipo = 2 # tipo: modelo a ser escolhido - 0 (exemplo 5.11), 1 (exemplo 5.12) ou 2 (exemplo 5.13)

Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], unidades_x=['s','K'], label_latex_x=[r'$t$','$T$'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko','E'], unidades_param=['unid1','unid2'],label_latex_param=[r'$k_o$',r'$E$'],
                            projeto='teste%d'%tipo)

#Tempo
x1 = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]

ux1 = [1]*41

#Temperatura
x2 = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

ux2 = [1]*41

Estime.setDados(0,(x1,ux1),(x2,ux2))

y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

uy1 = [1]*41

Estime.setDados(1,(y,uy1))

sup=[50,30000]
inf=[0 ,20000]

# ---------------------------------------------------------------------------------
# Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
# ---------------------------------------------------------------------------------
#
# x1 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
# y1 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
# x2 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
# y2 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
#
# ux1 = ones((10,1))
# ux2 = ones((10,1))
# uy1 = ones((10,1))
# uy2 = ones((10,1))
#
# x  = concatenate((x1,x2),axis=1)
# y  = concatenate((y1,y2),axis=1)
# ux = concatenate((ux1,ux2),axis=1)
# uy = concatenate((uy1,uy2),axis=1)
#
#
# Estime = EstimacaoNaoLinear(WLS,Modelo,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=[r'a%d'%i for i in xrange(4)],
#                             label_latex_param=[r'$\alpha_{%d}$'%i for i in xrange(4)],unidades_y=['kg','kg'],projeto='projeto')
# sup = [6.  ,.3  ,8.  ,0.7]
# inf = [1.  , 0  ,1.  ,0.]
#
# tipo = None

# =================================================================================
# PARTE III - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================
#

Estime.setConjunto(tipo='estimacao')

#Estime.setConjunto(tipo='predicao')

grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário

# Otimização
#Estime.otimiza(limite_superior=sup,limite_inferior=inf,algoritmo='PSOFamily',itmax=500,
#                 Num_particulas=30,metodo={'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-linear'},args=[tipo],printit=True)
Estime.SETparametro([0.0075862408745003265, 27642.662773759967],args=[tipo])
Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=True)
Estime.predicao()
Estime.analiseResiduos()
etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
etapas = ['grandezas-entrada']
Estime.graficos(etapas)
Estime.relatorio(export_y=True,export_cov_y=True)

##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS LINEARES
##################################################################################
##################################################################################

# =================================================================================
# PARTE I - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================
#
# # #Sem o cálculo do termo independente
ER = EstimacaoLinear(['y'],['x'],['p1'],projeto='LINEARsemB')
x = [0,1,2,3,4,5]
ux = [1,1,1,1,1,1]

y = [.1,.9,2.2,3.2,3.9,4.8]
uy = [1,1,1,1,1,1]

ER.setDados(0,(x,ux))
ER.setDados(1,(y,uy))

ER.setConjunto()

# Com o cálculo do termo independente
ER = EstimacaoLinear(['y'],['x'],['p1','p2'],projeto='LINEARcomB1')
x = [0,1,2,3,4,5]
ux = [1,1,1,1,1,1]
#
y = [.1,.9,2.2,3.2,3.9,4.8]
uy = [1,1,1,1,1,1]
#
ER.setDados(0,(x,ux))
ER.setDados(1,(y,uy))
#
ER.setConjunto()

# =================================================================================
# PARTE II - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================

ER.otimiza()
#ER.SETparametro([0.9571428571428567, 0.12380952380952503])
ER.incertezaParametros(iteracoes=1000,metodoPreenchimento='MonteCarlo')
ER.predicao(delta=1e-6)
ER.analiseResiduos()
ER.graficos(['analiseResiduos','regiaoAbrangencia', 'grandezas-entrada', 'predicao','grandezas-calculadas','otimizacao'])
ER.relatorio()
