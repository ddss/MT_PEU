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

from MT_PEU import EstimacaoNaoLinear
from numpy import exp

##################################################################################
##################################################################################
# EXEMPLOS PARA MODELOS NÃO LINEARES
##################################################################################
##################################################################################

def Modelo (param, x, args):
    tipo = args[0][0]

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y1 = [exp(-(ko*10**17)*tempo*exp(-E/T)),exp(-tempo*exp(ko-E/T)),exp(-ko*tempo*exp(-E*(1/T-1./630.)))]

    return y1[tipo]

# =================================================================================
# PARTE II - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

# ---------------------------------------------------------------------------------
# Exemplo validação: Exemplo resolvido 5.11, 5.12-1, 5.12-2 (capítulo 5) (Análise de Dados experimentais I)
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

# =================================================================================
# PARTE III - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================
#

Estime.setConjunto(tipo='estimacao')

#Estime.setConjunto(tipo='predicao')

grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário

# Otimização
Estime.otimiza(estimativa_inicial= [0.006, 22000.000],algoritmo='Nelder-Mead',args=[tipo])
#Estime.SETparametro([0.0075862408745003265, 27642.662773759967],args=[tipo])
Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=False)
Estime.predicao()
Estime.analiseResiduos()
etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
#etapas = ['grandezas-entrada']
Estime.graficos(etapas)
Estime.relatorio(export_y=True,export_cov_y=True)
