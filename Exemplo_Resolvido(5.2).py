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
from numpy import concatenate


def Modelo1(param,x,args):
    x1 = x[:,0:1]
    x2 = x[:,1:]

    alpha1 = param[0]
    beta1  = param[1]
    alpha2 = param[2]
    beta2  = param[3]

    y1 = alpha1*x1/(1+beta1*x1)
    y2 = alpha2*(x2**beta2)

    y1 = concatenate((y1,y2),axis=1)

    return y1

# ---------------------------------------------------------------------------------
# Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
# ---------------------------------------------------------------------------------
Estime = EstimacaoNaoLinear(Modelo1,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=[r'a%d'%i for i in xrange(4)],
                          label_latex_param=[r'$\alpha_{%d}$'%i for i in xrange(4)],unidades_y=['kg','kg'],projeto='projeto')
tipo = None

x1 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y1 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]
x2 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y2 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]

ux1 = [1]*10
ux2 = [1]*10
uy1 = [1]*10
uy2 = [1]*10

Estime.setDados(0,(x1,ux1),(x2,ux2))
Estime.setDados(1, (y1, uy1), (y2, uy2))


sup = [6., .3, 8., 0.7]
inf = [1., 0, 1., 0.]


# =================================================================================
# PARTE III - GENÉRICO (INDEPENDE DO EXEMPLO)
# =================================================================================
#

Estime.setConjunto(tipo='estimacao')

#Estime.setConjunto(tipo='predicao')

grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário

# Otimização
Estime.otimiza(estimativa_inicial=[0.005, 20000.000, 0, 0],algoritmo='Powell',args=[tipo])
#Estime.SETparametro([0.0075862408745003265, 27642.662773759967],args=[tipo])
Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=False)
Estime.predicao()
Estime.analiseResiduos()
etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
etapas = ['grandezas-entrada']
Estime.graficos(etapas)
Estime.relatorio(export_y=True,export_cov_y=True)