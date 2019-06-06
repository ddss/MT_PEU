# -*- coding: utf-8 -*-
"""
Exemplo de uso do MT_PEU

EXEMPLO (5.12-2) - Retirado do livro Schwaab e Pinto (2007, p. 366) que trata sobre a estimação de parâmetros do modelo cinético:

SUMÁRIO:

I - INCLUSÃO DAS BIBLIOTECAS
II - CRIAÇÃO DO MODELO
III - INICIALIZAÇÃO DA CLASSE
IV - INCLUSÃO DE DADOS
V - OTIMIZAÇÃO
VI - INCERTEZA
VII - PREDIÇÃO E ANALISE DE RESIDUOS
VIII - GERAÇÃO DE GRÁFICOS E RELATÓRIO
IX - OPCIONAL: PREDIÇÃO
    IX.I - PREDIÇÃO E ANALISE DE RESIDUOS
    IX.II- GRÁFICOS E RELATÓRIO

"""

# =================================================================================
# I - INCLUSÃO DAS BIBLIOTECAS
# =================================================================================
u""" 
Abaixo estão representadas as bibliotecas necessárias para o correto funcionamento do programa:

"""

 # Define que o matplotlib não usará recursos de vídeo
from matplotlib import use
use('Agg')

from MT_PEU import EstimacaoNaoLinear
from numpy import exp

# =================================================================================
# II - CRIAÇÃO DO MODELO
# =================================================================================

u"""
Similar ao Exemplo_1, k0 e E são os parametros a serem estimados, diferindo apenas a equação do modelo. 

"""

def Modelo (param, x, args):

    tempo = x[:,0:1]
    T     = x[:,1:2]

    ko = param[0]
    E  = param[1]

    y2 = exp(-ko*tempo*exp(-E*(1/T-1./630.)))

    return y2

# =================================================================================
# PARTE III - INICIALIZAÇÃO DA CLASSE
# =================================================================================

u"""
Além de alterar unidades e renomear a pasta, mais entradas opcionais são fornecidas neste exemplo. 
Através da opção label_latex é possivel formatar os titulos dos eixos dos gráficos .

"""

Estime = EstimacaoNaoLinear(Modelo, simbolos_x=[r't','T'], unidades_x=['s','K'], label_latex_x=[r'$t$','$T$'],
                            simbolos_y=[r'y'], unidades_y=['adm'],
                            simbolos_param=['ko','E'], unidades_param=['adm','K'],label_latex_param=[r'$k_o$',r'$E$'],
                            projeto='Teste2')

# =================================================================================
# PARTE IV - INCLUSÃO DE DADOS (DEPENDE DO EXEMPLO)
# =================================================================================

u"""

Os dados experimentais da variável dependente (y) e das variáveis independentes (t e T)
são disponibilizados em Schwaab e Pinto (2007, p.324), e apresentados abaixo na forma de listas:

"""
#Tempo
tempo = [120.0,60.0,120.0,60.0,30.0,15.0,45.1,90.0,150.0,60.0,60.0,30.0,150.0,90.4,120.0,60.0,60.0,60.0,30.0,
         45.1,30.0,45.0,15.0,90.0,25.0,60.1,60.0,30.0,60.0]

#Temperatura
temperatura = [600.0,612.0,612.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
               620.0,631.0,631.0,631.0,631.0,639.0,639.0,639.0,639.0,639.0,639.0,639.0]

y = [0.9,0.886,0.791,0.787,0.877,0.938,0.827,0.696,0.582,0.795,0.790,0.883,0.576,0.715,0.673,0.802,0.804,0.804,0.764,
     0.688,0.802,0.695,0.808,0.309,0.689,0.437,0.425,0.659,0.449]

u"""

Como entrada obrigatória, a plataforma MT_PEU necessita da incerteza dos dados experimentais (ux1, ux2, uy1).
Neste exemplo, foram adotados 0.001 para incerteza uxtempo, 0.05 para incerteza uxtemperatura e 1 para uy.

"""

uxtempo = [0.001]*29
uxtemperatura = [0.05]*29
uy1 = [1]*29

u"""

Inclusão de dados experimentais na estimação:
Inclui os dados experimentais nesse objeto (setDados), onde a opção 0 é para a grandeza dependente,
e a opção 1 é para a grandeza independente.

"""

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy1))

u"""

Define que os dados experimentais previamente inseridos serão utilizados como um conjunto de dados 
para o qual os parâmetros serão estimados:

"""

Estime.setConjunto(tipo='estimacao')

# =================================================================================
# PARTE V - OTIMIZAÇÃO
# =================================================================================

u"""
Neste exemplo, o usuário tem a opção de escolha do algoritmo a ser utilizado na otimização. 
Disponiveis: 'Nelder-Mead', 'Powell', 'BFGS', 'L-BFGS-B', 'CG'. 

"""

Estime.otimiza(estimativa_inicial= [0.005, 20000.000],algoritmo='Powell')

# =================================================================================
# PARTE VI - INCERTEZA
# =================================================================================

u"""
 Neste exemplo é possível escolher o método útilizado para avaliar a incerteza. 
 Métodos disponíveis: 2InvHessiana, Geral, SensibilidadeModelo. 
 Por definição o preenchimeto da região de verossimilhança é 'True', caso necessário esta opção pode ser alterada

"""

Estime.incertezaParametros(delta=1e-5,metodoIncerteza='SensibilidadeModelo',preencherregiao=False)

# =================================================================================
# VII - PREDIÇÃO E ANALISE DE RESIDUOS
# =================================================================================
u"""
 No método predição, é feita a avaliação da grandeza dependente com base nos parametros fornecidos. 
 A covariância é avaliada, e consequentemente a eficiencia do modelo. 
 Em analise de residuos é possível vericar possíveis relações de dependencia e/ou tendencia entre as variaveis. 
 Testes estatisticos como o de homocedasticidade, chi quadrado, dentre outros são realizados
 nesta etapa. A analise de residuos é feita prioritariamente com os dados de validação.

"""

Estime.predicao()
Estime.analiseResiduos()

# =================================================================================
# VIII - GRÁFICOS E RELATÓRIO
# =================================================================================

u"""
 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos. 
 Os gráficos são gerados de acordo com as etapas que foram realizadas. No relátorio contém informações a respeito
 dos testes estatisticos, função objetivo, matriz de covariância, status da otimização, dentre outros.

"""

etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)
Estime.relatorio()

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

u"""
Como entrada obrigatória, a plataforma MT_PEU necessita da incerteza dos dados experimentais (ux1, ux2, uy1).
Neste exemplo, foram adotados 0.001 para incerteza uxtempo, 0.05 para incerteza uxtemperatura e 1 para uy.

"""

uxtempo = [0.2]*12
uxtemperatura = [0.2]*12
uy1 = [0.2]*12
u"""

Inclusão de dados experimentais na estimação:
Inclui os dados experimentais nesse objeto (setDados), onde a opção 0 é para a grandeza dependente,
e a opção 1 é para a grandeza independente.

"""

Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy1))

u"""

Define que os dados de validação previamente inseridos serão utilizados como um conjunto de dados 
para o qual os parâmetros serão preditos:

"""

Estime.setConjunto(tipo='predicao')

# =================================================================================
# IX.I - PREDIÇÃO E ANALISE DE RESIDUOS
# =================================================================================

u"""
 No método predição, é feita a avaliação da grandeza dependente com base nos parametros fornecidos. 
 A covariância é avaliada, e consequentemente a eficiencia do modelo. 
 Em analise de residuos é possível vericar possíveis relações de dependencia e/ou tendencia entre as variaveis. 
 Testes estatisticos como o de homocedasticidade, chi quadrado, dentre outros são realizados
 nesta etapa. A analise de residuos é feita prioritariamente com os dados de validação.

"""

Estime.predicao()
Estime.analiseResiduos()

# =================================================================================
# IX.II- GRÁFICOS E RELATÓRIO
# =================================================================================

u"""
 Nesta etapa ocorre a geração dos dados de saída do programa : relátorios e gráficos. 
 Os gráficos são gerados de acordo com as etapas que foram realizadas, seguindo um controle de fluxo. 
 Para dados de estimação, os graficos apresentam 'fl0' na sua nomenclatura. Para dados de predição 'fl1'.
 No relátorio contém informações a respeito
 dos testes estatisticos, função objetivo, matriz de covariância, status da otimização, dentre outros.

"""

etapas = ['otimizacao','grandezas-entrada', 'predicao','grandezas-calculadas','analiseResiduos', 'regiaoAbrangencia']
Estime.graficos(etapas)
Estime.relatorio



u"""

Referências: 

SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros. 
Rio de Janeiro: e-papers, 2007.

Avaliação de dados de medição — Guia para a expressão de incerteza de medição 
http://www.inmetro.gov.br/noticias/conteudo/iso_gum_versao_site.pdf 


"""