# Calculation Engine for a Parameter Estimator with Uncertainty - MT-PEU

<p align="justify">
O MT-PEU é um motor de cálculo open-source e gratuito voltado para a estimação de parâmetros de modelos (lineares e não-lineares), em estado estacionário, na presença de incertezas dos dados observados, fornecendo ferramentas necessárias para realizar avaliações estatíticas sobre a qualidade do modelo (região de abrangência dos parâmetros, teste de hipóteses sobre os resíduos). 
	
Seu uso envolve duas principais classes, EstimacaoNaoLinear e EstimacaoLinear, cujos métodos permitem: (i) <em>otimização</em> (estimação dos parâmetros), (ii) <em>avaliação da incerteza dos parâmetros</em> (incluindo sua região de abrangência de verossimilhança), (iii) <em>avaliação da incerteza da predição do modelo</em>, e (iv) <em>análise de resíduos</em> (métricas úteis para avaliação da qualidade do modelo). 
</p>

# Funcionalidades

O motor de cálculo está baseado na linguagem Python (bibliotecas: Numpy, Scipy, Statsmodel, casadi, matplotlib), construído numa estrutura de classes cujas principais funcionalidades são:

* **Conjunto de dados**
  <p align="justify"> Permite a inclusão de conjuntos de dados experimentais tanto para as etapas de estimação de parâmetros, quando para validação.</p>

* **Estimação de parâmetros de um modelo**   
  <p align="justify">
    <text> Os parâmetros são obtidos por meio da minimização da função objetivo de mínimos quadrados ponderada pelo inverso da variância: </text>  </p>
  <p align="center">
  <img src = "./Imagens/ObjectiveFunction.png">
  </p>
  <p align="justify">
   As rotinas de otimização foram desenvolvidas utilizando computação simbólica por meio do pacote <em>Casadi</em>. Dentre os algoritmos de otimização disponíveis, tem-se: (i) <em>ipopt</em>, baseado no método primal-dual do ponto interior e indicado para problemas não lineares de dimensão elevada; e (ii) <em>sqpmethod</em>, o qual utiliza programação quadrática sequencial. Para modelos lineares a solução é obtida analiticamente.
</p>
  
* **Avaliação de incerteza dos parâmetros** 
  <p align="justify"> A avaliação da incerteza dos parâmetros, após a otimização, pode ser realizada através de três métodos:
   </p>
   
  * Geral, baseado na sensibilidade da função objetivo à pequenas variações do parâmetro no ponto ótimo:
  
  <p align="center">
  <img src = "./Imagens/Geral.png"> 
  </p>
  
  * 2InvHessiana, baseado em uma aproximação de (1):
  
  <p align="center">
  <img src = "./Imagens/2invHessian.png">
  </p> 
  
  * SensibilidadeModelo, baseado em uma aproximação de (1):
  <p align="center">
  <img src = "./Imagens/Sensibilidade.png">
  </p> 
  
  Recomenda-se comparar as matrizes de covariância dos parâmetros obtidas, pelos métodos, de forma a garantir consistência do resultado.
  
* **Avaliação da incerteza das grandezas de saída estimadas**

  <p align="justify">
		<text>Com base nos parâmetros estimados e no conjunto de dados informado, é avaliado a predição do modelo bem como a incerteza associada: </text>
  </p>
  
  <p align="center">
  <img src = "./Imagens/Uyy.png">
  </p> 

* **Análise de resíduos**
  <p align="justify">
    <text>Utilizada para avaliar os resíduos (diferença entre os valores observados e preditos) obtidos após a estimação dos parâmetros de modo a investigar a significância estatística dos resultados (validação das hipóteses). Os resíduos são avaliados por meio de uma série de testes estatísticos relacionados às seguintes características: (i) normalidade, (ii) média, (iii) autocorrelação, e (iv) homocedasticidade. Além disso, também é avaliado estatisticamente o valor da função objetivo.</text>
  </p>

* **Exportação de gráficos e relatórios**

  <p align="justify">
	De modo a proporcionar ao usuário uma melhor visualização dos resultados obtidos, o MT-PEU oferece a exportação de gráficos e relatórios, os quais podem ser solicitados em qualquer parte do código.
  </p>
  
  * São três tipos de **relatórios** disponíveis: (i) *otimização*, o qual descreve o procedimento de otimização realizado para encontrar a solução ótima; (ii) *parâmetros*, o qual informa valores, variâncias e incertezas obtidas para os parâmetros estimados; e (iii) *predição*, o qual apresenta os resultados dos testes estatísticos realizados para os resíduos. 
  
  * Há uma variedade de **gráficos** disponíveis para as grandezas envolvidas no problema, dentre eles: (i) *tendência*, (ii) *boxplot*, (iii) *autocorrelação*, e (iv) *matriz de correlação*. Para os parâmetros, é gerado o gráfico da região de abrangência (elipsoidal e de verossimilhança).
  
# Instalação

<p align="justify">
A instalação do MT-PEU é realizada por meio do download dos arquivos através do link (https://github.com/ddss/MT_PEU/Teste). Para o correto funcionamento do motor de cálculo, é necessário que os seguintes pacotes estejam instalados: 
</p>

* numpy
* scipy
* matplotlib
* casadi
* statsmodels

Com exceção do casadi, todos os pacotes estão disponíveis na distribuição Anaconda.

# Exemplo
```python

# packages imports
from MT_PEU import EstimacaoNaoLinear
from numpy import exp

# model definition
def Modelo(param,x,*args):

    ko, E = param[0], param[1]
    tempo, T = x[:,0], x[:,1]

    return exp(-(ko*10**17)*tempo*exp(-E/T))

# class initializing
Estime = EstimacaoNaoLinear(Modelo, simbolos_x=['t','Tao'], simbolos_y=['y'], simbolos_param=['ko','E'], Folder='Exemplo1')

# dependent quantity observed data
y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]

# independent quantity observed data
tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]

# independent quantity observed data
temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

uy = [1]*41; uxtempo = [1]*41; uxtemperatura = [1]*41

# passing data to the MT_PEU
Estime.setDados(0,(tempo,uxtempo),(temperatura,uxtemperatura))
Estime.setDados(1,(y,uy))

# setting the dataset
Estime.setConjunto(tipo='estimacao')

# plotting a graph with observed data
Estime.graficos()

# executing the parameter estimation process
Estime.optimize(initial_estimative=[0.5,25000], algoritmo='ipopt')

# calculating parameters uncertainty
Estime.incertezaParametros(metodoIncerteza='Geral')

# model's predictions
Estime.predicao()

# residuals analysis
Estime.analiseResiduos()

# plotting graphs with residuals analysis and predicted data
Estime.graficos()
```
# Referências
Este projeto basea-se nos trabalhos de:

* BARD, Y. Nonlinear parameter estimation. New York: Academic Press, 1974
* SCHWAAB, M. M.; PINTO, J. C. Análise de Dados Experimentais I: Fundamentos da Estatística e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.
