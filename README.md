# Parameter Estimator with Uncertainty - MT_PEU

<p align="justify">
O MT_PEU é um motor de cálculo escrito em linguagem Python, desenvolvido para aplicações no campo da estimação de parâmetros de modelos lineares e não lineares em estado estacionário com avaliação da incerteza.
</p>

# Funcionalidades

O motor de cálculo foi construído numa estrutura de classes cujas principais funcionalidades são:

* **Estimação de parâmetros de um modelo**   
  <p align="justify">
    <text> Para modelos não lineares, os parâmetros são obtidos por meio da minimização da função objetivo de mínimos quadrados ponderada pela incerteza ao quadrado (Equação 1). As rotinas de otimização foram desenvolvidas utilizando computação simbólica por meio do pacote <em>Casadi</em>. Dentre os algoritmos de otimização disponíveis, tem-se: (i) <em>ipopt</em>, baseado no método primal-dual do ponto interior e indicado para problemas não lineares de dimensão elevada; e (ii) <em>sqpmethod</em>, o qual utiliza programação quadrática sequencial. Para modelos lineares a solução é obtida analiticamente.</text>
  </p>

  <p align="center">
  <img src = "./Imagens/ObjectiveFunction.png">
  </p>
  
* **Cálculo de incerteza dos parâmetros** 
  <p align="justify">Três métodos de cálculo estão disponíveis: (i) 2InvHessiana, Equação (2); (ii) Geral, Equação (3); e (iii) SensibilidadeModelo, Equação (4). 
   </p>
  
  <p align="center">
  <img src = "./Imagens/2invHessian.png">
  </p> 
  
  <p align="center">
  <img src = "./Imagens/Geral.png">
  </p> 
  
  <p align="center">
  <img src = "./Imagens/Sensibilidade.png">
  </p> 
  
* **Cálculo de incerteza das grandezas de saída estimadas**

  <p align="justify">
		<text>É realizado através da Equação (5). </text>
  </p>
  
  <p align="center">
  <img src = "./Imagens/Uyy.png">
  </p> 

* **Análise de resíduos**
  <p align="justify">
    <text>Utilizada para avaliar os resíduos (diferença entre os valores observados e preditos) obtidos após a estimação dos parâmetros de modo a investigar a significância estatística dos resultados. Os resíduos são avaliados por meio de uma série de testes estatísticos relacionados às seguintes características: (i) normalidade; (ii) média; (iii) autocorrelação; (iv) e homocedasticidade. Além disso, também é avaliado estatisticamente o valor da função objetivo.</text>
  </p>

* **Exportação de gráficos e relatórios**

  <p align="justify">
	De modo a proporcionar ao usuário uma melhor visualização dos resultados obtidos, o MT_PEU oferece a exportação de gráficos e relatórios, os quais podem ser solicitados em qualquer região do script de código.
  </p>
  <p align="justify">
  - São três tipos de <strong>relatórios</strong> disponíveis: (i) otimização, o qual descreve o procedimento de otimização realizado para encontrar a solução ótima; (ii) parâmetros, o qual informa valores, variâncias e incertezas obtidas para os parâmetros estimados; e (iii) predição, o qual apresenta os resultados dos testes estatísticos realizados para os resíduos. 
  </p>
  <p align="justify">
  - Há uma variedade de <strong>gráficos</strong> disponíveis para as grandezas envolvidas no problema, dentre eles: (i) tendência; (ii) boxplot; (iii) autocorrelação; e (iv) matriz de correlação; para os parâmetros há o gráfico de região de abrangência dos parâmetros
  </p>

# Instalação

<p align="justify">
A instalação do MT_PEU é realizada por meio do download dos arquivos através do link (https://github.com/ddss/MT_PEU). Para o correto funcionamento do motor de cálculo, é necessário que os seguintes pacotes estejam instalados: 
</p>

* numpy
* scipy
* matplotlib
* casadi

Com exceção do casadi, todos vem disponíveis na distribuição anaconda.

# Exemplo prático
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
