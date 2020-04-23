# Calculation Engine for Parameter Estimator with Uncertainty - MT_PEU

<p align="justify">
The MT_PEU is an open-source calculation engine developed for steady-state model parameter estimation applications (linear and nonlinear cases) in the presence of uncertainty for observed data. In addition, the MT_PEU perform statistical evaluations about in relation to the model quality(região de abrangência dos parâmetros and hypothesis test on residues).</p>

<p align="justify">
Two main class are present in MT_PEU (EstimacaoNaoLinear and EstimacaoLinear) whose methods allows: (i) <i>optimization</i> (parameter estimation); (ii) <i>parameter uncertainty evaluation</i> (includ the likelihood region); (iii) <i>uncertainty evaluation for prediction model estimatives</i>; and (iv) <i>residual analysis</i> (important to evaluate the model quality).
</p>

# Functionalities

<p align="justify">
The calculation engine is based on python language (packages: Numpy, Scipy, Statsmodel, casadi, matplotlib) and constructed on class structure whose main functionalities are:
</p>

* **Dataset**
  <p align="justify"> Allows to insert experimental datasets both for parameter estimation steps and validation.

* **Model parameter estimation**
  <p align="justify">
    <text> The parameters are obtained through weighted least squares objective function minimization. </text>  </p>
  <p align="center">
  <img src = "./Imagens/ObjectiveFunction.png">
  </p>
  <p align="justify">The optimization routines were developed by symbolic computation using <i>casadi</i> package. The follow algorithms are available: (i) <i>ipopt</i>, based on interior point primal-dual method and indicated for large dimension nonlinear problems; and (ii) <i>sqpmethod</i>, that uses sequential quadratic programming. For linear models the solution is analytic.</p>

* **Parameter uncertainty evaluation**
  <p align="justify"> The parameter uncertainty evaluation, after the optimization step, can be performed through three methods.</p>

  * Geral, based on objective function sensibility to little variations in the parameters at optimal point.

  <p align="center">
  <img src = "./Imagens/Geral.png">
  </p>

  * 2InvHessiana, based on an approximation of (1):

  <p align="center">
  <img src = "./Imagens/2invHessian.png">
  </p>

  * SensibilidadeModelo, based on an approximation of (1):
  <p align="center">
  <img src = "./Imagens/Sensibilidade.png">
  </p>

    <p align="justify">It's recommended to compare the parameter covariance matrix obtained by each method, in order to ensure that the results are consistent.</p>

* **Uncertainty evaluation for out quantities estimated**

  <p align="justify">
		<text>The model prediction as well as the associated uncertainty is evaluated based on estimated parameters and experimental dataset: </text>
	</p>

  <p align="center">
  <img src = "./Imagens/Uyy.png">
  </p>

* **Residual analysis**
  <p align="justify">
    <text> Is used to evaluate the residues (difference between observed and predicted values) obtained after the parameters estimation in order to check the statistical significance for the results (hypothesis validation). The residues are evaluated by statistical tests according to to the follow features: (i) normality, (ii) mean, (iii) autocorrelation, and (iv) homoscedasticity. In addition it's also statistically evaluated the objective function value.</text>
  </p>

* **Graphs and reports export**

  <p align="justify"> In order to provide a better view of the obtained results the MT_PEU offer a graphs and reports export which can be requested anywhere in the code.</p>

  * <p align="justify">There are three <b>report</b> types available: (i) <i>optimization</i>, that describes the optimization procedure; (ii) <i>parameters</i>, that contains the values, variances and uncertainties obtained for the estimated parameters; and (iii) <i>prediction</i>, which presents the results of statistical tests applied for residues.</p>

  * <p align="justify">There are many graphs available in MT_PEU, for example: (i) <i>tendecy</i>, (ii) <i>boxplot</i>, (iii) <i>autocorrelation</i>, (iv) <i>correlation matrix</i> and (iv) <i>Região de abrangência dos parâmetros</i>(elipsoidal e de verossimilhança).</p>

    <p align="justify"> Figure 1 shows some graphs produced by MT_PEU .</p>

<p align="center">
    <img src = "./Imagens/Region.png" width="400" />
    <img src = "./Imagens/CorrelationMatrix.png" width="400" />
</p>

<p align="center">
    <img src = "./Imagens/autocorrelation.png" width="400" />
    <img src = "./Imagens/Tendencia.png" width="400" />
</p>

# How to install

<p align="justify"> To install MT_PEU it's necessary download the archives through the link (https://github.com/ddss/MT_PEU/Teste). In addition, it's necessary to install the follow packages:</p>

* numpy - version 1.16.2 (available in **anaconda** distribution)
* scipy - version 1.2.0 (available in **anaconda** distribution)
* matplotlib - version 3.1.1 (available in **anaconda** distribution)
* casadi - version 3.4.5 (may be installed by pip: **pip install casadi**)
* statsmodels - version 0.9.0 (available in **anaconda** distribution)

*link for download anaconda distribution*: https://www.anaconda.com/distribution/

# Getting Started

Let's start with an pratical example using the python's interface.

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
* <p align="justify">This and other examples can be found in <i>Jupyter Notebook</i> plataform</p>
# References
This project is based in:

* BARD, Y. Nonlinear parameter estimation. New York: Academic Press, 1974
* SCHWAAB, M. M.; PINTO, J. C. Análise de Dados Experimentais I: Fundamentos da Estatística e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.
