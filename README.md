# Calculation Engine for Parameter Estimator with Uncertainty - MT_PEU

<p align="justify">
The MT_PEU is an open-source calculation engine developed for parameter estimation of steady-state models in the presence of uncertainty on the observed data. Additionally, the MT_PEU performs statistical evaluations regarding the quality of the estimation, i.e. how well the model describes the observed data, using: (i) coverage region of parameters, (ii) hypothesis testing on residues, (iii) charts describing the prediction of the model.</p>

<p align="justify">
MT_PEU builts on two main classes (EstimacaoNaoLinear and EstimacaoLinear), whose methods allow: (i) <i>optimization</i> (parameter estimation); (ii) <i>evaluation of parameter uncertainty</i> (including the coverage region); (iii) <i>evaluation of  prediction model estimates and its uncertainty</i>; and (iv) <i>residual analysis</i> (important to evaluate the model quality).
</p>

# Functionalities

<p align="justify">
This calculation engine is based on Python programming language and builts on a class structure, which main functionalities are, namely:
</p>

* **Inclusion of different datasets**
  <p align="justify"> It allows to insert observed datasets, obtained experimentally, both for parameter estimation and validation purposes. </p>

* **Parameter estimation**
  <p align="justify">
    <text> The following optimization problem is solved (Rosario, 2022): </text>  </p>
  <p align="center">
  <img src = "./Imagens/ObjectiveFunction.png">
  </p>
  
  <p align="justify"> where $\theta$ is the parameters vector, $z^e$ is the vector of experimental data, $z^m$ is the vector of model predictions, $g$ is the vector representing the model equations.</p>
  
  <p align="justify">The optimization routines were developed using symbolic computation using the <i>casadi</i> package. The following algorithms are available: (i) <i>ipopt</i>, based on interior point primal-dual method and indicated for large dimension nonlinear problems; and (ii) <i>sqpmethod</i>, which uses sequential quadratic programming. Regarding linear models on the parameters, the solution is obtained analytically.</p>


* **Uncertainty evaluation**
  <p align="justify"> The evaluation of uncertainty is carried out after the optimization step and is performed through:</p>

  <p align="center">
  <img src = "./Imagens/Geral.png">
  </p>

  <p align="justify"> where $\eta$ is the vector of model predictions, parameters and lagrange multipliers, and $L$ is the lagrangean function.</p>

* **Residual analysis**
  <p align="justify">
    <text> It is used to evaluate the residues (the difference between observed and predicted values) in order to check the statistical significance for the results (hypothesis validation). 
          The residues are evaluated by statistical testing according to its desired behavior, i.e. hypotheses imposed to obtain the objective function: (i) normality, (ii) zero mean, (iii) autocorrelation, and (iv) homoscedasticity. 
          Additionally, it is also evaluated if the optimum objective function remains inside the statistical interval, following the chi-square distribution.</text>
  </p>

* **Graphs and reports export**

  <p align="justify"> In order to provide a better view of the obtained results the MT-PEU offers charts and reports, which can be requested anywhere in the code.</p>

  * <p align="justify">There are three type of  <b>reports</b> available: (i) <i>optimization</i>, that describes the optimization procedure; (ii) <i>parameters</i>, that contains the values, covariance matrix and uncertainties obtained for the estimated parameters; and (iii) <i>prediction</i>, which presents the results of residual analysis.</p>

  * <p align="justify">There are many graphs and charts available in MT-PEU, for example: (i) <i>tendency</i>, (ii) <i>boxplot</i>, (iii) <i>autocorrelation</i>, (iv) <i>correlation matrix</i> and (iv) <i>coverage region of parameters</i>(likelihood and the one assuming that parameters follow a normal distribution).</p>

    <p align="justify"> Figure 1 shows some examples of graphs produced by MT-PEU .</p>

<p align="center">
    <img src = "./Imagens/Region.png" width="400" />
    <img src = "./Imagens/CorrelationMatrix.png" width="400" />
</p>

<p align="center">
    <img src = "./Imagens/autocorrelation.png" width="400" />
    <img src = "./Imagens/Tendencia.png" width="400" />
</p>

# How to install

<p align="justify"> To use MT-PEU it's necessary to have <b>Python 3</b> installed with the following packages:

* numpy - version 1.21.5 (available in **anaconda** distribution)
* scipy - version 1.7.3 (available in **anaconda** distribution)
* matplotlib - version 3.2.0 (available in **anaconda** distribution)
* statsmodels - version 0.13.2 (available in **anaconda** distribution)
* pandas- version 1.4.1 (available in **anaconda** distribution)
* casadi - version 3.5.5 (may be installed by pip: **pip install casadi**)

*The easiest way to install the Python 3 and the referred packages is through the anaconda distribution*: https://www.anaconda.com/distribution/
*After installing the Anaconda distribution, one can use the Anaconda Prompt and install casadi through the command: **pip install casadi**.*

Finally, one can download the MT-PEU files and, through a code Editor, like PyCharm, use the engine. A
simplest way to use MT-PEU is through the Jupyter Notebook - just start the Jupyter at Anaconda Navigator and using the interface lookfor ".ipynb" files in the MT-PEU folder.

# Getting Started

We have included examples to help users in using the MT-PEU. The most *detailed examples are
presented using the Jupyter framework*, so just start the jupyter (actually a jupyter server) at 
Anaconda Navigator, and through the interface (it will open in your browser) open one of the files Exemplo_1.ipynb to Exemplo_5.ipynb.

The same examples are presented in simple .py files to be used in code editors.

To exemplify the usage of MT-PEU, let's reproduce the Example_1.py file:

```python
#%% Packages importing
from sys import path
path.append("../../modules")#A list of strings that specifies the search path for modules
from modules.MT_PEU import EstimacaoNaoLinear
from numpy import exp

#%% Model definition
# def Model: The subroutine that specifies the equations with their respective parameters.
def model(param, z, gamma, *args):

    ko, E = param[0], param[1]
    reacfrac, time, T = z[0], z[1], z[2]

    return [reacfrac - exp(-(ko*10**17)*time*exp(-E/T))]

#%% Starting the MT_PEU main object
# Model: Pass the model defined in def Model;
# symbols_z: list of symbols for quantity z;
# symbols_uz: list of symbols for uncertainty z;
# symbols_param: list of symbols for the parameters to be estimated;
# folder: string with the name of the folder where reports and charts will be saved;
Estime = EstimacaoNaoLinear(model, symbols_z=['frac', 'time', 'temperature'], symbols_uz=['ufrac', 'utime', 'utemperature'],
                            symbols_param=['ko','E'], folder='resultado')

#%% Defining the observed data set
Frac = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]
# uncertainty of dependent variables
ufrac = [1]*41
# Observed data of independent variable (input 1)
time = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0]
# input 1 uncertainty
uxtime = [0.01]*41
# Observed data of independent variable (input 2)
temperature = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]
# input 2 uncertainty
uxtemperature = [0.01]*41

#Data entry manual
Estime.setData(data={'time':time, 'utime':uxtime, 'temperature':temperature,
                      'utemperature':uxtemperature,'frac':Frac,'ufrac':ufrac})

#%% Optimization - estimating the parameters
# initial_estimate: list containing initial estimate for optimization algorithm
Estime.optimize(initial_estimative=[0.5,25000]+Frac+time+temperature,
                lower_bound=[0,20000]+[0]*41+[0]*41+[500]*41,
                upper_bound=[1,30000]+[1]*41+[200]*41+[700]*41)

#%% Evaluating the parameters uncertainty and coverage region
# using solely default options
Estime.setupSolveModel(['frac'], ['time','temperature'])

Estime.uncertainty(objectiveFunctionMapping=True, iterations=50,  searchLimitFactor=1/10, compresscov=5e3)

#%% prediction
Estime.prediction()

#%% Evaluating residuals and quality index
# using solely default options
Estime.residualAnalysis()

#%% Plotting the main results
# using solely default options
Estime.plots()

Estime.reports(export_z=True, export_cov_z=True)

#%% Reference of this case study
# SCHWAAB, M.M.;PINTO, J.C. Análise de Dados Experimentais I: Fundamentos da Estátistica e Estimação de Parâmetros.
# Rio de Janeiro: e-papers, 2007.
#%%
```

# References
This project is based in:

* BARD, Y. Nonlinear parameter estimation. New York: Academic Press, 1974
* SCHWAAB, M. M.; PINTO, J. C. Análise de Dados Experimentais I: Fundamentos da Estatística e Estimação de Parâmetros. Rio de Janeiro: e-papers, 2007.
* Rosario, T.C. Abordagem simultânea na reconciliação de dados e estimação de parâmetros: avaliação da matriz de covariância e regiões de abrangência das variáveis de decisão. Thesis (Master). Universidade Federal da Bahia, Salvador - BA, 2022.