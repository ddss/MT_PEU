 # -*- coding: utf-8 -*-
"""
Main class of the PEU calculation engine

@author(s): Daniel, Francisco, Anderson, Victor, Leonardo, Regiane
@ResearchGroup: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# --------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------
# Scientific calculations
from numpy import array, size, linspace, min, max, copy,\
    mean, nanmax, nanmin, arange,inf, reshape, asarray
from numpy.core.multiarray import ndarray
from numpy.random import uniform, triangular
from scipy.stats import f, t, chi2
from scipy.special import factorial
from numpy.linalg import inv
import pandas as pd #importação da biblioteca pandas para possibilitar trabalhar com o xlsx
from math import floor, log10
#from threading import Thread
from scipy import transpose, dot, concatenate, matrix
from scipy.optimize import  minimize, rosen, rosen_der
# Operating System Packages
from os import getcwd, sep
from casadi import MX,DM,vertcat,horzcat,nlpsol,sum1,jacobian,hessian,mtimes,inv as inv_cas, diag,Function
# Exception Handling
from warnings import warn
from os import listdir
from collections import Counter

# System
#TODO: CORRIGIR ENCONDING
#import sys
#sys.setdefaultencoding("utf-8") # Forçar o sistema utilizar o coding utf-8

# ----------------------------------------------------------------
# IMPORT OF OWN SUBROUTINES AND ADAPTATIONS (DEVELOPED BY GI-UFBA)
# ----------------------------------------------------------------
from Grandeza import Grandeza
from subrotinas import Validacao_Diretorio, eval_cov_ellipse, WLS
from Graficos import Grafico
from Relatorio import Report
from Flag import flag

class EstimacaoNaoLinear:

    class Fluxo:

        def __init__(self):
            u"""
            __init__(self)

            =================================================================================
            Class to control the flux of execution of methods of the EstimacaoNaoLinear class
            =================================================================================

            - Attributes
            ------------

            Each attribute represents one step of the EstimacaoNaoLinear class and can take two values:

            -0: the method was NOT executed

            -1: the method was executed

            - Methods
            ---------

            -SET_ETAPA: This method indicates which step of the estimation procedure is running.\
            To execute a step, the method evaluates if the predecessor step was executed.

            -reiniciar: restarts the flux. Assigns the value 0 to all the attributes.

            -reiniciarParcial: partially restarts the flux, for example, when entering validation data.

            - Properties
            ------------

            define the predecessor and/or successor steps of each step (attribute)

            """
            self.setDados = 0
            self.otimizacao = 0
            self.SETparametro = 0
            self.GETFOotimo = 0
            self.incertezaParametros = 0
            self.regiaoAbrangencia = 0
            self.predicao = 0
            self.analiseResiduos = 0
            self.armazenarDicionario = 0
            self.mapeamentoFO = 0
            self.Hessiana = 0
            self.Gy = 0
            self.S = 0

            self.__fluxoID = 0

        def SET_ETAPA(self,etapa,ignoreValidacao=False):
            u"""
            SET_ETAPA(self,etapa,ignoreValidacao=False)

            ================================================
            Method for defining a stage and its validations
            ================================================

            - Parameters
            ------------

            etapa : string
                defines the stage that is in execution and will be validated.
                It must have the same name as one of the attributes (in __init__).
            ignoreValidacao : bool
                will ignore the validation and assign the value 1 to the stage.

            - Notes
            --------

            If the method occurs without errors, it will assign 1 to the step.

            """
            if not ignoreValidacao:
                # Test to verify if the predecessor steps were executed.
                teste = [getattr(self,elemento) for elemento in getattr(self, '_predecessora_'+etapa)]
                # If there is no predecessor step, the value is assigned to True.
                teste = teste if teste != [] else [True]
                # If no predecessor step has been executed, returns an error
                if not any(teste):
                    raise SyntaxError('To run the {} method you must first run {}.'.format(etapa, ' or '.join(
                        getattr(self, '_predecessora_' + etapa))))
            # Assigning the value 1 (executed) to the attribute related to the step being executed
            setattr(self, etapa, 1)

        def reiniciar(self,manter='setDados'):
            u"""
            reiniciar(self,manter='setDados')

            ====================================================
            Method used to restart the flux. (IMPACTS all steps)
            ====================================================

            - Parameters
            ------------

            manter : string
                step that will be maintained with the value 1.

            - Notes
            --------

            -By flow reset we mean assigning the value of all attributes to zero,
            i.e. as if the EstimacaoNaoLinear methods had not been executed.

            -Every time experimental data is added, the flux restarts and defines that no validation data has been entered

            """
            for atributo in vars(self).keys():
                if atributo != '_Fluxo__fluxo':
                    setattr(self, atributo, 0)

            self.__fluxoID = 0
            setattr(self, manter, 1)

        def reiniciarParcial(self, etapas=None):
            u"""
            reiniciarParcial(self, etapas=None)

            ==========================================
            Method used to restart only specific steps
            ==========================================

            - Parameters
            _____________

            etapas : list
                list with the steps that will be restarted.

            - Notes
            -------

            Every time validation data is added a new workflow begins. The objective is to correctly
            perform the prediction and residual analysis and generate the graphs and reports exclusive to this step.

            """

            etapas = etapas if etapas is not None else self._sucessoresValidacao

            for atributo in etapas:
                setattr(self, atributo, 0)

            self.__fluxoID += 1

        @property
        def FLUXO_ID(self):
            u"""
            Obtains the flux identification number
            """
            return self.__fluxoID

        @property
        def _predecessora_setDados(self):
            return []

        @property
        def _predecessora_otimizacao(self):
            return ['setDados']

        @property
        def _predecessora_SETparametro(self):
            return []

        @property
        def _predecessora_GETFOotimo(self):
            return ['otimizacao', 'SETparametro']

        @property
        def _predecessora_incertezaParametros(self):
            return ['otimizacao','SETparametro']

        @property
        def _predecessora_regiaoAbrangencia(self):
            return ['mapeamentoFO']

        @property
        def _predecessora_predicao(self):
            return ['otimizacao','SETparametro']

        @property
        def _predecessora_analiseResiduos(self):
            return ['predicao']

        @property
        def _predecessora_armazenarDicionario(self):
            return ['setDados']

        @property
        def _predecessora_mapeamentoFO(self):
            return ['incertezaParametros']

        @property
        def _predecessora_Hessiana(self):
            return ['otimizacao', 'SETparametro']

        @property
        def _predecessora_Gy(self):
            return ['otimizacao', 'SETparametro']

        @property
        def _predecessora_S(self):
            return ['otimizacao', 'SETparametro']

        @property
        def _sucessoresValidacao(self):
            return ['predicao', 'analiseResiduos', 'armazenarDicionario', 'Gy', 'S']

    def __init__(self, Model, symbols_y,symbols_uy, symbols_x,symbols_ux, symbols_param, PA=0.95, Folder='Projeto', **kwargs):
        u"""
        __init__(self, Model, symbols_y, symbols_x, symbols_param, PA=0.95, Folder='Projeto', **kwargs)

        ====================================================
        Class for estimating parameters of nonlinear models.
        ====================================================

        Class description
        ------------------

        This class has a set of methods for performing the following main functions:
        (1) obtaining the optimal point of the optimization problem, using the objective function WLS (weighted least squares);
        (2) evaluation of the uncertainty of the parameters (with evaluation of the coverage region);
        (3) estimation of the prediction;
        (4) calculation of the uncertainty of the prediction;
        and (5) residual analysis.

        Auxiliary classes
        ------------------

        -Grandeza

        -Flag

        -Graficos

        -Relatorio

        Required libraries
        -------------------

        -Numpy

        -Scipy

        -Matplotlib

        -Math

        -statsmodels

        -casadi

        -Pandas

        We recommend the use of Anaconda Python 3 distribution.

        - **Parameters**
        ----------------

        **Model : (Thread)**
            The model must return an array with the number of columns equal to the number of dependent quantities.
            It must have the following structure:

                def Model(param, x):
                    .
                    .
                    .
                return y

            Where y is the mathematical expression of the model.

            -The definition of the variables of the model must be in accordance with the order in which
            the experimental data are informed in the "setDados" method.
        **symbols_y : list**
            list with the symbols of the dependent quantities (No special characters allowed).
        **symbols_x : list**
            list with the symbols of the independent quantities (No special characters allowed).
        **symbols_param : list**
            list with the symbols of the parameters (No special characters allowed).
        **PA : float, optional**
            coverage probability. Must be between 0 and 1. Default: 0.95.
        **folder : string, optional**
            Name of the folder where the results will be stored. (No special characters allowed).

        - **kwargs**
        ------------

        **names_x : list**
            list with the names of the independent quantities.
        **units_x : list**
            list with the units of the independent quantities (Latex format is accepted).
        **label_latex_x : list**
            list with the symbols of the independent quantities in latex format.
        **names_y : list**
            list with the names of the dependent quantities.
        **units_y : list**
            list with the units of the dependent quantities (Latex format is accepted).
        **label_latex_y : list**
            list with the symbols of the dependent quantities in latex format.
        **names_param : list**
            list with the names of the parameters.
        **units_param : list**
            list with the units of the parameters (Latex format is accepted).
        **label_latex_param : list**
            list with the symbols of the parameters in latex format.
        **base_path : string**
            defines the directory to store the files generated by the calculation engine

        - **Class Methods**
        -------------------

        Main methods
        ------------

        **setDados**
            method for entering the experimental data and  defines the purpose of the experimental data included:
            (i) parameter estimation or (ii) validation. (See method documentation)


        **optimize**
            performs the optimization, based on the data set defined in setConjunto. (See method documentation)

        **parametersUncertainty**
            evaluates the uncertainty of the parameters. (See method documentation)

        **SETparameter**
            allows you to manually add the values of the parameter estimates and their covariance matrix.
            It is assumed that the parameters were estimated for the data set provided for estimation.

        **prediction**
            evaluates the model prediction and its uncertainty or using the validation data.
            If these are not available, the same estimation data will be used (See method documentation)

        **residualAnalysis**
            performs the residue analysis. (see method documentation)

        **plots**
            create the plots. (see method documentation)

        **report**
            create the reports containing the main results. (see method documentation)


        **obs**: The sequence of execution of the methods is important. This class only allows the execution of methods,
        if the predecessor steps have been executed. However, some methods have flexibility, for example:

        -setConjunto method to define the estimation data should always be executed before optimize method.

        -setConjunto method to define the validation data should always be executed before prediction method.


        - **Fluxes**
        -------------

        The EstimacaoNaoLinear class has an internal class called 'Fluxo' which validates the correct order of execution of the methods.
        It is important to note that each time the setConjunto method is uxecuted, two possibilities can occur:

        - **(1)**: Inserting validation data starts new fluxes.

        - **(2)**: Entering new estimation data deletes the flux history and restarts the procedure.

        This feature allows the evaluation of different validation data consecutively
        (through Prediction, residualAnalysis and plots methods),
        after the estimation of parameters (optimize, parametersUncertainty)

         - **Outputs**
        ---------------

        The outputs of this calculation engine are mainly in the form of attributes and graphics.
        The main attributes of an Estimation variable are:

        - **x**: object of the 'Grandeza' class that contains all the information concerning the independent quantities in the form of attributes:

            -**estimacao**: it contains information from the experimental data. Main attributes: `matriz_estimativa`, `matriz_covariancia`

            -**calculado**: it contains information from the data calculated by the model. Main attributes: `matriz_estimativa`, `matriz_covariancia`

            -**predicao**: it contains information from the estimation residues. Main attributes: `matriz_estimativa`, `estatisticas`

        - **y**: object of the 'Grandeza' class that contains all the information concerning the dependent quantities in the form of attributes: **The attributes are the same as "x" object**

        - **parametros**: object of the 'Grandeza' class that contains all the information concerning the parameters in the form of attributes:

            -**estimativa**: estimate of the parameters

            -**matriz_covariancia**: the covariance matrix

            -**matriz_correlacao**: the correlation matrix

            -**regiao_abrangencia**: points that belong to the coverage region

        - **Notes**
        ------------

        - **._configFolder**: variable that contains the name of all the folders created by the algorithm in the Graphics and Reports steps.
        Changing the contents of a key changes the names of the folders. It is allowed to change the contents of the keys (folder names), b
        ut changing the keys will cause errors.

        - **.__flag: controls the behavior of the algorithm. available Flags:

            -**dadosestimacao**: identifies if estimation data were entered

            -**dadospredicao**: identifies if validation data were entered

        - **.__base_path**: identifies the root path on which all graphs and reports will be saved

        - **.__controleFluxo**: controls the validation of the steps, and the flux execution of the calculation engine.

            -**.__controleFLuxo.FLUXO_ID**: 0 if only experimental data. >0 if 0 if validation data has been entered
            (This is used in plotting of graphs, to prevent the use of successive validation data from overwriting the graphs)

        """

        # ---------------------------------------------------------------------
        # CONTROL OF THE INFORMATION FLUX OF THE ALGORITHM
        # ---------------------------------------------------------------------
        # INFORMATION FLOW -> set of algorithm steps
        self.__controleFluxo = self.Fluxo()

        # ---------------------------------------------------------------------
        # GENERAL KEYWORD VALIDATIONS
        # ---------------------------------------------------------------------
        # Available Keywords for the input method
        self.__keywordsEntrada = ('names_x', 'units_x', 'label_latex_x', 'names_y', 'units_y', 'label_latex_y',
                                  'names_param','units_param', 'label_latex_param', 'base_path')

        # Validation to check if keywords were typed incorrectly:
        keyincorreta = [key for key in kwargs.keys() if not key in self.__keywordsEntrada]

        if len(keyincorreta) != 0:
            raise NameError('keyword(s) incorreta(s): ' + ', '.join(keyincorreta) + '.' +
                            ' Keywords disponíveis: ' + ', '.join(self.__keywordsEntrada) + '.')

        # Check if PA is between 0 and 1
        if not 0 < PA < 1:
            raise ValueError('The coverage probability must be between 0 and 1.')

        # Validation to check if the project name is a string
        if not isinstance(Folder, str):
            raise TypeError('The Folder name must be a string.')

        # Validation to check if the project name has special characters.
        if not Folder.isalnum():
            raise NameError('The folder name must not contain special characters')

        # Check if base_path is a string
        if kwargs.get(self.__keywordsEntrada[9]) is not None and not isinstance(kwargs.get(self.__keywordsEntrada[9]),
                                                                                  str):
            raise TypeError('The keyword {} must be a string.'.format(self.__keywordsEntrada[9]))

        # ---------------------------------------------------------------------
        # INITIALIZATION OF QUANTITIES
        # ---------------------------------------------------------------------
        # Variable      = Grandeza(symbols  ,symbols_uncertainty    , names                                ,units                                ,label_latex                          )
        self.x          = Grandeza(symbols_x,symbols_ux             ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        self.y          = Grandeza(symbols_y,symbols_uy             ,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        self.parametros = Grandeza(symbols_param,None               ,kwargs.get(self.__keywordsEntrada[6]),kwargs.get(self.__keywordsEntrada[7]),kwargs.get(self.__keywordsEntrada[8]))

        # Check if the symbols are different
        # set: set of distinct non-ordered elements (works with set theory)
        if len(set(self.y.simbolos).intersection(self.x.simbolos)) != 0 or len(set(self.y.simbolos).intersection(self.parametros.simbolos)) != 0 or len(set(self.x.simbolos).intersection(self.parametros.simbolos)) != 0:
            raise NameError('The symbols of the quantities must be different.')


        # ---------------------------------------------------------------------
        # OTHER VARIABLES
        # ---------------------------------------------------------------------
        # Coverage probability
        self.PA = PA

        # Incremento das derivadas numéricas
        #self._deltaHessiana = 1e-5  # Hessiana da função objetivo
        #self._deltaGy = 1e-5        # Gy (derivada parcial segunda da função objetivo em relação aos parâmetros e dados experimentais)
        #self._deltaS = 1e-5         # S (transposto do jacobiano do modelo)

        # ---------------------------------------------------------------------
        # INTERNAL VARIABLES CREATION
        # ---------------------------------------------------------------------
        # Model
        self.__modelo    = Model
        # Optimization algorithm position history (parameters) (used in optimizes and / or objective function mapping
        self.__decisonVariablesMapped = []
        # Fitness history (objective function value) of the optimization algorithm (used in optimizing and / or objective function mapping)
        self.__OFMapped = []
        # Base path for the files, if the base_path keyword is defined it will be used.
        if kwargs.get(self.__keywordsEntrada[9]) is None:
            self.__base_path = getcwd()+ sep +str(Folder)+sep ####
        else:
            self.__base_path = kwargs.get(self.__keywordsEntrada[9])

        # Flags for information control
        self.__flag = flag()
        self.__flag.setCaracteristica(['dadosestimacao','dadospredicao',
                                       'reconciliacao','mapeamentoFO',
                                       'graficootimizacao','relatoriootimizacao','Linear'])
        # use of the characteristics:
        # dadosestimacao: indicates if estimation data was entered
        # dadospredicao: indicates if prediction data was entered

        # Variable that controls the name of the folders created by the graphic methods and reports
        self._configFolder = {'plots':'Graficos',
                              'plots-{}'.format(self.__tipoGraficos[0]): 'Regiao',
                              'plots-{}'.format(self.__tipoGraficos[1]): 'Grandezas',
                              'plots-{}'.format(self.__tipoGraficos[2]): 'Predicao',
                              'plots-{}'.format(self.__tipoGraficos[3]):'Grandezas',
                              'plots-{}'.format(self.__tipoGraficos[4]):'Otimizacao',
                              'plots-{}'.format(self.__tipoGraficos[5]):'Grandezas',
                              'plots-subfolder-DadosEstimacao': 'Dados Estimacao',
                              'plots-subfolder-Dadosvalidacao': 'Dados Validacao',
                              'plots-subfolder-matrizcorrelacao': 'Matrizes Correlacao',
                              'plots-subfolder-grandezatendencia': 'Tendencia observada',
                              'report':'Reports'}




        # Report class initialization
        self._out = Report(str(self.__controleFluxo.FLUXO_ID), self.__base_path, sep + self._configFolder['report'] + sep, **kwargs)

    @property
    def __tiposDisponiveisEntrada(self):
        # Available data set
        return ('estimacao', 'predicao')

    @property
    def __AlgoritmosOtimizacao(self):
        # Availabe optimization algorithm
        return ('ipopt', 'bonmin', 'sqpmethod')

    @property
    def __tipoGraficos(self):
        return ('regiaoAbrangencia', 'grandezas-entrada', 'predicao', 'grandezas-calculadas', 'otimizacao', 'analiseResiduos')

    @property
    def __metodosIncerteza(self):
        # methods for uncertainty evaluation
        return ('2InvHessiana', 'Geral', 'SensibilidadeModelo')

    @property
    def __keywordsDerivadas(self):
        # available keywords to evaluate the derivatives
        return ('deltaHess', 'deltaGy', 'deltaS', 'delta')

    @property
    def __tipoObjectiveFunctionMapping(self):
        # objective function mapping algorithms available
        return ('MonteCarlo',)

    @property
    def __graph_flux_association(self):
        return {'setDados':[self.__tipoGraficos[1]],'incertezaParametros':[self.__tipoGraficos[0],self.__tipoGraficos[3]],
                'predicao':[self.__tipoGraficos[2],self.__tipoGraficos[3]],'analiseResiduos':[self.__tipoGraficos[5]]}

    @property
    def _args_model(self):
        """
        _args_model(self)

        ==============================================================
        Method that returns extra arguments to be passed to the model.
        ==============================================================

        """
        # ---------------------------------------------------------------------
        # LIST OF ATTRIBUTES TO INSERT IN THE MODEL
        # ---------------------------------------------------------------------

        return [self.__args_user,self.x.simbolos,self.y.simbolos,self.parametros.simbolos]

    def __validacaoDadosEntrada(self,dados,udados,NV):
        u"""
        __validacaoDadosEntrada(self,dados,udados,NV):

        ==============================================
        Method to perform the validation of input data
        ==============================================

        - Parameters
        ------------

        dados : ndarray
            contains the experimental data
        udados : ndarray
            contains the uncertainties of the experimental data
        NV : int
            Number of quantities to be validated

        - Notes
        -------

        -check if the column number of the input arrays is equal to the number of symbols of the defined variables (y, x)

        -check if the number of points is the same

        -check if the degrees of freedom are sufficient to perform the estimation

        """
        if dados.shape[1] != NV:
            raise ValueError('The number of variables defined was {:d}, but data was entered for {:d} variables.'.format(NV,dados.shape[1]))

        if udados.shape[1] != NV:
            raise ValueError('The number of variables defined was {:d}, but uncertainties were inserted for {:d}.'.format(NV,udados.shape[1]))

        if dados.shape[0] != udados.shape[0]:
            raise ValueError('Data vectors and their uncertainties must have the same number of points.')

        if udados.shape[0]*self.y.NV-float(self.parametros.NV) <= 0: # Verificar se há graus de liberdade suficiente
            warn('Insufficient degrees of freedom. Your experimental data set is not enough to estimate the parameters!',UserWarning)

    def  setDados(self, data,separador=';',decimal='.', glx=[], gly=[]):

        u"""
                setDados(self,data,worksheet=None , glx=[],gly=[]):
                ===================================================================================================================
                  Method for collecting input data from dependent and independent quantities and including estimating data
                ===================================================================================================================
                - Parameters
                ------------

                data: dict,list and  str

                      dict: manual input format
                      list: data entry type to excel CSV
                      str: data entry type to excel format and CSV



                glx : list, optional
                    list with the freedom degrees for the input quantities.
                gly : list, optional
                    list with the freedom degrees for the output quantities.


                How to use each input format

                    Dict is  manual input format

                        Estime.setDados(data={0:[(time,uxtime),(temperature,uxtemperature)],1:[(y,uy)]})

                    List is input format for CSV file
                        Estime.setDados(data=["name_data_independent.csv","name_data_exa1_dependent.csv"])


                    Str is input format for excel file:

                        (i) If the user has changed the name of the standard worksheets.

                        *Estime.setData(data="file_name.xlsx",worksheet= {0:"name_worksheet_independent-data",1:"name_worksheet_independent-data"})

                        (ii) If the user has used default names for the excel worksheets, (independent variable) and (dependent variable ).

                        *Estime.setData(date="file_name .xlsx")


        """

        # EXECUTION
        self.__controleFluxo.SET_ETAPA('setDados')
        #MANUAL DATA ENTRY
        aux_list=[]#list used for error
        if isinstance(data,dict):
        # VALIDATION TO MANUAL DATA ENTRY
            # Tests if input data lists are the same size
            for i in list(data.keys()):
                if len(data[i]) != len(data[list(data.keys())[0]]):
                    aux_list.append(i)
            if len(aux_list)==1:
                raise ValueError(f"The list {','.join(i)} differs in the amount of points from the first list")
            if len(aux_list)>1:
                raise ValueError(f"The list {','.join(i)} differs in the amount of points from the first list")

            #Test if the symbols passed in the object instantiation parameters of the MT_PEU class are all in the dataset
            lista_nomes=self.x.simbolos + self.y.simbolos + self.y.simbolos_incertezas + self.x.simbolos_incertezas
            lista_dados=data.keys()
            if len(list(set(lista_dados) & set(lista_nomes))) != len(lista_nomes):
                if len(list(set(lista_nomes) - set(lista_dados) & set(lista_nomes))) > 1:
                    raise ValueError("The symbols {} were not passed  in database".format(
                        ','.join(list(set(lista_nomes) - set(lista_dados) & set(lista_nomes)))))
                else:
                    raise ValueError("The symbol {} was not passed  in database ".format(
                        ','.join(list(set(lista_nomes) - set(lista_dados) & set(lista_nomes)))))
            #Test if an empty list was passed
            for listas in data:
                if len(data[listas]) == 0:
                    aux_list.append(listas)
            if len(aux_list) ==1:
                raise ValueError(f"Data list {','.join(aux_list)} cannot be empty")
            elif len(aux_list) >1:
                raise ValueError(f"Data lists {','.join(aux_list)} cannot be empty")


            X = transpose(array([data[i] for i in self.x.simbolos], ndmin=2, dtype=float))
            uX = transpose(array([data[i] for i in self.x.simbolos_incertezas], ndmin=2, dtype=float))
            Y = transpose(array([data[i] for i in self.y.simbolos], ndmin=2, dtype=float))
            uY = transpose(array([data[i] for i in self.y.simbolos_incertezas], ndmin=2, dtype=float))
        #-----------------------------------------------------------------------------
        # ROUTINE THAT IMPORTS AND VALIDATES DATA FROM .CSV AND .XLSX FILES
        #-----------------------------------------------------------------------------
        elif isinstance(data,list)  or isinstance(data,str): #Data input, in file import mode, accepts string or list of strings

            if  not isinstance(data,list):
                data = [data]  #The data is expected in list format, but if it is a string, it is added to a list
            ###VALIDATION###
            error=False
            for nome_validação in data:
                if not isinstance(nome_validação,str):
                    error=True
            if error:
                raise ValueError(f"You must pass a strigs in list of setDados")

            for value in Counter(data):#warns that files with repeated names were passed
                if Counter(data)[value] != 1:
                    aux_list.append(value)
            if len(aux_list) > 1:
                warn(f"Files passed with same names :{','.join(aux_list)}.", Warning)
            elif len(aux_list) == 1:
                warn(f"File passed with same name: {','.join(aux_list)}.", Warning)

            data=(list(set(data)))#remove repeated names from past file list

            lista_dataframe = []  #empty list to concatenate dataframes
            for i in range(len(data)):#iterative loop with all filenames passed
                #In case the filenames already have the extension
                if '.csv' in data[i] or '.xlsx' in data[i]:#Fetch the format in the past names
                    #Caso sejam o arquivo seja .xlsx usa essa rotina de chamada de dados
                    if '.xlsx' in data[i]:
                        # nomes_planilhas é a lista com títulos das  planilhas do excel
                        nomes_planilhas = pd.ExcelFile(data[i]).sheet_names
                        if len(nomes_planilhas) > 1:
                            #Caso o .xlsx tenha mais de uma planilha
                            data_variable = pd.read_excel(data[i], sheet_name=nomes_planilhas)
                                                          #data_variable armazena cada planilha do excel como um dataframe, em forma de dicionário
                            for i2 in data_variable:
                                lista_dataframe.append(data_variable[i2])  # cria uma lista de dataframes usando o dict de dataframes
                        else:
                            # Caso o .xlsx tenha apenas uma planilha de dados
                            lista_dataframe.append(pd.read_excel(data[i]))
                    else:
                        #Caso onde a extensão é .csv
                        lista_dataframe.append(pd.read_csv(data[i], sep=separador ,decimal=decimal))

                # Caso que o usuário não passou extensão
                else:
                    lista_arquivos_ini = listdir()  #importa lista de arquivos na mesma pasta
                    lista_arquivos = sorted(lista_arquivos_ini,
                                            key=len)  # organiza a lista em ordem crescente do tamanho das strings

                    controle = False #variável responsável por limitar que em cada iteração traga apenas um arquivo
                    for j in range(len(lista_arquivos)):
                        if not controle :
                            if data[i] in lista_arquivos[j]:#traz o arquivo que contenha o mesmo nome antes do ponto
                                name1 = lista_arquivos[j]  # nome de arquivo com o seu formato
                                #No entanto esses arquivos podem ser dados de planilha eletrônica ou csv
                                if name1.replace(data[i],'') == '.xlsx'or name1.replace(data[i],'') == '.xls'or name1.replace(data[i],'') == '.xlsm' or name1.replace(data[i],'') == '.xlsb' or name1.replace(data[i],'') == '.odf':  # Supports xls, xlsx, xlsm, xlsb, odf, ods and odt file extensions
                                    # função que traz a lista com títulos das  planilhas
                                    nomes_planilhas = pd.ExcelFile(name1).sheet_names
                                    if len(nomes_planilhas) > 1:
                                        data_variable = pd.read_excel(name1, sheet_name=nomes_planilhas)
                                        for i2 in data_variable:
                                            lista_dataframe.append(data_variable[i2])  # cria uma lista de dataframes usando o dict de dataframes
                                            controle = True
                                    else:#caso que o arquivo xlsx possui apenas uma planilha
                                        lista_dataframe.append(pd.read_excel(name1))
                                        controle = True
                                elif name1.replace(data[i], '') == '.csv' or name1.replace(data[i], '') == '.CSV':
                                    data_frame_csv=pd.read_csv(name1, decimal=decimal, sep=separador)
                                    lista_dataframe.append(data_frame_csv)
                                    controle = True
                    if controle == False:#se não houver nenhum arquivo com o nome passado pelo usuário
                        raise TypeError('There is no file with the name {} in the file folder'.format(data[i]))
            dataframe_geral = pd.concat(lista_dataframe, axis=1)  # concatenates all dataframes into a single one
            for nulos in dict(
                    dataframe_geral.isnull().sum()):  # dictionary that brings the symbols as keys and the number of empty lines as values
                if dict(dataframe_geral.isnull().sum())[
                    nulos] != 0:  # Tests if there are empty rows in the columns of each symbol
                    aux_list.append(nulos)
            if len(aux_list) > 0:
                raise ValueError(
                    f"In quantity{'s'[:int(len(aux_list)) ^ 1]} {','.join(aux_list)} there are empty lines or the quantitity of data points are inconsistenty")

            # Test if the symbols passed in the object instantiation parameters of the MT_PEU class are all in the dataset
            lista_dados=list(dataframe_geral.columns)
            lista_nomes = self.x.simbolos + self.y.simbolos + self.y.simbolos_incertezas + self.x.simbolos_incertezas
            if len(list(set(lista_dados) & set(lista_nomes))) != len(lista_nomes):
                if len(list(set(lista_nomes) - set(lista_dados) & set(lista_nomes))) > 1:
                    raise ValueError("The symbols {} were not passed in the dataset ".format(','.join(
                        list(set(lista_nomes) - set(lista_dados) & set(lista_nomes)))))
                else:
                    raise ValueError("The symbol {} was not passed in the dataset ".format(','.join(
                        list(set(lista_nomes) - set(lista_dados) & set(lista_nomes)))))

            #Creation of estimation and uncertainty matrices
            X = dataframe_geral[self.x.simbolos].to_numpy(dtype=float)
            Y = dataframe_geral[self.y.simbolos].to_numpy(dtype=float)
            uX = dataframe_geral[self.x.simbolos_incertezas].to_numpy(dtype=float)
            uY = dataframe_geral[self.y.simbolos_incertezas].to_numpy(dtype=float)

        else:
            raise TypeError(" The data input can be  a list or string or dictionary, check if the input follows any of these formats")

        # validation data
        self.__validacaoDadosEntrada(X, uX, self.x.NV)
        self.__validacaoDadosEntrada(Y, uY, self.y.NV)
        ######################################################EXECUTION#########################################################
        #Automaticamente chamando setdados a primeira vez vai para estimação, chamando dados pela segunda vez vai para validação (Predição).
        if not self.__flag.info['dadosestimacao']:  # rodando a primeira vez (estimação)
             dataType = self.__tiposDisponiveisEntrada[0]
        else:  # rodando a segunda vez, vai agora para predição
             dataType = self.__tiposDisponiveisEntrada[1]

        # experimental data
        if dataType == self.__tiposDisponiveisEntrada[0]:
            self.__flag.ToggleActive('dadosestimacao')
            # if flux ID is equal to zero, so it's not necessary to restart, otherwise, restart.
            if self.__controleFluxo.FLUXO_ID != 0:
                self.__controleFluxo.reiniciar()
                if self.__flag.info['dadospredicao']:
                    warn('The flux was restarted, so new validation data has to be included.', UserWarning)
            # ---------------------------------------------------------------------
            # ASSIGNMENT OF VALUES TO QUANTITIES
            # ---------------------------------------------------------------------
            # Saving the experimental data in the variables.
            try:
                self.x._SETdadosestimacao(estimativa=X, matriz_incerteza=uX,gL=glx)
            except Exception as erro:
                raise RuntimeError(
                    'Error in the creation of the estimation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosestimacao(estimativa=Y, matriz_incerteza=uY,gL=glx)
            except Exception as erro:
                raise RuntimeError(
                    'Error in the creation of the estimation set of the quantity Y: {}'.format(erro))

        # prediction data
        if dataType == self.__tiposDisponiveisEntrada[1]:
            self.__flag.ToggleActive('dadospredicao')
            self.__controleFluxo.reiniciarParcial()
            # ---------------------------------------------------------------------
            # ASSIGNMENT OF VALUES TO QUANTITIES
            # ---------------------------------------------------------------------
            # Saving the validation data.
            try:
                self.x._SETdadosvalidacao(estimativa=X, matriz_incerteza=uX,gL=glx)
            except Exception as error:
                raise RuntimeError(
                    'Error in the creation of the validation set of the quantity X: {}'.format(error))

            try:
                self.y._SETdadosvalidacao(estimativa=Y, matriz_incerteza=uY,gL=gly)
            except Exception as error:
                raise RuntimeError(
                    'Error in the creation of the validation set of the quantity Y: {}'.format(error))

        if not self.__flag.info['dadospredicao']:
            # If setConjunto method is only performed for experimental data,
            # it will be assumed that also are validation data because all prediction
            # calculation is performed to the validation data.
            # ---------------------------------------------------------------------
            # ASSIGNMENT OF VALUES TO QUANTITIES
            # ---------------------------------------------------------------------
            # Saving validation data.
            try:
                self.x._SETdadosvalidacao(estimativa=X, matriz_incerteza=uX,gL=glx)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=Y, matriz_incerteza=uY,gL=gly)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity Y: {}'.format(erro))

        # initialization of casadi's variables
        self._constructionCasadiVariables()

        # TODO:
        # validação de args
        # graus de liberdade devem ser passados por aqui

    def _constructionCasadiVariables(self): # construction of the casadi variables
        u"""
        _constructionCasadiVariables(self)

        ============================
        Symbolic variables creation.
        ============================

        """

        # --------------------------------------------------------------------------------
        # CREATION OF CASADI'S VARIABLES THAT WILL BE USED TO BUILD THE CASADI'S MODEL
        # --------------------------------------------------------------------- ----------

        if not self.__flag.info['dadospredicao']:
            # if no prediction data were entered, then estimation is being performed and
            # estimation data should be used

            self.__symXr   = []; self.__symUxo = [] # x
            self.__symYo    = []; self.__symYest = []; self.__symUyo = []
            self.__symParam = []

            self.__symVariables  = []
            self._values         = []

            # Creation of parameters in casadi's format
            for i in range(self.parametros.NV):
                self.__symParam = vertcat(self.__symParam,MX.sym(self.parametros.simbolos[i]))

            # Creation of independent variables in casadi's format
            xmodel = []
            for j in range(self.x.NV):
                self.__symXo = []
                for i in range(self.x.estimacao.NE):
                    self.__symXo = vertcat(self.__symXo, MX.sym('xo' + str(j + 1) + '_' + str(i)))
                    self.__symXr = vertcat(self.__symXr, MX.sym('xr' + str(j + 1) + '_' + str(i)))
                xmodel = horzcat(xmodel,self.__symXo)
                self.__symVariables = vertcat(self.__symVariables, self.__symXo)
            if self._EstimacaoNaoLinear__flag.info['Linear']:
                if self._EstimacaoNaoLinear__flag.info['calc_termo_independente']: # Testing if it's a linear case with independent term calculation
                    self._values = vertcat(self._values, self.x.estimacao.vetor_estimativa[
                                                         :self.x.estimacao.NE])  # para não trazer a coluna de '1' como dado de entrada
                else:
                    self._values = vertcat(self._values, self.x.estimacao.vetor_estimativa)
            else:
                self._values = vertcat(self._values, self.x.estimacao.vetor_estimativa)

            # Creation of dependent variables in casadi's format
            for j in range(self.y.NV):
                for i in range(self.y.estimacao.NE):
                    self.__symYo = vertcat(self.__symYo,MX.sym('yo'+str(j+1)+'_'+str(i)))
                    self.__symYest = vertcat(self.__symYest,MX.sym('y'+str(j+1)+'_'+str(i)))
            self.__symVariables = vertcat(self.__symVariables, self.__symYo)
            self._values = vertcat(self._values, self.y.estimacao.vetor_estimativa)

            # Creation of uncertainties of dependent variables in casadi's format
            for j in range(self.y.NV):
                for i in range(self.y.estimacao.NE):
                    self.__symUyo = vertcat(self.__symUyo, MX.sym('Uyo'+str(j+1)+'_'+str(i)))
            self.__symVariables = vertcat(self.__symVariables, self.__symUyo)
            self._values = vertcat(self._values,
                                   self.y.estimacao.matriz_incerteza.reshape(self.y.estimacao.NE*self.y.NV,1))

            # Model definition
            self.__symModel = self.__modelo(self.__symParam, xmodel, self.y.estimacao.NE)  # Symbolic
            self.__excModel = Function('Model', [self.__symParam, self.__symVariables],[self.__symModel])  # Executable

            # Objective function definition
            self.__symObjectiveFunction = sum1(((self.__symYo - (self.__symModel)) ** 2) / (self.__symUyo ** 2))  # Symbolic
            self._excObjectiveFunction = Function('Objective_Function', [self.__symParam, self.__symVariables],
                                                  [self.__symObjectiveFunction])  # Executable

        else:
            self.__symParam = [];
            self.__symXr    = []; self.__symUxo = []  # x
            self.__symYo    = []; self.__symYest = []; self.__symUyo = []

            self.__symVariables = []
            self._values = []

            # Creation of parameters in casadi's format
            for i in range(self.parametros.NV):
                self.__symParam = vertcat(self.__symParam, MX.sym(self.parametros.simbolos[i]))

            # Creation of independent variables in casadi's format
            xmodel = []
            for j in range(self.x.NV):
                self.__symXo = []
                for i in range(self.x.predicao.NE):
                    self.__symXo = vertcat(self.__symXo, MX.sym('xo' + str(j + 1) + '_' + str(i)))
                    self.__symXr = vertcat(self.__symXr, MX.sym('xr' + str(j + 1) + '_' + str(i)))
                xmodel = horzcat(xmodel, self.__symXo)
                self.__symVariables = vertcat(self.__symVariables, self.__symXo)  #
            if self._EstimacaoNaoLinear__flag.info['Linear']:
                if self._EstimacaoNaoLinear__flag.info['calc_termo_independente']:  # Testing if it's a linear case with independent term calculation
                    self._values = vertcat(self._values, self.x.predicao.vetor_estimativa[
                                                         :self.x.predicao.NE])  # para não trazer a coluna de '1' como dado de entrada
                else:
                    self._values = vertcat(self._values, self.x.predicao.vetor_estimativa)
            else:
                self._values = vertcat(self._values, self.x.predicao.vetor_estimativa)

            # Creation of dependent variables in casadi's format
            for j in range(self.y.NV):
                for i in range(self.y.predicao.NE):
                    self.__symYo = vertcat(self.__symYo, MX.sym('yo' + str(j + 1) + '_' + str(i)))
                    self.__symYest = vertcat(self.__symYest, MX.sym('y' + str(j + 1) + '_' + str(i)))
            self.__symVariables = vertcat(self.__symVariables, self.__symYo)
            self._values = vertcat(self._values, self.y.predicao.vetor_estimativa)

            # Creation of uncertainties of dependent variables in casadi's format
            for j in range(self.y.NV):
                for i in range(self.y.predicao.NE):
                    self.__symUyo = vertcat(self.__symUyo, MX.sym('Uyo' + str(j + 1) + '_' + str(i)))
            self.__symVariables = vertcat(self.__symVariables, self.__symUyo)
            self._values = vertcat(self._values,
                                   self.y.predicao.matriz_incerteza.reshape(self.y.predicao.NE * self.y.NV, 1))

            # Model definition
            # it's necessary to define a new model because the
            # prediction data size could be different of the estimation data size
            self.__symModel = self.__modelo(self.__symParam, xmodel, self.y.predicao.NE)  # Symbolic
            self.__excModel = Function('Model', [self.__symParam, self.__symVariables], [self.__symModel])  # Executable

    def _armazenarDicionario(self):
        u"""
        Método opcional para armazenar as Grandezas (x,y e parâmetros) na
        forma de um dicionário, cujas chaves são os símbolos.

        ======
        Saídas
        ======

        * grandeza: dicionário cujas chaves são os símbolos das grandezas e respectivos
        conteúdos objetos da classe Grandezas.
        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('armazenarDicionario')

        # ---------------------------------------------------------------------
        # GERANDO O DICIONÁRIO
        # ---------------------------------------------------------------------    

        grandeza = {}

        # GRANDEZAS DEPENDENTES (y)
        for j, simbolo in enumerate(self.y.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.y.nomes[j]],[self.y.unidades[j]],[self.y.label_latex[j]])

            # Salvando os dados estimação
            if self.__flag.info['dadosestimacao']:
                # Salvando dados experimentais
                grandeza[simbolo]._SETdadosestimacao(estimativa=self.y.estimacao.matriz_estimativa[:,j:j+1],
                                                   matriz_incerteza=self.y.estimacao.matriz_incerteza[:,j:j+1],
                                                   gL=self.y.estimacao.gL[j])

            # Salvando os dados predição
            if self.__flag.info['dadospredicao']:
                # Salvando dados estimação
                grandeza[simbolo]._SETdadosvalidacao(estimativa=self.y.predicao.matriz_estimativa[:,j:j+1],
                                                matriz_incerteza=self.y.predicao.matriz_incerteza[:,j:j+1],
                                                gL=self.y.predicao.gL[j])

            # Salvando os dados calculados
            if self.__controleFluxo.predicao:
                grandeza[simbolo]._SETcalculado(estimativa=self.y.calculado.matriz_estimativa[:,j:j+1],
                                                matriz_incerteza=self.y.calculado.matriz_incerteza[:,j:j+1],
                                                gL=self.y.calculado.gL[j])

            # Salvando os resíduos
            if self.__controleFluxo.analiseResiduos:
                grandeza[simbolo]._SETresiduos(estimativa=self.y.residuos.matriz_estimativa[:,j:j+1])

        # GRANDEZAS INDEPENDENTES (x)
        for j, simbolo in enumerate(self.x.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.x.nomes[j]],[self.x.unidades[j]],[self.x.label_latex[j]])

            # Salvando dados estimação
            if self.__flag.info['dadosestimacao']:
                grandeza[simbolo]._SETdadosestimacao(estimativa=self.x.estimacao.matriz_estimativa[:,j:j+1],
                                                   matriz_incerteza=self.x.estimacao.matriz_incerteza[:,j:j+1],
                                                   gL=self.x.estimacao.gL[j])

            # Salvando dados de predição
            if self.__flag.info['dadospredicao']:
                grandeza[simbolo]._SETdadosvalidacao(estimativa=self.x.predicao.matriz_estimativa[:,j:j+1],
                                                matriz_incerteza=self.x.predicao.matriz_incerteza[:,j:j+1],
                                                gL=self.x.predicao.gL[j])

            # Salvando dados calculados
            if self.__controleFluxo.predicao:
                grandeza[simbolo]._SETcalculado(estimativa=self.x.calculado.matriz_estimativa[:,j:j+1],
                                                matriz_incerteza=self.x.calculado.matriz_incerteza[:,j:j+1],
                                                gL=self.x.calculado.gL[j])

            # Salvando os resíduos
            if self.__controleFluxo.analiseResiduos:
                grandeza[simbolo]._SETresiduos(estimativa=self.x.residuos.matriz_estimativa[:,j:j+1])

        # PARÂMETROS
        for j,simbolo in enumerate(self.parametros.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.parametros.nomes[j]],[self.parametros.unidades[j]],[self.parametros.label_latex[j]])
            if self.__controleFluxo.otimizacao or self.__controleFluxo.SETparametro:
                # Salvando as informações dos parâmetros
                if self.parametros.matriz_covariancia is None:
                    grandeza[simbolo]._SETparametro([self.parametros.estimativa[j]],None,None)
                else:
                    grandeza[simbolo]._SETparametro([self.parametros.estimativa[j]],array([self.parametros.matriz_covariancia[j,j]],ndmin=2),None)

        return grandeza

    def optimize(self, initial_estimative, lower_bound=-inf, upper_bound=inf, algorithm ='ipopt', optimizationReport = True, parametersReport = False):
        u"""
        optimize(self, initial_estimative, lower_bound=-inf, upper_bound=inf, algorithm ='ipopt', optimizationReport = True, parametersReport = False)

        ==============================
        Solve the optimization problem.
        ==============================

        - Parameters
        ------------

        initial_estimative : list
            list with the initial estimates for the parameters.
        lower_bound : list, optional
            list with the lower bounds for the parameters.
        upper_bound : list, optional
            list with the upper bounds for the parameters.
        algorithm : string, optional
            informs the optimization algorithm that will be used. Each algorithm has its own keywords.

            ==================== ===================================================
            available algorithms                     font
            ==================== ===================================================
            ipopt                https://github.com/coin-or/Ipopt
            bonmin               https://github.com/coin-or/Bonmin
            sqpmethod            http://casadi.sourceforge.net/v1.9.0/api/html/de/dd4/classCasADi_1_1SQPMethod.html
            ==================== ===================================================

        optimizationReport : bool, optional
            informs whether the optimization report should be created.
        parametersReport : bool, optional
            informs whether the parameters report should be created.

        - Notes
        -------

        -Before executing the optimize method it's necessary to execute the "setConjunto" method
         and define the estimation data.

        -Every time the optimization method is run, the information about the parameters is lost.
        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('otimizacao')
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        # if don't have experimental data -> error
        if not self.__flag.info['dadosestimacao']:
            raise SyntaxError('To execute the optimize method is necessary to input the estimation data.')

        # the SETparameter method must not be executed before the optimize method.
        if self.__controleFluxo.SETparametro:
            raise SyntaxError('The method {} cannot be executed before {}'.format('optimize', 'SETparameter'))

        # check if the algorithm argument has string type
        if not isinstance(algorithm, str):
            raise TypeError('The algorithm name must be a string.')

        # check if the informed algorithm is available.
        if not algorithm in self.__AlgoritmosOtimizacao:
            raise NameError(
                'The algorithm option {} is not right. Available algorithms: '.format(algorithm) + ', '.join(
                    self.__AlgoritmosOtimizacao) + '.')

        # validation of the initial estimative:
        if initial_estimative is None:
            raise SyntaxError('To execute the optimize method it is necessary to give an initial estimative')
        if not isinstance(initial_estimative, list) or len(initial_estimative) != self.parametros.NV:
            raise TypeError(
                'The initial estimative must be a list with the size of the number of parameters, defined in the symbols. Number of parameters: {}'.format(
                    self.parametros.NV))

        # ---------------------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------------------
        # EstimacaoNaoLinear only performs estimation WITHOUT data reconciliation
        self.__flag.ToggleInactive('reconciliacao')
        # indicates that this algorithm has performance reporting.
        self.__flag.ToggleActive('relatoriootimizacao')

        # ---------------------------------------------------------------------
        # MODEL VALIDATION
        # ---------------------------------------------------------------------

        # Check if the model is executable in the search boundaries.
        try:  # Validates the informed upper and lower limits. The initial estimative of parameters is required.
            if upper_bound is not None:
                aux = self.__excModel(upper_bound, self._values)
            if lower_bound is not None:
                aux = self.__excModel(lower_bound, self._values)
            if initial_estimative is None:
                aux = self.__excModel(initial_estimative, self._values)

        except Exception as erro:
            raise SyntaxError(
                u'Error in the model when evaluated within the defined search limits. Error identified: "{}".'.format(erro))

        # ---------------------------------------------------------------------
        # PEFORMS THE OPTIMIZATION
        # ---------------------------------------------------------------------
        # define the optimization problem
        nlp = {'x': self.__symParam, 'p': self.__symVariables, 'f': self.__symObjectiveFunction}

        # options for printing the optimization information
        if optimizationReport is True:
            # with optimization report
            if algorithm == 'ipopt':
                options = {'print_time': False, 'ipopt' :{'print_level': 0, 'file_print_level': 5,
                                                          'output_file': self._out.optimization()+  'Optimization_report.txt'}}
            elif algorithm == 'bonmin':
                options = {'print_time': False, 'bonmin':{'file_print_level': 5,
                                                          'output_file': self._out.optimization() + 'Optimization_report.txt'}}
            elif algorithm =='sqpmethod':
                options = {'print_iteration': False, 'qpsol_options':{'printLevel': 'none'}}

        else:
            # without optimization report
            if algorithm == 'ipopt':
                options = {'print_time': False, 'ipopt': {'print_level': 0}}
            elif algorithm == 'bonmin':
                options = {'print_time': False, 'bonmin': {}}
            elif algorithm == 'sqpmethod':
                options = {'print_iteration': False, 'qpsol_options': {'printLevel': 'none'}}
        # optimization problem setup
        S = nlpsol('S', algorithm, nlp, options)
        # passing the arguments for the optimization problem
        self.Otimizacao = S(x0=initial_estimative, p=self._values, lbx=lower_bound, ubx=upper_bound)

        # ASSIGNMENT OF VALUES TO QUANTITIES

        # ---------------------------------------------------------------------
        # OPTIMAL POINT OF THE OBJECTIVE FUNCTION
        # ---------------------------------------------------------------------
        self.FOotimo = float(self.Otimizacao['f'])
        # ---------------------------------------------------------------------
        # OPTIMAL VALUE OF THE PARAMETERS
        # ---------------------------------------------------------------------
        self.__opt_param = [float(self.Otimizacao['x'][i]) for i in range(self.parametros.NV)] # converts DM type in float type

        # every time optimization is run all previous information about parameters is lost
        self.parametros._SETparametro(self.__opt_param, None, None, limite_superior=upper_bound,limite_inferior=lower_bound)

        # check if the parameters estimative is equal to the informed boundaries
        if lower_bound != -inf and upper_bound != inf:
            for i in range(self.parametros.NV):
                if self.parametros.estimativa[i] == lower_bound[i] or self.parametros.estimativa[i] == upper_bound[i]:
                    warn('Estimated parameters equal to the upper or lower limit.')

        # parameters report creation
        if parametersReport is True:
            self._out.Parametros(self.parametros,self.FOotimo)

         #Conversion of the optimization report to html
        if optimizationReport is not False:
            with open(self._out.optimization() +'Optimization_report.txt', 'r') as f:
                n_linhas = len(f.readlines())
            # lendo as linhas do arquivo
            with open(self._out.optimization() +'Optimization_report.txt', 'r') as arquivo:
                linhas = arquivo.readlines()  # cada linha é um elemento da lista linhas
            for i in range(n_linhas):  # editando a segunda linha
                linhas[i] = linhas[i] + '<p>'
            # escrevendo de novo
            with open(self._out.optimization() +'Optimization_report.html', 'w') as arquivo:
                arquivo.writelines(linhas)

    def __Hessiana_FO_Param(self):

        aux = Function('Hessiana', [self.__symParam, self.__symVariables],
                                 [hessian(self.__symObjectiveFunction,self.__symParam)[0]]) #function

        self.Hessiana = array(aux(self.parametros.estimativa, self._values)) #numeric

        return self.Hessiana

    def __Matriz_Gy(self):

        aux = Function('Gy', [self.__symParam, self.__symVariables],
                       [jacobian(jacobian(self.__symObjectiveFunction, self.__symParam), self.__symYo)]) # function

        self.Gy = array(aux(self.parametros.estimativa, self._values))

        return self.Gy

    def __Matriz_S(self):
        u"""
               Method for calvulate the array S(first derivatives of the model function in relation to the parameters)."""

        aux = Function('S', [self.__symParam, self.__symVariables], [jacobian(self.__symModel,self.__symParam)])

        #if not self.__flag.info['dadospredicao']:

        self.S = array(aux(self.parametros.estimativa, self._values))

        return self.S


    def SETparameter(self,estimative,variance=None,region=None,parametersReport=True,**kwargs):
        u"""
        SETparameter(self,estimative,variance=None,region=None,parametersReport=True,**kwargs)

        =============================================================================================================================================
        Method for assigning an estimate to parameters. An estimate can also be defined for the parameters covariance matrix and the coverage region.
        =============================================================================================================================================

        - Parameters
        ------------

        estimative : list
            list with the estimation of parameters
        variance : array, ndmin=2
            covariance matrix of the parameters
        region : list
            list containing lists with the parameters belonging to the coverage region
        parametersReport : bool
            informs whether the parameters report should be created.

        - Kwargs
        --------

        limite_superior : list
            upper bound of the paramaters
        limite_inferior : list
            lower_bound of the parameters
        args : dict
            extra arguments to be passed to the model.

        - Notes
        -------

        -Inclusion of parameter estimation: will replace the optimization method. You will need to execute the uncertaintyParameter method.

        -Inclusion of parameter estimation and variance: will replace the optimization method and a part of the uncertainty method.
        For objective Function Mapping the region by the likelihood method, the uncertaintyParameter method must be performed (will override the uncertainty inseparated).

        -Inclusion of parameter estimation, variance and region: will replace optimization and uncertaintyParameter method.

        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('SETparametro')
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        # If there is no estimation data -> error
        if not self.__flag.info['dadosestimacao']:
            raise SyntaxError('It is necessary to add estimation data.')

        # SETparameter cannot run in conjunction with the optimize method.
        if self.__controleFluxo.otimizacao:
            raise SyntaxError('The SETparameter method cannot be executed with optimize method')

        # ---------------------------------------------------------------------
        # EXTRA ARGUMENTS TO BE PASSED TO THE MODEL
        # ---------------------------------------------------------------------
        # Obtaining args_user
        if kwargs.get('args') is not None:
            self.__args_user = kwargs.pop('args')

        # ---------------------------------------------------------------------
        # ATTRIBUTION TO QUANTITIES
        # ---------------------------------------------------------------------
        #Assigning the values to the estimation of the parameters and their
        # covariance matrix
        self.parametros._SETparametro(estimative, variance, region, **kwargs)

        # ---------------------------------------------------------------------
        # MODEL EVALUATION
        # ---------------------------------------------------------------------
        # Evaluation of the model at the optimal point informed
        try:
            aux = self.__excModel(self.parametros.estimativa,self._values)
        except Exception as erro:
            raise SyntaxError(u'Error in the model when evaluated in the informed parameters estimative. Error identified: "{}"'.format(erro))

        # ---------------------------------------------------------------------
        # OBTAINING THE OPTIMAL POINT
        # ---------------------------------------------------------------------

        self.FOotimo = float(self._excObjectiveFunction(self.parametros.estimativa, self._values))

        # ---------------------------------------------------------------------
        # INTERNAL VARIABLES
        # ---------------------------------------------------------------------

        # If variance is defined, it is assumed that the parametersUncertainty method.
        # has been executed, even if the inclusion of scope is optional.
        if variance is not None:
            self.__controleFluxo.SET_ETAPA('incertezaParametros')

        # If the region is defined, it is assumed that the regiaoAbrangencia method has been executed.
        if region is not None:
            self.__controleFluxo.SET_ETAPA('regiaoAbrangencia', ignoreValidacao=True)

        # Parameters report creation
        if parametersReport is True:
            self._out.Parametros(self.parametros, self.FOotimo)

    def parametersUncertainty(self,uncertaintyMethod ='Geral', parametersReport = True, objectiveFunctionMapping=True, **kwargs):
        u"""
        parametersUncertainty(self,uncertaintyMethod ='Geral', parametersReport = True, objectiveFunctionMapping=True, **kwargs)

        ===================================================================================
        Method to evaluate the covariance matrix of the parameters and the coverage region.
        ===================================================================================

        - Parameters
        ------------

        uncertaintyMethod : string
            method for calculating the covariance matrix of the parameters.
            available methods: 2InvHessian, Geral, SensibilidadeModelo
        parametersReport : bool
            informs whether the parameters report should be created.
        objectivefunctionMapping : bool
            Indicates whether the algorithm to map the coverage region should be executed

        - kwargs
        --------

        See documentation of self.__objectiveFunctionMapping

        - Notes
        -------

        -Before performing this method it is necessary to perform one of the following methods: (i) optimize or (ii) SETparameter

        -The coverage region is only executed if there is optimization history and the attribute regiao_abrangencia
        is not defined for the parameters.

        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('incertezaParametros')
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------         

        if uncertaintyMethod not in self.__metodosIncerteza:
            raise NameError('The method requested to calculate the uncertainty of the parameters {}'.format(uncertaintyMethod)
                            + ' is not available. Available methods ' + ', '.join(self.__metodosIncerteza) + '.')

        if not isinstance(objectiveFunctionMapping, bool):
            raise TypeError('The argument objectiveFunctionMapping must be boolean (True ou False).')

        # ---------------------------------------------------------------------
        # COVARIANCE MATRIX OF THE PARAMETERS
        # ---------------------------------------------------------------------

        # Evaluation of the auxiliary matrices
        # Hessian matrix of the objective function
        # Only evaluated if the chosen method is 2InvHess or Geral
        if uncertaintyMethod == self.__metodosIncerteza[0] or uncertaintyMethod == self.__metodosIncerteza[1]:
            self.__Hessiana_FO_Param()

            # Inverse of the Hessian matrix of the objective function in relation to the parameters
            invHess = inv(self.Hessiana)

        # Gy: second partial derivatives of the objective function in relation to parameters and experimental data
        # Only evaluated if the chosen method is: Geral
        if uncertaintyMethod == self.__metodosIncerteza[1]:
            self.__Matriz_Gy()

        # Model sensitivity matrix relative to parameters
        # Only evaluated if the method is: simplificado
        if uncertaintyMethod == self.__metodosIncerteza[2]:
            self.__Matriz_S()

        # ---------------------------------------------------------------------
        # ASSESSMENT OF THE UNCERTAINTY OF THE PARAMETERS
        # ---------------------------------------------------------------------

        # COVARIANCE MATRIX
        # Method: 2InvHessiana ->  2*inv(Hess)
        if uncertaintyMethod == self.__metodosIncerteza[0]:
            matriz_covariancia = 2*invHess

        # Method: geral - > inv(H)*Gy*Uyy*GyT*inv(H)
        elif uncertaintyMethod == self.__metodosIncerteza[1]:
            matriz_covariancia  = invHess.dot(self.Gy).dot(self.y.estimacao.matriz_covariancia).dot(self.Gy.transpose()).dot(invHess)

        # Method: simplificado -> inv(trans(S)*inv(Uyy)*S)
        elif uncertaintyMethod == self.__metodosIncerteza[2]:
            matriz_covariancia = inv(self.S.transpose().dot(inv(self.y.estimacao.matriz_covariancia)).dot(self.S))

        # ---------------------------------------------------------------------
        # ATTRIBUTION TO THE QUANTITIES
        # ---------------------------------------------------------------------
        self.parametros._updateParametro(matriz_covariancia=matriz_covariancia)

        # ---------------------------------------------------------------------
        # COVERAGE REGION
        # ---------------------------------------------------------------------
        # MAPPING OF OBJECTIVE FUNCTION:
        if objectiveFunctionMapping and self.parametros.NV != 1:
            self.__objectiveFunctionMapping(**kwargs)
            self.__flag.ToggleActive('mapeamentoFO')

        # The coverage region is only executed if there is a history of positions and fitness
        if self.__controleFluxo.mapeamentoFO and self.parametros.NV != 1:
            # OBTAINING THE REGION:
            regiao = self.regiaoAbrangencia()
            # ATTRIBUTION TO THE QUANTITY
            self.parametros._updateParametro(regiao_abrangencia=regiao)

        # parameters report creation
        if parametersReport is True:
            self._out.Parametros(self.parametros,self.FOotimo)

    def prediction(self,predictionReport = True, **kwargs):
        u"""
        prediction(self,predictionReport = True, **kwargs)

        ==============================
        Performs the model prediction.
        ==============================

        - Parameters
        ------------

        predictionReport : bool, optional
            informs whether the prediction report should be created. If is true the prediction report is created without statistical tests.\
            The statistical tests could be included in the 'residualAnalysis' method.

        - Keywords
        -----------

        See documentation of Relatorio.Predicao.

        - Notes
        ----------

        Before executing the prediction method it's necessary to execute the optimize and parametersUncertainty methods./
        Other option is to include the parameters value and the parameters uncertainty through the SETparameter method.
        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('predicao')

        # ---------------------------------------------------------------------
        # EVALUATION OF AUXILIARY MATRICES
        # ---------------------------------------------------------------------

        # Hessian matrix of the objective function
        # Only revaluated if the method that evaluates it has not been performed AND has no validation data
        if not self.__controleFluxo.Hessiana and not self.__flag.info['dadospredicao']:
            self.__Hessiana_FO_Param()

        # The inverse of the Hessian matrix of the objective function
        # Only revaluated if the method that evaluates it has not been performed AND has no validation data
        if self.__controleFluxo.Hessiana:
            invHess = inv(self.Hessiana)

        # Gy: partial second derivatives of the objective function concerning the parameters and experimental data
        # Only revaluated if the method that evaluates it has not been performed AND has no validation data
        if not self.__controleFluxo.Gy and not self.__flag.info['dadospredicao']:
            self.__Matriz_Gy()

        # S: Matrix of the sensitivity of the model concerning the parameters
        # Only revaluated if the method that evaluates it has not been performed AND has no validation data
        if not self.__controleFluxo.S or self.__flag.info['dadospredicao']:
            self.__Matriz_S()

        # ---------------------------------------------------------------------
        # PREDICTION
        # ---------------------------------------------------------------------
        aux = array(self.__excModel(self.parametros.estimativa,self._values))

        # ---------------------------------------------------------------------
        # PREDICTION EVALUATION (Y CALCULATED BY THE MODEL)
        # ---------------------------------------------------------------------    
        # COVARIANCE MATRIX OF Y
        # If the validation data are different from the experimental data, the covariance between the parameters
        # and experimental data will be disregarded.

        if not self.__controleFluxo.incertezaParametros:

            Uyycalculado = None

        else:

            if self.__flag.info['dadospredicao']:

                Uyycalculado = self.S.dot(self.parametros.matriz_covariancia).dot(self.S.transpose()) + self.y.predicao.matriz_covariancia

            else:
                # In this case, the validation data are the experimental data and the covariance between the parameters
                # and the experimental data will be considered.
                # COVARIANCE BETWEEN PARAMETERS AND EXPERIMENTAL DATA
                Covar_param_y_experimental = -inv(self.Hessiana).dot(self.Gy).dot(self.y.predicao.matriz_covariancia)
                # FIRST PART
                Uyycalculado_1 = self.S.dot(self.parametros.matriz_covariancia).dot(self.S.transpose())
                # SECOND PART
                Uyycalculado_2 = self.S.dot(Covar_param_y_experimental)
                # THIRD PART
                Uyycalculado_3 = Covar_param_y_experimental.transpose().dot(self.S.transpose())
                # COVARIANCE MATRIX OF Y
                Uyycalculado   = Uyycalculado_1 + Uyycalculado_2 + Uyycalculado_3 + self.y.estimacao.matriz_covariancia

        # --------------------------------------------------------------------
        # ASSIGNMENT OF VALUES TO QUANTITIES
        # -------------------------------------------------------------------
        self.y._SETcalculado(estimativa=aux,matriz_covariancia=Uyycalculado,
                             gL=[[self.y.estimacao.NE*self.y.NV-self.parametros.NV]*self.y.predicao.NE]*self.y.NV,
                             NE=self.y.predicao.NE)
        self.x._SETcalculado(estimativa=self.x.predicao.matriz_estimativa,matriz_covariancia=self.x.predicao.matriz_covariancia,
                             gL=[[self.x.estimacao.NE*self.x.NV-self.parametros.NV]*self.x.predicao.NE]*self.x.NV,
                             NE=self.x.predicao.NE)

        # prediction report creation
        if predictionReport is True:
            self._out.Predicao(self.x, self.y, None, **kwargs)


    def __Matriz_Sx(self,delta=1e-5):
        u"""
        Método para calcular a matriz Sx(derivadas primeiras da função do modelo em relação as grandezas de entrada x).

        Método de derivada central de primeira ordem em relação aos parâmetros(considera os parâmetros como variáveis do modelo).

        ========
        Entradas
        ========

        * delta(float): valor do incremento relativo para o cálculo da derivada. Incremento relativo à ordem de grandeza do parâmetro.

        =====
        Saída
        =====

        Retorna a matriz Sx(array).
        """
        pass

    def __objectiveFunctionMapping(self,**kwargs):
        u"""
        __objectiveFunctionMapping(self,**kwargs)

        ===============================================
         Performs the mapping of the objective function
        ===============================================

        - kwargs
        --------

        MethodObjectivefunctionmapping : string
            Method used to perform the mapping of the objective function
        iterations : int, > 0
            Defines the number of iterations used in the monte carlo method
        upper_bound : list
            Lower bound of the parameters
        lower_bound : list
            Upper bound of the parameters
        symmetryFactorLimit : list
            This variable is used to generate more points to fill the coverage region. It is recommended that the
            greater the interval of the region, the greater the limits of the symmetrical factor.

            symmetryFactorLimit[0] -> lower bound of the symmetry factor

            symmetryFactorLimit[1] -> upper bound of the symmetry factor

            symmetryFactorLimit[2] -> number of points generated from the symmetric factor
        searchLimitFactor : float, > 0

        distribution : string
            Type of distribution used to generate random parameters in the monte carlo method

        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('mapeamentoFO')

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------

        if kwargs.get('MethodObjectivefunctionmapping') is not None:
            tipo = kwargs.pop('MethodObjectivefunctionmapping')
        else:
            tipo = 'MonteCarlo'

        # evaluating whether the objective function mapping type is available
        if tipo not in self.__tipoObjectiveFunctionMapping:
            raise NameError('O método solicitado para mapeamento da função objetivo {}'.format(
                tipo) + ' não está disponível. Métodos disponíveis ' + ', '.join(self.__tipoObjectiveFunctionMapping) + '.')

        # if MethodObjectivefunctionmapping = 'MonteCarlo':
        if tipo == self.__tipoObjectiveFunctionMapping[0]:
            kwargsdisponiveis = ('iterations', 'upper_bound', 'lower_bound', 'searchLimitFactor', 'distribution', 'symmetryFactorLimit')

            # evaluating whether keywords are available
            if not set(kwargs.keys()).issubset(kwargsdisponiveis):
                raise NameError('Error in the keywords typed. keywords available: ' +
                                ', '.join(kwargsdisponiveis) + '.')
            # evaluating the iterations number-> must be greater than 1
            if kwargs.get(kwargsdisponiveis[0]) is not None:
                if kwargs.get(kwargsdisponiveis[0]) < 1:
                    raise ValueError('The number of iterations must be integer and positive.')
            # evaluating the search limit factor -> must be positive
            if kwargs.get(kwargsdisponiveis[3]) is not None:
                if kwargs.get(kwargsdisponiveis[3]) < 0:
                    raise ValueError('The search limit factor must be positive.')
            # evaluating the distribution
            if kwargs.get(kwargsdisponiveis[4]) is not None:
                if kwargs.get(kwargsdisponiveis[4]) not in ['uniform', 'triangular']:
                    raise ValueError('The distributions available for the MonteCarlo Method are: {}.'.format(['uniform','triangular']))
            # evaluating the symmetry Factor Limit
            if kwargs.get(kwargsdisponiveis[5]) is not None:
                if not isinstance(kwargs.get(kwargsdisponiveis[5]), list):
                    raise TypeError('The symmetry factor limit must be a list')
                if isinstance(kwargs.get(kwargsdisponiveis[5]), list) and len(kwargs.get(kwargsdisponiveis[5])) != 3:
                    raise ValueError('The size of symmetry factor limit must be equal to trhee. See documentation of objectiveFunctionMapping method')

        # ---------------------------------------------------------------------
        # Search limit
        # ---------------------------------------------------------------------
        upper_bound = kwargs.get('upper_bound')
        lower_bound = kwargs.get('lower_bound')
        searchLimitFactor = kwargs.get('searchLimitFactor') if kwargs.get('searchLimitFactor') is not None else 1/10.

        #Validation
        if (((not isinstance(upper_bound, list) and not isinstance(upper_bound, tuple)) and upper_bound is not None) or (((not isinstance(lower_bound, list)) and (not isinstance(lower_bound, tuple))) and lower_bound is not None)):
            raise TypeError('The upper_limit and the lower_limit must be lists or tuples.')
        if (upper_bound is not None and len(upper_bound) != self.parametros.NV) or (lower_bound is not None and len(lower_bound) != self.parametros.NV):
            raise TypeError('Upper_limits and lower_limits must be lists or tuples of the same size as self.paramtros.NV')

        if upper_bound is None or lower_bound is None:
            extremo_elipse_superior = [0 for i in range(self.parametros.NV)]
            extremo_elipse_inferior = [0 for i in range(self.parametros.NV)]

            fisher, FOcomparacao = self.__criteriosAbrangencia()

            Combinacoes = int(factorial(self.parametros.NV) / (factorial(self.parametros.NV - 2) * factorial(2)))
            p1 = 0
            p2 = 1
            cont = 0
            passo = 1

            for pos in range(Combinacoes):
                if pos == (self.parametros.NV - 1) + cont:
                    p1 += 1
                    p2 = p1 + 1
                    passo += 1
                    cont += self.parametros.NV - passo

                cov = array([[self.parametros.matriz_covariancia[p1, p1], self.parametros.matriz_covariancia[p1, p2]],
                             [self.parametros.matriz_covariancia[p2, p1], self.parametros.matriz_covariancia[p2, p2]]])

                coordenadas_x, coordenadas_y, width, height, theta = eval_cov_ellipse(cov, [self.parametros.estimativa[p1],
                                                                      self.parametros.estimativa[p2]],
                                                                FOcomparacao, ax=False)

                extremo_elipse_superior[p1] = nanmax(coordenadas_x)
                extremo_elipse_superior[p2] = nanmax(coordenadas_y)
                extremo_elipse_inferior[p1] = nanmin(coordenadas_x)
                extremo_elipse_inferior[p2] = nanmin(coordenadas_y)
                p2+=1

        if upper_bound is None:
            upper_bound = [extremo_elipse_superior[i] + (extremo_elipse_superior[i]-extremo_elipse_inferior[i])*searchLimitFactor for i in range(self.parametros.NV)]
        else:
            kwargs.pop('upper_bound') # removes the upper_bound from the extra arguments

        if lower_bound is None:
            lower_bound = [extremo_elipse_inferior[i] - (extremo_elipse_superior[i]-extremo_elipse_inferior[i])*searchLimitFactor for i in range(self.parametros.NV)]
        else:
            kwargs.pop('lower_bound') # removes the lower_bound from the extra arguments

        # Validating limits
        # Checks if upper bound is greater than lower bound
        test_bounds = [0]*self.parametros.NV
        for i in arange(self.parametros.NV):
            if (lower_bound[i]>upper_bound[i]) or (lower_bound[i]>self.parametros.estimativa[i] or self.parametros.estimativa[i]>upper_bound[i]):
                test_bounds[i] = 1

        index_test_bounds = [i for i, ele in enumerate(test_bounds) if ele]

        if any(test_bounds):
            raise TypeError(('The parameter estimate of '+'{} '*len(index_test_bounds)+' ​​must be between the lower_limit and the upper_limit. Parameter estimate: {}').format(*[self.parametros.simbolos[i] for i in index_test_bounds],self.parametros.estimativa))

        # ---------------------------------------------------------------------
        # MONTE CARLO METHOD
        # ---------------------------------------------------------------------
        if tipo == self.__tipoObjectiveFunctionMapping[0]:
            iterations = int(kwargs.get('iterations') if kwargs.get('iterations') is not None else 500)

            for cont in range(iterations):

                # samples generated with uniform distribution
                amostra_total_uni = [uniform(lower_bound[i], upper_bound[i], 1)[0] for i in range(self.parametros.NV)]

                # samples generated with triangular distribution, considering the whole area of the Cartesian plane
                amostra_total = [triangular(lower_bound[i], self.parametros.estimativa[i], upper_bound[i], 1)[0] for i in range(self.parametros.NV)]

                # samples generated with triangular distribution, considering the third quadrant of the Cartesian plane
                amostra_inf = [triangular(lower_bound[i], (lower_bound[i]+self.parametros.estimativa[i])/2, self.parametros.estimativa[i], 1)[0] for i in range(self.parametros.NV)]

                # samples generated with triangular distribution, considering the first quadrant of the Cartesian plane
                amostra_sup = [triangular(self.parametros.estimativa[i], (upper_bound[i] + self.parametros.estimativa[i]) / 2, upper_bound[i], 1)[0] for i in range(self.parametros.NV)]

                # Symmetry factor
                # It is applied to symmetricals to generate more points
                SF_limits = kwargs.get('symmetryFactorLimit') if kwargs.get('symmetryFactorLimit') is not None else [-2,2,50]
                SF = linspace(SF_limits[0], SF_limits[1], SF_limits[2], endpoint=True)

                # Calculating the symmetrical points
                amostras_simetricas = []
                for i in range(self.parametros.NV-1):
                    for factor in SF:

                        # Third quadrant (inferior)
                        # Symmetry with respect to the y axis
                        simetrica_y_inf = [None] * self.parametros.NV
                        simetrica_y_inf[i] = self.parametros.estimativa[i] + factor * abs(self.parametros.estimativa[i] - amostra_inf[i])
                        if simetrica_y_inf[i] > upper_bound[i]:
                            simetrica_y_inf[i] = upper_bound[i]
                        elif simetrica_y_inf[i] < lower_bound[i]:
                            simetrica_y_inf[i] = lower_bound[i]

                        # Symmetry with respect to the x axis
                        simetrica_x_inf = [None] * self.parametros.NV
                        simetrica_x_inf[i + 1] = self.parametros.estimativa[i + 1] + factor * abs(self.parametros.estimativa[i + 1] - amostra_inf[i + 1])
                        if simetrica_x_inf[i+1] > upper_bound[i+1]:
                            simetrica_x_inf[i+1] = upper_bound[i+1]
                        elif simetrica_x_inf[i+1] < lower_bound[i+1]:
                            simetrica_x_inf[i+1] = lower_bound[i+1]

                        # Symmetry with respect to the origin
                        simetrica_o_inf = [None] * self.parametros.NV
                        simetrica_o_inf[i] = simetrica_y_inf[i]
                        simetrica_o_inf[i + 1] = simetrica_x_inf[i+1]

                        # First quadrant (superior)
                        # Symmetry with respect to the y axis
                        simetrica_y_sup = [None] * self.parametros.NV
                        simetrica_y_sup[i] = self.parametros.estimativa[i] - factor * abs(self.parametros.estimativa[i] - amostra_sup[i])
                        if simetrica_y_sup[i] > upper_bound[i]:
                            simetrica_y_sup[i] = upper_bound[i]
                        elif simetrica_y_sup[i] < lower_bound[i]:
                            simetrica_y_sup[i] = lower_bound[i]

                        # Symmetry with respect to the x axis
                        simetrica_x_sup = [None] * self.parametros.NV
                        simetrica_x_sup[i + 1] = self.parametros.estimativa[i + 1] - factor * abs(self.parametros.estimativa[i + 1] - amostra_sup[i + 1])
                        if simetrica_x_sup[i + 1] > upper_bound[i + 1]:
                            simetrica_x_sup[i + 1] = upper_bound[i + 1]
                        elif simetrica_x_sup[i + 1] < lower_bound[i + 1]:
                            simetrica_x_sup[i + 1] = lower_bound[i + 1]

                        # Symmetry with respect to the origin
                        simetrica_o_sup= [None] * self.parametros.NV
                        simetrica_o_sup[i] = simetrica_y_sup[i] # O simétrico em relação ao eixo y corresponde ao x do par ordenado
                        simetrica_o_sup[i + 1] = simetrica_x_sup[i+1]

                        # Completing the list with the parameters that remained constant for each symmetry
                        simetricos_inf = [simetrica_x_inf, simetrica_y_inf, simetrica_o_inf]
                        for sim in simetricos_inf:
                            for j in range(self.parametros.NV):
                                if sim[j] is None:
                                    sim[j] = amostra_inf[j]

                        simetricos_sup = [simetrica_x_sup, simetrica_y_sup, simetrica_o_sup]
                        for sim in simetricos_sup:
                            for j in range(self.parametros.NV):
                                if sim[j] is None:
                                    sim[j] = amostra_sup[j]

                        # Adding the symmetrical points to a list
                        amostras_simetricas.append(simetrica_x_inf)
                        amostras_simetricas.append(simetrica_y_inf)
                        amostras_simetricas.append(simetrica_o_inf)
                        amostras_simetricas.append(simetrica_x_sup)
                        amostras_simetricas.append(simetrica_y_sup)
                        amostras_simetricas.append(simetrica_o_sup)

                amostra = [amostra_total, amostra_inf, amostra_sup, amostra_total_uni, *amostras_simetricas]

                FO = [float(self._excObjectiveFunction(amo_i, self._values)) for amo_i in amostra] #self._excFO returns a DM object, it's necessary convert to float object

                for i,FO_i in enumerate(FO):
                    self.__decisonVariablesMapped.append(amostra[i])
                    self.__OFMapped.append(FO_i)

    def __criteriosAbrangencia(self):
        u"""
         __criteriosAbrangencia(self)

        =======================================================================================================
         Returns the values of the Fisher and chi2 distributions and the limit value of the objective function.
        =======================================================================================================

            - Notes
            ----------
            Used to evaluate the coverage region.
        """

        # F test = F(PA,NP,NE*NY-NP)
        fisher = f.ppf(self.PA,self.parametros.NV,(self.y.estimacao.NE*self.y.NV-self.parametros.NV))

        # Value for the coverage ellipse:
        ellipseComparacao = self.FOotimo*(float(self.parametros.NV)/(self.y.estimacao.NE*self.y.NV-float(self.parametros.NV))*fisher)

        return fisher, ellipseComparacao

    def regiaoAbrangencia(self):
        u"""
        regiaoAbrangencia(self)

        ==============================================================================================
        Method to evaluate the coverage region by Fisher's criteria, known as likelihood region [1]
        ==============================================================================================

             - References
             ------------
             [1] SCHWAAB, M. et al. Nonlinear parameter estimation through particle swarm optimization. Chemical Engineering Science, v. 63, n. 6, p. 1542–1552, mar. 2008.

        ==========
        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('regiaoAbrangencia')

        # ---------------------------------------------------------------------
        # DETERMINATION OF THE COVERAGE REGION BY THE FISHER CRITERIA
        # ---------------------------------------------------------------------
        fisher, ellipseComparacao = self.__criteriosAbrangencia()

        # Comparison of the objective function value evaluated in the optimization step with the OFMapped variable.
        # If they are smaller, the respective parameters will be contained in the coverage region.
        regiao = []
        for pos,OFMapped in enumerate(self.__OFMapped):
            if OFMapped <= ellipseComparacao+self.FOotimo:
                regiao.append(self.__decisonVariablesMapped[pos])

        # -------------------------------------------------------------------
        # ASSESSING WHETHER POINTS WERE OBTAINED TO FILL THE COVERAGE REGION
        # -------------------------------------------------------------------
        if regiao == []:
            warn('The coverage region evaluated by the likelihood method contains no points. Review the parameters of the algorithm used.',UserWarning)

        return regiao

    def residualAnalysis(self, report=True, **kwargs):
        u"""
        residualAnalysis(self, report=True, **kwargs)

        ========================================
         Method to perform the residual analysis
        ========================================

        - Parameters
        ------------

        report : bool, optional
        informs whether the The statistical tests results should be included in the the prediction report.

        - Keywords
        -----------

        See documentation of Relatorio.Predicao.

        - Notes
        ----------

        -Preferably, the analysis is performed with the validation data.

        -The statistical tests are applied for x and y quantities.

        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('analiseResiduos')
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------         

        # Size of the vectors:
        if self.y.predicao.NE != self.y.calculado.NE:
            raise TypeError(u'The length of the validation and calculated vectors are not consistent. Evaluate the need to perform the prediction method.')
        # ---------------------------------------------------------------------
        # RESIDUES CALCULATION
        # ---------------------------------------------------------------------          
        # Residues calculation (or deviations) - are based on the validation data
        residuo_y = self.y.predicao.matriz_estimativa - self.y.calculado.matriz_estimativa
        residuo_x = self.x.predicao.matriz_estimativa - self.x.calculado.matriz_estimativa

        # ---------------------------------------------------------------------
        # ATTRIBUTION TO QUANTITIES
        # ---------------------------------------------------------------------       
        # Attribution of values on objects
        self.x._SETresiduos(estimativa=residuo_x)
        self.y._SETresiduos(estimativa=residuo_y)

        # ---------------------------------------------------------------------
        # R2 and R2 adjusted calculation
        # ---------------------------------------------------------------------   
        self.estatisticas = {'R2': {}, 'R2ajustado': {}, 'FuncaoObjetivo': {}}
        # For y:
        for i,symb in enumerate(self.y.simbolos):
            SSE = sum(self.y.residuos.matriz_estimativa[:,i]**2)
            SST = sum((self.y.predicao.matriz_estimativa[:,i]-\
                  mean(self.y.predicao.matriz_estimativa[:,i]))**2)
            self.estatisticas['R2'][symb]         = 1 - SSE/SST
            self.estatisticas['R2ajustado'][symb] = 1 - (SSE/(self.y.predicao.NE-self.parametros.NV))\
                                       /(SST/(self.y.predicao.NE - 1))
        # For x:
        for i,symb in enumerate(self.x.simbolos):
            if self.__flag.info['reconciliacao']:
                SSEx = sum(self.x.residuos.matriz_estimativa[:,i]**2)
                SSTx = sum((self.x.predicao.matriz_estimativa[:,i]-\
                      mean(self.x.predicao.matriz_estimativa[:,i]))**2)
                self.estatisticas['R2'][symb]         = 1 - SSEx/SSTx
                self.estatisticas['R2ajustado'][symb] = 1 - (SSEx/(self.x.predicao.NE-self.parametros.NV))\
                                           /(SSTx/(self.x.predicao.NE - 1))
            else:
                self.estatisticas['R2'][symb]         = None
                self.estatisticas['R2ajustado'][symb] = None

        # ---------------------------------------------------------------------
        # EXECUTION OF STATISTICAL TESTS
        # ---------------------------------------------------------------------             
        # Independent quantities
        if self.__flag.info['reconciliacao']:
            self.x._testesEstatisticos(self.y.predicao.matriz_estimativa)

        # Dependent quantities
        self.y._testesEstatisticos(self.x.predicao.matriz_estimativa)

        # -----------------------------------------------------------------
        # VALIDATION OF THE VALUE OF THE OBJECTIVE FUNCTION AS A CHI-SQUARE
        # -----------------------------------------------------------------
        # TODO: substituir pelo grau de liberdade dos parâmetros, após merge com IncertezaParametros
        gL = self.y.estimacao.NE*self.y.NV - self.parametros.NV

        chi2max = chi2.ppf(self.PA+(1-self.PA)/2,gL)
        chi2min = chi2.ppf((1-self.PA)/2,gL)

        self.estatisticas['FuncaoObjetivo'] = {'chi2max':chi2max, 'chi2min':chi2min, 'FO':self.FOotimo}

        # prediction report creation
        if report is True:
            kwargs['PA'] = self.PA
            self._out.Predicao(self.x, self.y, self.estatisticas, **kwargs)

    def plots(self,**kwargs):
        u"""
        plots(self,**kwargs)

        ======================================
        Routines for creating and saving plots
        ======================================

        **- kwargs**
        ------------

        **types : list**
            It informs which plots should be created. See, below, "Available plots" .

        **- Notes**
        ------------

        After using the setDados method, the plots method can be used anywhere in the code.
        However, it will create the plots according to the information already obtained

        Available plots:
            'regiaoAbrangencia': plots the coverage region of the parameters

            'grandezas-entrada': plots for input and validation data

            'predicao": plots for the prediction results

            'grandezas-calculadas': plots for the calculated values of each quantity
            
            'analiseResiduos': plots for the residual analysis

        """
        if kwargs.get('types') is None:
            types = []
            for fl_key in self.__graph_flux_association.keys():
                if getattr(self.__controleFluxo, fl_key):
                    types.extend(self.__graph_flux_association[fl_key])

        # Initialization of the Figure that will contain the graphs -> object
        Fig = Grafico(dpi=300)

        # ---------------------------------------------------------------------
        # BASE PATH
        # ---------------------------------------------------------------------         
        base_path = self.__base_path + sep + self._configFolder['plots'] + sep

        # ---------------------------------------------------------------------
        # PLOTS
        # ---------------------------------------------------------------------
        if (self.__tipoGraficos[1] in types):
            # if setDados method was executed at any time:
            if self.__controleFluxo.setDados:
                base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[1])] + sep
                Validacao_Diretorio(base_path,base_dir)
                # Internal folders
                # ------------------------------------------------------------------------------------
                folder = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[1])] + sep + self._configFolder['plots-subfolder-DadosEstimacao']+ sep + self._configFolder['plots-subfolder-grandezatendencia']+sep
                Validacao_Diretorio(base_path, folder)
                # -----------------------------------------------------------------------------------
                # created plots for the experimental data
                if self.__flag.info['dadosestimacao'] == True:
                    self.x.Graficos(base_path, base_dir, ID=['estimacao'], fluxo=0, Fig=Fig)
                    self.y.Graficos(base_path, base_dir, ID=['estimacao'], fluxo=0, Fig=Fig)

                    # Plots for y quantities by x quantities
                    for iy in range(self.y.NV):
                        for ix in range(self.x.NV):
                            # plots without uncertainty
                            Fig.grafico_dispersao_sem_incerteza(self.x.estimacao.matriz_estimativa[:,ix],
                                                                self.y.estimacao.matriz_estimativa[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                marker='o', linestyle='None')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_em_funcao_de_'+self.x.simbolos[ix]+'_sem_incerteza')
                            # plots with uncertainty
                            Fig.grafico_dispersao_com_incerteza(self.x.estimacao.matriz_estimativa[:,ix],
                                                                self.y.estimacao.matriz_estimativa[:,iy],
                                                                self.x.estimacao.matriz_incerteza[:,ix],
                                                                self.y.estimacao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                fator_abrangencia_x=[2.]*self.x.estimacao.NE,
                                                                fator_abrangencia_y=[2.]*self.y.estimacao.NE, fmt='o')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_em_funcao_de_'+' '+self.x.simbolos[ix]+'_com_incerteza')

                # If the validation data is different from the experimental data, graphics will be created for the validation data.
                if self.__flag.info['dadospredicao'] == True:
                    # Internal folders
                    # ------------------------------------------------------------------------------------
                    if self.__controleFluxo.FLUXO_ID == 0:
                        folder = self._configFolder['plots{}'.format(self.__tipoGraficos[5])] +  sep +self._configFolder['plots-subfolder-DadosEstimacao']+ sep+ self._configFolder['plots-subfolder-grandezatendencia']+sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(self.__controleFluxo.FLUXO_ID)+ sep+ self._configFolder['plots-subfolder-grandezatendencia']+sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    self.x.Graficos(base_path, base_dir, ID=['predicao'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)
                    self.y.Graficos(base_path, base_dir, ID=['predicao'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

                    # Plots for y quantities by x quantities
                    for iy in range(self.y.NV):
                        for ix in range(self.x.NV):
                            # plots without uncertainty
                            Fig.grafico_dispersao_sem_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                marker='o', linestyle='None')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_sem_incerteza')
                            # plots with uncertainty
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.predicao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                fator_abrangencia_x=[2.]*self.x.predicao.NE,
                                                                fator_abrangencia_y=[2.]*self.y.predicao.NE, fmt= 'o')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')
            else:
                warn('The input graphs could not be created because the setConjunto method was not executed.',UserWarning)

        # created plots for the output data (calculated)
        # quantities-calculated
        if self.__tipoGraficos[3] in types:
            base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[3])] + sep
            Validacao_Diretorio(base_path, base_dir)

            # evaluates if the parametersUncertainty method was executed at any time
            if self.__controleFluxo.incertezaParametros:
                self.parametros.Graficos(base_path, base_dir, ID=['parametro'], fluxo=self.__controleFluxo.FLUXO_ID)
            else:
                warn('The graphs involving only calculated quantities (X and Y) could not be created because the parametersUncertainty method was not executed.',UserWarning)

            # evaluates if the prediction method was executed at any time
            if self.__controleFluxo.predicao:
                self.x.Graficos(base_path, base_dir, ID=['calculado'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)
                self.y.Graficos(base_path, base_dir, ID=['calculado'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

            else:
                warn('The graphs involving only the calculated quantities (X and Y) could not be created, because the prediction method was not executed.',UserWarning)

        # coverage region
        if self.__tipoGraficos[0] in types:
            # The plots of the coverage region will be created only if the covariance matrix of the parameters has been calculated.
            if self.__controleFluxo.incertezaParametros:
                # Estimation plots
                if self.parametros.NV >1:
                    base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[0])] + sep
                    Validacao_Diretorio(base_path, base_dir)
                # the plots can only be executed if the number of parameters is greater than 1
                if self.parametros.NV != 1:
                    # number of non-repeated combinations for the parameters
                    Combinacoes = int(factorial(self.parametros.NV)/(factorial(self.parametros.NV-2)*factorial(2)))
                    p1 = 0; p2 = 1; cont = 0; passo = 1 # inicialiação dos contadores (pi e p2 são indinces dos parâmetros
                    # passo: counts the number of evaluated parameters
                    # cont: compute (param.NV - step1)+(param.NV - step2)

                    for pos in range(Combinacoes):
                        if pos == (self.parametros.NV-1)+cont:
                            p1 +=1; p2 = p1+1; passo +=1
                            cont += self.parametros.NV-passo

                        # Plots the coverage region by likelihood method
                        if self.__controleFluxo.regiaoAbrangencia and self.parametros.regiao_abrangencia != []:
                            aux1 = [] # auxiliary list -> coverage region for the parameter P1
                            aux2 = [] # auxiliary list -> coverage region for the parameter P2
                            for it in range(int(size(self.parametros.regiao_abrangencia)/self.parametros.NV)):
                                aux1.append(self.parametros.regiao_abrangencia[it][p1])
                                aux2.append(self.parametros.regiao_abrangencia[it][p2])
                            Fig.grafico_dispersao_sem_incerteza(array(aux1), array(aux2),
                                                                add_legenda=True, corrigir_limites=False,
                                                                marker='o', linestyle='None', color='b', linewidth=2.0, zorder=1)
                        # Plots the coverage region by linearization (ellipse) method
                        fisher, ellipseComparacao = self.__criteriosAbrangencia()

                        cov = array([[self.parametros.matriz_covariancia[p1,p1], self.parametros.matriz_covariancia[p1,p2]],
                                     [self.parametros.matriz_covariancia[p2,p1], self.parametros.matriz_covariancia[p2,p2]]])

                        Fig.elipse_covariancia(cov,[self.parametros.estimativa[p1],self.parametros.estimativa[p2]],ellipseComparacao)

                        if self.__controleFluxo.regiaoAbrangencia and self.parametros.regiao_abrangencia != []:
                            Fig.set_legenda([u'Verossimilhança','Elipse'], loc='best')
                        else:
                            Fig.set_legenda(['Elipse'], loc='best')

                        Fig.set_label(self.parametros.labelGraficos()[p1], self.parametros.labelGraficos()[p2])

                        # SAVE THE PLOT
                        Fig.salvar_e_fechar(base_path+base_dir+'regiao_verossimilhanca'+'_'+
                                    str(self.parametros.simbolos[p1])+'_'+str(self.parametros.simbolos[p2])+'.png',
                                            config_axes=True)
                        p2+=1
                else:
                    warn('The coverage region graphs could not be created, because there is only one parameter.',UserWarning)

            else:
                warn('The coverage region graphs could not be created because the uncertaintyParameters method was not run OR in the SETparameter method the parameters variance was not defined',UserWarning)

        # prediction
        if self.__tipoGraficos[2] in types:
            # The execution of the prediction method is necessary for this flux
            if self.__controleFluxo.predicao:
                # Internal folders
                # ------------------------------------------------------------------------------------
                if self.__controleFluxo.FLUXO_ID == 0:
                    folderone = self._configFolder['plots-{}'.format(self.__tipoGraficos[2])] + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + 'Saida calculada em funcao das entradas observadas' + sep
                    Validacao_Diretorio(base_path, folderone)
                else:
                    folderone = self._configFolder['plots-{}'.format(self.__tipoGraficos[2])] + sep + self._configFolder['plots-subfolder-Dadosvalidacao'] + ' ' + str(self.__controleFluxo.FLUXO_ID) + sep+ 'Saida calculada em funcao das entradas observadas' + sep
                    Validacao_Diretorio(base_path, folderone)
                # ------------------------------------------------------------------------------------
                # ------------------------------------------------------------------------------------
                if self.__controleFluxo.FLUXO_ID == 0:
                    foldertwo = self._configFolder['plots-{}'.format(self.__tipoGraficos[2])] + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + 'Saida calculada em funcao das saidas observadas' + sep
                    Validacao_Diretorio(base_path, foldertwo)
                else:
                    foldertwo = self._configFolder['plots-{}'.format(self.__tipoGraficos[2])] + sep + self._configFolder['plots-subfolder-Dadosvalidacao'] + ' ' + str(self.__controleFluxo.FLUXO_ID) + sep + 'Saida calculada em funcao das saidas observadas' + sep
                    Validacao_Diretorio(base_path, foldertwo)
                # ------------------------------------------------------------------------------------
                # Plots for y quantities by y quantities
                for iy in range(self.y.NV):
                    for ix in range(self.x.NV):
                        # Plots without uncertainty
                        x_plot = self.x.estimacao.matriz_estimativa[:,ix] if self.__controleFluxo.FLUXO_ID==0 else self.x.predicao.matriz_estimativa[:,ix]
                        Fig.grafico_dispersao_sem_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                            self.y.calculado.matriz_estimativa[:,iy],
                                                            marker='o', linestyle='None', color = 'b',
                                                            config_axes=True,add_legenda=True)
                        Fig.grafico_dispersao_sem_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                            self.y.predicao.matriz_estimativa[:,iy],
                                                            label_x=self.x.labelGraficos()[ix],
                                                            label_y=self.y.labelGraficos()[iy],
                                                            marker='o', linestyle='None', color = 'r',
                                                            config_axes=True, add_legenda=True)
                        Fig.set_legenda(['calculado','observado'],loc='best', fontsize=12)
                        Fig.salvar_e_fechar(base_path+folderone+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_sem_incerteza')
                        #Fig.salvar_e_fechar(base_path+folderone+'calculado' +'_'+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_sem_incerteza')

                        # Plots with uncertainty
                        if self.y.calculado.matriz_correlacao is not None:
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.calculado.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.calculado.matriz_incerteza[:,iy],
                                                                fator_abrangencia_x=[2.]*self.x.predicao.NE,
                                                                fator_abrangencia_y=[2.]*self.y.calculado.NE, fmt='o',
                                                                color='b',add_legenda=True)
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.predicao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos()[ix],
                                                                label_y=self.y.labelGraficos()[iy],
                                                                fator_abrangencia_x=[2.]*self.x.predicao.NE,
                                                                fator_abrangencia_y=[2.]*self.y.predicao.NE, fmt='o',
                                                                color='r',add_legenda=True)
                            Fig.set_legenda(['calculado','observado'],loc='best', fontsize=12)
                            Fig.salvar_e_fechar(base_path+folderone+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')
                            #Fig.salvar_e_fechar(base_path+folderone+'calculado' +'_'+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')


                for iy in range(self.y.NV):
                    y  = self.y.predicao.matriz_estimativa[:,iy]
                    ym = self.y.calculado.matriz_estimativa[:,iy]
                    # Coverage factors for validation y and calculated
                    t_cal = [-t.ppf((1 - self.PA) / 2, self.y.calculado.gL[iy][j]) for j in range(self.y.calculado.NE)]
                    t_val = [-t.ppf((1 - self.PA) / 2, self.y.predicao.gL[iy][j]) for j in range(self.y.predicao.NE)]
                    amostras = arange(1,self.y.predicao.NE+1,1)

                    diagonal = linspace(min(y), max(y))
                    # Comparison between the experimental and calculated values by the model, without variance
                    Fig.grafico_dispersao_sem_incerteza(y, ym, marker='o', linestyle='None',
                                                        corrigir_limites=False, config_axes=False)
                    Fig.grafico_dispersao_sem_incerteza(diagonal, diagonal, linestyle='-', color='k', linewidth = 2.0,
                                                        corrigir_limites=True, config_axes=False)
                    # Set_label has the fontsize (font size on the X and Y axes) defined according to the value set in Plots.
                    Fig.set_label(self.y.labelGraficos('observado')[iy] \
                                  if self.__flag.info['dadospredicao'] else self.y.labelGraficos('observado')[iy],
                                  self.y.labelGraficos('calculado')[iy])


                    Fig.salvar_e_fechar((base_path+foldertwo+'observado' if self.__flag.info['dadospredicao'] else base_path+foldertwo+'observado')+'_' + str(self.y.simbolos[iy])+'_funcao_'+str(self.y.simbolos[iy])+'_calculado_sem_incerteza.png',config_axes=True)


                    # Comparison between the experimental and calculated values by the model, without variance,
                    # by samples
                    Fig.grafico_dispersao_sem_incerteza(amostras, y, marker='o', linestyle='None', color='b', add_legenda=True)
                    Fig.grafico_dispersao_sem_incerteza(amostras, ym, marker='o', linestyle='None', color='r',
                                                        corrigir_limites=False, config_axes=False, add_legenda=True)
                    Fig.set_label('Amostras', self.y.labelGraficos()[iy])
                    Fig.set_legenda(['dados para predicao' if self.__flag.info['dadospredicao'] else 'dados para estimacao','calculado'],
                                    fontsize=12, loc='best')
                    Fig.salvar_e_fechar(
                        (base_path + foldertwo +'observado' if self.__flag.info['dadospredicao'] else base_path+foldertwo+'observado') + \
                         '_' + str(self.y.simbolos[iy]) + \
                        '_funcao_amostras_calculado_sem_incerteza.png',
                        config_axes=True
                        )

                    # Comparison between the experimental and calculated values by the model, with variance
                    if self.y.calculado.matriz_incerteza is not None:
                        yerr_calculado = self.y.calculado.matriz_incerteza[:,iy]

                        yerr_validacao = self.y.predicao.matriz_incerteza[:,iy]

                        # Comparison between the experimental (validation) and calculated values by the model, without variance,
                        # by samples
                        Fig.grafico_dispersao_com_incerteza(amostras, y, None, yerr_validacao, fator_abrangencia_x=[2.]*len(amostras),
                                                            fator_abrangencia_y=t_val, fmt="o", color = 'b',
                                                            config_axes=False, corrigir_limites=False,
                                                            add_legenda=True)
                        Fig.grafico_dispersao_com_incerteza(amostras, ym, None, yerr_calculado,fator_abrangencia_x=[2.]*len(amostras),
                                                            fator_abrangencia_y=t_cal, fmt="o", color = 'r', config_axes=False, add_legenda=True)
                        Fig.set_label('Amostras', self.y.labelGraficos()[iy])
                        Fig.set_legenda(['dados para predicao' if self.__flag.info['dadospredicao'] else 'dados para estimacao', 'calculado'],fontsize=12, loc='best')
                        Fig.salvar_e_fechar((base_path+foldertwo+'observado' if self.__flag.info['dadospredicao'] else base_path + foldertwo+'observado') + '_' + str(self.y.simbolos[iy]) +'_funcao_amostras_calculado_com_incerteza.png', config_axes=True)

                        # calculated y by experimental y
                        Fig.grafico_dispersao_com_incerteza(y, ym, yerr_validacao, yerr_calculado,
                                                            fator_abrangencia_x=t_cal, fator_abrangencia_y=t_val,
                                                            fmt="o", corrigir_limites=True, config_axes=False)
                        Fig.grafico_dispersao_sem_incerteza(diagonal, diagonal, linestyle='-', color='k', linewidth=2.0,
                                                             corrigir_limites=False, config_axes=False)
                        Fig.set_label(self.y.labelGraficos('observado')[iy] \
                                      if self.__flag.info['dadospredicao'] else
                                      self.y.labelGraficos('observado')[iy],
                                      self.y.labelGraficos('calculado')[iy])
                        Fig.salvar_e_fechar((base_path+foldertwo+'observado' if self.__flag.info['dadospredicao'] else base_path+foldertwo+'observado' )+ \
                                              '_' + str(self.y.simbolos[iy]) + \
                                            '_funcao_' + str(self.y.simbolos[iy]) + '_calculado_com_incerteza.png',
                                            config_axes=True,
                                            reiniciar=(False if not self.__flag.info['dadospredicao'] else True))
                                            # If there is no validation data, a test based on the F test is applied

                        # Comparison between the experimental (validation) and calculated values by the model, with variance,
                        # by samples
                        # plots based on test F
                        if not self.__flag.info['dadospredicao']:
                            # test F plot
                            ycalc_inferior_F = []
                            ycalc_superior_F = []
                            for iNE in range(self.y.calculado.NE):

                                ycalc_inferior_F.append(self.y.calculado.matriz_estimativa[iNE,iy]+\
                                            t_val[iNE]\
                                            *(f.ppf((self.PA+(1-self.PA)/2),self.y.calculado.gL[iy][iNE],\
                                            self.y.predicao.gL[iy][iNE])*self.y.predicao.matriz_covariancia[iNE,iNE])**0.5)

                                ycalc_superior_F.append(self.y.calculado.matriz_estimativa[iNE,iy]-t_val[iNE]\
                                               *(f.ppf((self.PA+(1-self.PA)/2),self.y.calculado.gL[iy][iNE],\
                                            self.y.predicao.gL[iy][iNE])*self.y.predicao.matriz_covariancia[iNE,iNE])**0.5)

                            Fig.grafico_dispersao_sem_incerteza(y, array(ycalc_inferior_F),
                                                                color='r', corrigir_limites=False, config_axes=False)
                            Fig.grafico_dispersao_sem_incerteza(y, array(ycalc_superior_F), color='r',
                                                                corrigir_limites=True, config_axes=False, add_legenda=True)
                            Fig.set_legenda(['Limites baseados no teste F'], fontsize = 12, loc='best')
                            Fig.salvar_e_fechar(base_path + foldertwo + 'observado' + '_' + str(self.y.simbolos[iy]) + '_funcao_' + str(self.y.simbolos[iy]) + '_calculado_com_incerteza.png',
                                                config_axes=False)

            else:
                warn('The graphs involving the estimation (prediction) could not be created because the prediction method was not executed.',UserWarning)

        # Residual analysis
        if (self.__tipoGraficos[5] in types):
            # the residualAnalysis method must been executed
            if self.__controleFluxo.analiseResiduos:
                base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] + sep
                Validacao_Diretorio(base_path,base_dir)
                # Plots for the residues of the independent quantities (if the reconciliation was performed)
                if self.__flag.info['reconciliacao'] == True:
                    self.x.Graficos(base_path, base_dir, ID=['residuo'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

                # Plots for the residues of the dependent quantities
                self.y.Graficos(base_path, base_dir, ID=['residuo'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

                # Plots for the residues by validation (or experimental) data and calculated data
                for i,simb in enumerate(self.y.simbolos):
                    # Internal folders
                    # ------------------------------------------------------------------------------------
                    if self.__controleFluxo.FLUXO_ID == 0:
                        folder = self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] +  sep +self._configFolder['plots-subfolder-DadosEstimacao']+ sep + self.y.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(self.__controleFluxo.FLUXO_ID)+ sep + self.y.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    # Residues by y calculated
                    Fig.grafico_dispersao_sem_incerteza(array([min(self.y.calculado.matriz_estimativa[:, i]), max(self.y.calculado.matriz_estimativa[:, i])]),
                                                        array([mean(self.y.residuos.matriz_estimativa[:, i])] * 2),
                                                        linestyle='-.', color = 'r', linewidth = 2,
                                                        add_legenda=True, corrigir_limites=False, config_axes=False)
                    Fig.grafico_dispersao_sem_incerteza(self.y.calculado.matriz_estimativa[:,i], self.y.residuos.matriz_estimativa[:,i],
                                                        marker='o', linestyle = 'none',
                                                        label_x= self.y.labelGraficos()[i] + ' calculado',
                                                        label_y=u'Resíduos '+self.y.labelGraficos()[i])
                    Fig.set_legenda([u'Média resíduos ' + self.y.simbolos[i]], fontsize=12, loc='best')
                    Fig.axes.axhline(0, color='black', lw=1, zorder=1)
                    Fig.salvar_e_fechar(base_path+folder+'residuos'+'_funcao_'\
                                        +self.y.simbolos[i]+'_calculado.png')

                    # Residues by y validated
                    Fig.grafico_dispersao_sem_incerteza(array([min(self.y.predicao.matriz_estimativa[:, i]),
                                                               max(self.y.predicao.matriz_estimativa[:, i])]),
                                                        array([mean(self.y.residuos.matriz_estimativa[:, i])] * 2),
                                                        linestyle='-.', color='r', linewidth=2,
                                                        add_legenda=True, corrigir_limites=False, config_axes=False)
                    Fig.grafico_dispersao_sem_incerteza(self.y.predicao.matriz_estimativa[:, i],
                                                        self.y.residuos.matriz_estimativa[:, i],
                                                        marker='o', linestyle='none')
                    Fig.set_label(label_x=self.y.labelGraficos()[i]+' '+(u'validação' if self.__flag.info['dadospredicao'] else u'observado'),
                                  label_y=u'Resíduos ' + self.y.labelGraficos()[i])
                    Fig.set_legenda([u'Média resíduos ' + self.y.simbolos[i]], fontsize=12, loc='best')
                    Fig.axes.axhline(0, color='black', lw=1, zorder=1)
                    Fig.salvar_e_fechar(
                        base_path + folder + 'residuos_' + '_funcao_' +
                        self.y.simbolos[i] + '_' + ('observado' if self.__flag.info['dadospredicao'] else 'observado')+'.png')

                    for j, simbol in enumerate(self.x.simbolos):
                        # Residues by estimation/validation
                        if self.__flag.info['dadospredicao']:
                            x = self.x.predicao.matriz_estimativa[:,j]
                        else:
                            x = self.x.estimacao.matriz_estimativa[:,j]

                        Fig.grafico_dispersao_sem_incerteza(array([min(x), max(x)]),
                                                            array([mean(self.y.residuos.matriz_estimativa[:, i])] * 2),
                                                            linestyle='-.', color='r', linewidth=2,
                                                            add_legenda=True, corrigir_limites=False, config_axes=False)
                        Fig.grafico_dispersao_sem_incerteza(x, self.y.residuos.matriz_estimativa[:, i],
                                                        marker='o', linestyle='none')
                        Fig.set_label(label_x= self.x.labelGraficos()[j] +' '+ (u'observado' if self.__flag.info['dadospredicao'] else u'observado'),
                                  label_y=u'Resíduos ' + self.y.labelGraficos()[i])
                        Fig.set_legenda([u'Média resíduos ' + self.y.simbolos[i]], fontsize=12, loc='best')
                        Fig.axes.axhline(0, color='black', lw=1, zorder=1)
                        Fig.salvar_e_fechar(base_path+folder+'residuos'+ '_funcao_' \
                                            +self.x.simbolos[j]+'_'+ \
                                            ('observado' if self.__flag.info['dadospredicao'] else 'observado')+'.png')

            else:
                warn('Plots involving residue analysis could not be created because the residualAnalysis method was not executed.',UserWarning)

    def reports(self,**kwargs):
        u"""
        reports(self,**kwargs):

        ========================================================
        Method for creating the report(s) with the main results.
        ========================================================

        - kwargs
        --------

        See documentation of Relatorio.Predicao

        """

        # ---------------------------------------------------------------------
        # PARAMETERS REPORT
        # ---------------------------------------------------------------------
        # Creating the parameters report if the optimization method or SETparameter methods was executed.
        if self.__controleFluxo.otimizacao or self.__controleFluxo.SETparametro:
            self._out.Parametros(self.parametros,self.FOotimo)
        else:
            warn('The parameters report was not created because the optimize method or SETparameter method was not executed')
        # ---------------------------------------------------------------------
        # PREDICTION AND RESIDUAL ANALYSIS REPORT
        # ---------------------------------------------------------------------
        # Creating the prediction report if the prediction method was executed.
        if self.__controleFluxo.predicao:
            # If the residualAnalysis has been performed, a complete report can be made
            kwargs['PA'] = self.PA
            if self.__controleFluxo.analiseResiduos:
                self._out.Predicao(self.x,self.y,self.estatisticas,**kwargs)
            else:
                self._out.Predicao(self.x,self.y,None,**kwargs)
                warn('The residue analysis report has not been created because the residualAnalysis method has not been carried out. However, you can still export the prediction')
        else:
            warn('The report on the prediction and residual analysis was not created because the prediction method was not executed')

