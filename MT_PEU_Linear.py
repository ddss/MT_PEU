# -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# Importação de pacotes de terceiros
from numpy import array, transpose, concatenate,size, ones, \
hstack, shape, ndarray
from scipy.stats import t
from threading import Thread
from sys import exc_info

from casadi import mtimes, vertcat, horzcat, MX

from numpy.linalg import inv

# Exception Handling
from warnings import warn

# Rotinas Internas
from MT_PEU import EstimacaoNaoLinear

import pandas as pd #importação da biblioteca pandas para possibilitar trabalhar com o xlsx

# Fim da importação

def Model_1 (param,x, *args):

    # This model is used when there is no independent term calculation

    p = [] ; var = []
    for i in range(param.rows()):
        p = vertcat(p, param[i])

    var = horzcat(var, x)


    return mtimes(var,p)

def Model_2 (param,x,*args):

    # This model is used when the independent term is calculated
    p = [] ; var = []
    for i in range(param.rows()):
        p = vertcat(p, param[i])
    #for i in range(len(x)):
    rows = args[0] # This argument corresponds to the amount of input data
    var = horzcat(var,x)
    var = horzcat(var, MX.ones(rows, 1))

    return mtimes(var, p)

class EstimacaoLinear(EstimacaoNaoLinear):
    
    def __init__(self,symbols_y,symbols_x,symbols_param,PA=0.95,folder='Projeto',**kwargs):
        u'''
        Classe para executar a estimação de parâmetros de modelos MISO lineares nos parâmetros       

        ======================
        Bibliotecas requeridas
        ======================
        * Numpy
        * Scipy
        * Matplotlib
        * Math
        * PSO - **Obtida no link http://github.com/ddss/PSO. Os códigos devem estar dentro de uma pasta de nome PSO**

        =======================
        Entradas (obrigatórias)
        =======================
        * ``symbols_y`` (list)     : lista com os simbolos das variáveis y (Não podem haver caracteres especiais)
        * ``symbols_x`` (list)     : lista com os simbolos das variáveis x (Não podem haver caracteres especiais)
        * ``symbols_param`` (list) : lista com o simbolos dos parâmetros (Não podem haver caracteres especiais)

        ====================
        Entradas (opcionais)
        ====================
        * ``PA`` (float): probabilidade de abrangência da análise. Deve estar entre 0 e 1. Default: 0.95
        * ``folder`` (string): nome do projeto (Náo podem haver caracteres especiais)

        **AVISO**:
        * Para cálculo do coeficiente linear, basta que o número de parâmetros seja igual ao número de grandezas
        independentes + 1.

        ==============================
        Keywords (Entradas opcionais):
        ==============================
        
        * ``names_x``        (list): lista com os nomes para x
        * ``units_x``        (list): lista com as unidades para x (inclusive em formato LATEX)
        * ``label_latex_x``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``names_y``        (list): lista com os nomes para y
        * ``units_y``        (list): lista com as unidades para y (inclusive em formato LATEX)
        * ``label_latex_y``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``names_param``       (list): lista com os nomes para os parâmetros (inclusive em formato LATEX)
        * ``units_param``       (list): lista com as unidades para os parâmetros (inclusive em formato LATEX)
        * ``label_latex_param`` (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``base_path`` (string): String que define o diretório pai que serão criados/salvos os arquivos gerados pelo motor de cálculo
        
        =======        
        Métodos
        =======
        
        Para a realização da estimação de parâmetros de um certo modelo faz-se necessário executar \
        alguns métodos, na ordem indicada:
            
        **ESTIMAÇÂO DE PARÂMETROS** 
        
        * ``setConjunto``        : método para incluir dados obtidos de experimentos. Neste há a opção de determinar \
        se estes dados serão utilizados como dados para estimar os parâmetros ou para validação. (Vide documentação do método)
        * ``optimize``              : método para realizar a otimização, com base nos dados fornecidos em setConjunto.
        * ``parametersUncertainty``  : método que avalia a incerteza dos parâmetros (Vide documentação do método)
        * ``setConjunto``        : (é opcional para inclusão de dados de validação)
        * ``Prediction``             : método que avalia a predição do modelo e sua incerteza ou utilizando os pontos experimentais ou de \
        validação, se disponível (Vide documentação do método) 
        * ``residualAnalysis``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``plots``             : método para criação dos gráficos (Vide documentação do método)
        * ``_armazenarDicionario`` : método que retorna as grandezas sob a forma de um dicionário (Vide documentação do método)
        
        ====================
        Fluxo de trabalho        
        ====================
        
        Esta classe valida a correta ordem de execução dos métodos. É importante salientar que cada vez que o método ``setConjunto`` \
        é utilizado, é criado um novo ``Fluxo de trabalho``, ou seja, o motor de cálculo valida de alguns métodos precisam ser reexecutados \
        devido a entrada de novos dados.
        
        **Observação 1**: Se forem adicionados diferentes dados de validação (execuções do método gerarEntradas para incluir tais dados), \
        são iniciado novos fluxos, mas é mantido o histórico de toda execução.
        
        **Observação 2**: Se forem adicionados novos dados experimentais, todo o histórico de fluxos é apagado e reniciado.
         
        ======      
        Saídas
        ======
        
        As saídas deste motor de cálculo estão, principalmente, sob a forma de atributos e gráficos.
        Os principais atributos de uma variável Estimacao, são:
                
        * ``x`` : objeto Grandeza que contém todas as informações referentes às grandezas \
        independentes sob a forma de atributos:
            * ``estimação`` : referente aos dados experimentais. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``calculado``    : referente aos dados calculados pelo modelo. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``predição``    : referente aos dados de validação. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``residuos``     : referente aos resíduos de regressão. Principais atributos: ``matriz_estimativa``, ``estatisticas``
            
        * ``y``          : objeto Grandeza que contém todas as informações referentes às grandezas \
        dependentes sob a forma de atributos. Os atributos são os mesmos de x.

        * ``parametros`` : objeto Grandeza que contém todas as informações referentes aos parâmetros sob a forma de atributos.
            * ``estimativa``         : estimativa para os parâmetros
            * ``matriz_covariancia`` : matriz de covariância
            * ``regiao_abrangencia`` : pontos contidos na região de abrangência
        
        Obs.: Para informações mais detalhadas, consultar os Atributos da classe Grandeza.        
        '''
        # ---------------------------------------------------------------------
        # INICIANDO A CLASSE INIT
        # ---------------------------------------------------------------------
        # For to start the class it's necessary to check wich is the most suitable model

        if (len(symbols_param) == len(symbols_x) + 1):
            # The the independent term will be calculated and Model_2 should be used
            EstimacaoNaoLinear.__init__(self, Model_2, symbols_y, symbols_x, symbols_param, PA, folder, **kwargs)
        else:
            # The the independent term won't be calculated and Model_1 should be used
            EstimacaoNaoLinear.__init__(self, Model_1, symbols_y, symbols_x, symbols_param, PA, folder, **kwargs)

        self._EstimacaoNaoLinear__flag.ToggleActive('Linear') # Enable linear flag to correctly create 'self .__ values' when independent term calculation
        self._EstimacaoNaoLinear__flag.setCaracteristica(['calc_termo_independente'])

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if self.y.NV != 1:
            raise ValueError(u'It is performing the parameter estimation of linear models only for the MISO case.')

        if (self.parametros.NV != self.x.NV) and (self.parametros.NV != self.x.NV+1):
            raise ValueError(u'The number of parameters must be equal to the number of independent quantities (the linear coefficient is not calculated).'+\
            'OR equal to the number of independent quantities + 1 (the linear coefficient is calculated).')

        self.__coluna_dumb = False # this variable indicates that a column of ones has been added to independent quantities
        # ---------------------------------------------------------------------
        # Definindo se o b será calculado
        # ---------------------------------------------------------------------
        if (self.parametros.NV == self.x.NV+1):
            self._EstimacaoNaoLinear__flag.ToggleActive('calc_termo_independente')
            self.__coluna_dumb = True

    def setDados(self, data, worksheet=None, glx=[], gly=[],uxy=None):

        u'''
        Método para incluir os dados de entrada da estimação e predição (quando chamado setDados pela segunda vez)
        
        =======================
        Entradas (Obrigatórias)
        =======================
        
        * data        : array com os dados experimentais das variáveis independentes na forma de colunas
        * worksheet        : nome das planilhas

        **Aviso**:
        Caso não definidos dados de validação, será assumido os valores experimentais                    
        '''

         # EXECUTION

        self._EstimacaoNaoLinear__controleFluxo.SET_ETAPA('setDados')

        if type(data) == dict:

            # VALIDATION TO MANUAL DATA ENTRY

            if len(data[0]) == 0 or len(data[1]) == 0:
                raise TypeError('It is necessary to include at least data for one quantity: ([data],[uncertainty])')
            for i in range(2):
                for ele in data[i]:
                    if not (isinstance(ele, list) or isinstance(ele, tuple)):
                        raise TypeError(
                            'Each quantity pair (data and uncertainty) must be a tuple or list: ([data],[uncertainty]).')

                if len(ele) != 2:
                    raise TypeError('Each tuple must contain only 2 lists: ([data],[uncertainty])')

                for ele_i in ele:
                    if not (isinstance(ele, list) or isinstance(ele, tuple)):
                        raise TypeError(
                            'Each quantity pair (data and uncertainty) must be a tuple or list: ([data],[uncertainty]).')

            X = transpose(array([data[0][i][0] for i in range(len(data[0]))], ndmin=2, dtype=float))
            uX = transpose(array([data[0][i][1] for i in range(len(data[0]))], ndmin=2, dtype=float))
            Y = transpose(array([data[1][i][0] for i in range(len(data[1]))], ndmin=2, dtype=float))
            uY = transpose(array([data[1][i][1] for i in range(len(data[1]))], ndmin=2, dtype=float))
        elif type(data) == str:
            if worksheet is None:
                # VALIDATION SHEET NAME
                # valida os nomes das planilhas , mostra o erro caso o usuário mudou os nomes padrões
                if pd.ExcelFile(data).sheet_names[0] != "independent variable" or \
                        pd.ExcelFile(data).sheet_names[1] != "dependent variable":
                    raise TypeError(
                        "Worksheet names can be default  (independent variable) and (dependent variable), or set your  worksheet")

                data_dependent_variable = pd.read_excel(data, sheet_name="dependent variable")
                data_independent_variable = pd.read_excel(data, sheet_name="independent variable")
            else:
                data_dependent_variable = pd.read_excel(data, sheet_name=worksheet[1])
                data_independent_variable = pd.read_excel(data, sheet_name=worksheet[0])

            # VALIDATION TO DATA USING  EXCEL
            # Remove colunas sem títulos
            data_independent_variable.drop([col for col in data_independent_variable.columns if "Unnamed" in col],
                                           axis=1, inplace=True)
            data_dependent_variable.drop([col for col in data_dependent_variable.columns if "Unnamed" in col], axis=1,
                                         inplace=True)
            # informa quais linhas forma removidas
            if len(list(data_dependent_variable[data_dependent_variable.isnull().values.any(axis=1)].index.values)) != 0:
                warn("the respective lines of dependent variable have been removed {} ".format(list(asarray(
                    list(data_dependent_variable[data_dependent_variable.isnull().values.any(axis=1)].index.values)) + 2)),
                     UserWarning)
            if len(list(data_independent_variable[
                            data_independent_variable.isnull().values.any(axis=1)].index.values)) != 0:
                warn("the respective lines of independent variable have been removed {} ".format(list(asarray(list(
                    data_independent_variable[
                        data_independent_variable.isnull().values.any(axis=1)].index.values)) + 2)), UserWarning)

            # remove as linhas
            data_dependent_variable = data_dependent_variable.dropna()
            data_independent_variable = data_independent_variable.dropna()

            Y = data_dependent_variable[
                {data_dependent_variable.columns[i] for i in range(0, data_dependent_variable.shape[1], 2)}].to_numpy()

            uY = data_dependent_variable[
                {data_dependent_variable.columns[i] for i in range(1, data_dependent_variable.shape[1], 2)}].to_numpy()
            X = data_independent_variable[
                {data_independent_variable.columns[i] for i in
                 range(0, data_independent_variable.shape[1], 2)}].to_numpy()
            uX = data_independent_variable[
                {data_independent_variable.columns[i] for i in
                 range(1, data_independent_variable.shape[1], 2)}].to_numpy()


        else:
            raise TypeError(
                "The data input can be either in dictionary format or string ""excel file name"", check if the input follows any of these formats")

        self._EstimacaoNaoLinear__validacaoDadosEntrada(X, uX, self.x.NV)
        self._EstimacaoNaoLinear__validacaoDadosEntrada(Y, uY, self.y.NV)

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------

        # Validação do número de dados experimentais
        if X.shape[0] != Y.shape[0]:
            raise ValueError('{:d} data were entered for dependent quantities, but {:d} for independent quantities'.format(self.__ytemp.shape[0],self.__xtemp.shape[0]))

        # ---------------------------------------------------------------------
        # MODIFICAÇÕES DAS MATRIZES DE DADOS
        # ---------------------------------------------------------------------
        coluna_dumb = False
        if self._EstimacaoNaoLinear__flag.info['calc_termo_independente']:
            X = hstack((X, ones((shape(X)[0], 1))))
            uX = hstack((uX, ones((shape(uX)[0], 1))))
            coluna_dumb = True

       #Automatizando dataType para que o usuário não precise indicar na entrada de dados
        if not self._EstimacaoNaoLinear__flag.info['dadosestimacao']:  # rodando a primeira vez (estimação)
            dataType = self._EstimacaoNaoLinear__tiposDisponiveisEntrada[0]
        else:  # rodando a segunda vez (predição)
            dataType = self._EstimacaoNaoLinear__tiposDisponiveisEntrada[1]



        if dataType == 'estimacao':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadosestimacao')
            if self._EstimacaoNaoLinear__controleFluxo.FLUXO_ID != 0:
                self._EstimacaoNaoLinear__controleFluxo.reiniciar()
                if self.__flag.info['dadospredicao']:
                    warn('The flux was restarted, so new validation data has to be included')
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados experimentais nas variáveis.
            try:
                self.x._SETdadosestimacao(estimativa=X,matriz_incerteza= uX ,gL=glx,coluna_dumb=self.__coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the estimation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosestimacao(estimativa=Y,matriz_incerteza=uY,gL=gly)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the estimation set of the quantity Y: {}'.format(erro))

        if dataType == 'predicao':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadospredicao')
            self._EstimacaoNaoLinear__controleFluxo.reiniciarParcial()

            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            try:
                self.x._SETdadosvalidacao(estimativa=X,matriz_incerteza=uX,gL=glx,coluna_dumb=coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=Y,matriz_incerteza=uY,gL=gly)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity Y: {}'.format(erro))

        if self._EstimacaoNaoLinear__flag.info['dadospredicao'] == False:
            # Caso gerarEntradas seja executado somente para os dados experimentais,
            # será assumido que estes são os dados de validação, pois todos os cálculos 
            # de predição são realizados para os dados de validação.
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            try:
                self.x._SETdadosvalidacao(estimativa=X,matriz_incerteza=uX,gL=glx,coluna_dumb=self.__coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=Y,matriz_incerteza=uY,gL=gly)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity Y: {}'.format(erro))




    def optimize(self, parametersReport = True):
        u'''
        Método para obtenção da estimativa dos parâmetros e sua matriz de covariância.
        '''
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self._EstimacaoNaoLinear__controleFluxo.SET_ETAPA('otimizacao')

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if not self._EstimacaoNaoLinear__flag.info['dadosestimacao']:
            raise TypeError(u'For execute optimize is necessary to input the estimation data.')

        if self._EstimacaoNaoLinear__controleFluxo.SETparametro:
            raise TypeError(u'The optimization method cannot be run with SETparameter.')

        # ---------------------------------------------------------------------
        # RESOLUÇÃO
        # ---------------------------------------------------------------------
        X   = self.x.estimacao.matriz_estimativa
        Uyy = self.y.estimacao.matriz_covariancia
        y   = self.y.estimacao.vetor_estimativa
        variancia = inv(X.transpose().dot(inv(Uyy)).dot(X))
        parametros = variancia.dot(X.transpose().dot(inv(Uyy))).dot(y)
        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZA
        # ---------------------------------------------------------------------
        self.parametros._SETparametro(parametros.transpose()[0].tolist(),variancia,None)

        # ---------------------------------------------------------------------
        # FUNÇÃO OBJETIVO NO PONTO ÓTIMO
        # ---------------------------------------------------------------------
        # initialization of the method that create the symbolic's variables
        EstimacaoNaoLinear._constructionCasadiVariables(self)

        self.FOotimo = float(self._excObjectiveFunction(self.parametros.estimativa,self._values))

        # parameters report creation
        if parametersReport:
            self._out.Parametros(self.parametros, self.FOotimo)

    def parametersUncertainty(self, objectiveFunctionMapping=True, parametersReport = True, **kwargs):
        u'''
        Método para avaliar a região de abrangência dos parâmetros.

        **Observação**:
        A matriz de covariância dos parâmetros é calculada juntamente com a otimização, por ser parte constituinte da solução analítica. Entretanto,
        caso o método SETparametros seja executado e neste não seja definida a matriz de covariância, ela é calculada.

        ==================
        Entradas opcionais
        ==================

        * objectiveFunctionMapping (bool): identifica se será executado algoritmo para preenchimento da região de abrangência.
        '''
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self._EstimacaoNaoLinear__controleFluxo.SET_ETAPA('incertezaParametros')

        # ---------------------------------------------------------------------
        # CÁLCULO DA MATRIZ DE COVARIÂNCIA
        # ---------------------------------------------------------------------
        # Caso a matriz de covariância não seja calculada, ela será aqui calculada
        X   = self.x.estimacao.matriz_estimativa
        Uyy = self.y.estimacao.matriz_covariancia
        variancia = inv(X.transpose().dot(inv(Uyy)).dot(X))
        self.parametros._updateParametro(matriz_covariancia=variancia)

        # ---------------------------------------------------------------------
        # CÁLCULO DA REGIÃO DE ABRANGÊNCIA
        # ---------------------------------------------------------------------
        # A região de abrangência só é calculada caso não esteja definida
        if objectiveFunctionMapping and self.parametros.NV != 1:
            self._EstimacaoNaoLinear__objectiveFunctionMapping(**kwargs)
            self._EstimacaoNaoLinear__flag.ToggleActive('mapeamentoFO')

        # A região de abrangência só é executada caso haja histórico de posicoes e fitness
        if self._EstimacaoNaoLinear__controleFluxo.mapeamentoFO and self.parametros.NV != 1:
            # OBTENÇÃO DA REGIÃO:
            regiao = self.regiaoAbrangencia()
            # ATRIBUIÇÃO A GRANDEZA
            self.parametros._updateParametro(regiao_abrangencia=regiao)

        # parameters report creation
        if parametersReport:
            self._out.Parametros(self.parametros, self.FOotimo)