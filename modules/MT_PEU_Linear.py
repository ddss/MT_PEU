# -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# Importação de pacotes de terceiros
from numpy import array, ones, \
hstack, shape

from casadi import mtimes, vertcat, horzcat, MX
from  pandas import ExcelFile,read_excel,read_csv,concat,DataFrame
from numpy.linalg import inv

# Exception Handling
from warnings import warn

# Rotinas Internas
from MT_PEU import EstimacaoNaoLinear

import pandas as pd #importação da biblioteca pandas para possibilitar trabalhar com o xlsx
from os import listdir
from collections import Counter
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
    
    def __init__(self,symbols_y,symbols_uy, symbols_x,symbols_ux,symbols_param,PA=0.95,folder='Projeto',**kwargs):
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
            EstimacaoNaoLinear.__init__(self, Model_2,symbols_y,symbols_uy, symbols_x,symbols_ux, symbols_param, PA, folder, **kwargs)
        else:
            # The the independent term won't be calculated and Model_1 should be used
            EstimacaoNaoLinear.__init__(self, Model_1, symbols_y,symbols_uy, symbols_x,symbols_ux, symbols_param, PA, folder, **kwargs)

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

    def setDados(self, data,separador=';',decimal='.', glx=[], gly=[],uxy=None):

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
        # MANUAL DATA ENTRY
        aux_list = []  # list auxiliary used for error

        def manual_entry(data):  # VALIDATION TO MANUAL DATA ENTRY
            # Tests if input data lists are the same size
            for i in list(data.keys()):
                if len(data[i]) != len(data[list(data.keys())[0]]):
                    aux_list.append(i)
            if len(aux_list) == 1:
                raise ValueError(f"The list {','.join(aux_list)} differs in the amount of points from the first list")

            elif len(aux_list) > 1:
                raise ValueError(f"The list {','.join(aux_list)} differs in the amount of points from the first list")
            for listas in data:
                if len(data[listas]) == 0:
                    aux_list.append(listas)
            if len(aux_list) == 1:
                raise ValueError(f"Data list {','.join(aux_list)} cannot be empty")
            elif len(aux_list) > 1:
                raise ValueError(f"Data lists {','.join(aux_list)} cannot be empty")

        if isinstance(data, dict):  # manual input mode case passed only one dictionary
            manual_entry(data)  # VALIDATION
            # Test if the symbols passed in the object instantiation parameters of the MT_PEU class are all in the database
            list_names = self.x.simbolos + self.y.simbolos + self.y.simbolos_incertezas + self.x.simbolos_incertezas
            for symb in list_names:
                if symb not in list(data.keys()):
                    raise ValueError("The symbol {} was not passed  in database".format(symb))

            X = array([data[i] for i in self.x.simbolos], ndmin=2, dtype=float).transpose()
            uX = array([data[i] for i in self.x.simbolos_incertezas], ndmin=2, dtype=float).transpose()
            Y = array([data[i] for i in self.y.simbolos], ndmin=2, dtype=float).transpose()
            uY = array([data[i] for i in self.y.simbolos_incertezas], ndmin=2, dtype=float).transpose()

        # -----------------------------------------------------------------------------
        # ROUTINE THAT IMPORTS AND VALIDATES DATA FROM .CSV AND .XLSX FILES
        # -----------------------------------------------------------------------------
        elif isinstance(data, list) or isinstance(data,
                                                  str):  # Data input, in file import mode, accepts string or list of strings and dictionary
            if not isinstance(data, list):
                data = [data]  # The data is expected in list format, but if it is a string, it is added to a list
            ###VALIDATION###
            error = False
            for name_validation in data:
                if not (isinstance(name_validation, str) or isinstance(name_validation, dict)):
                    error = True
            if error:
                raise ValueError(f"You must pass a strig or dictionary in list of setDados")

            aux_list1 = []  # Auxiliary list to separate dictionaries from strings and thus validate repeated strings
            for element in data:
                if isinstance(element, dict):
                    aux_list1.append(element)
                    data.remove(element)

            for value in Counter(data):  # warns that files with repeated names were passed
                if Counter(data)[value] != 1:
                    aux_list.append(value)
            if len(aux_list) > 1:
                warn(f"Files passed with same names :{','.join(aux_list)}.", Warning)
            elif len(aux_list) == 1:
                warn(f"File passed with same name: {','.join(aux_list)}.", Warning)

            data = list(set(data)) + aux_list1  # remove repeated names from past file list

            lista_dataframe = []  # empty list to save the dataframes and then concatenate into one
            for i in range(len(data)):  # iterative loop with all filenames passed

                if isinstance(data[i], dict):  # Adds manual data to imported data
                    manual_entry(data[i])  # validation of manual input
                    lista_dataframe.append(DataFrame(data[i]))  # creates dataframe with manual data
                else:
                    # In case the filenames already have the extension
                    if '.csv' in data[i] or '.xlsx' in data[i]:  # Fetch the format in the past names
                        if '.xlsx' in data[i]:  # Caso  onde a extensão é .xlsx
                            # nomes_planilhas é a lista com títulos das  planilhas do excel
                            nomes_planilhas = ExcelFile(data[i]).sheet_names
                            if len(nomes_planilhas) > 1:  # Caso o .xlsx tenha mais de uma planilha de dados
                                # Caso o .xlsx tenha mais de uma planilha
                                data_variable = read_excel(data[i], sheet_name=nomes_planilhas)
                                # data_variable armazena cada planilha do excel como um dataframe, em forma de dicionário
                                for i2 in data_variable:
                                    lista_dataframe.append(data_variable[
                                                               i2])  # cria uma lista de dataframes usando o dicionário de dataframes
                            else:
                                # Caso o .xlsx tenha apenas uma planilha de dados
                                lista_dataframe.append(read_excel(data[i]))
                        else:  # Caso onde a extensão é.csv
                            lista_dataframe.append(read_csv(data[i], sep=separador, decimal=decimal))
                    else:
                        # Caso que o usuário passou o nome de arquivo sem  extensão
                        # O programa vai procurar na pasta um arquivo que possua essa extensão
                        # Já que não é possível importar um arquivo sem saber sua extensão
                        lista_arquivos_ini = listdir()  # importa lista de arquivos que esteja na mesma pasta
                        lista_arquivos = sorted(lista_arquivos_ini,
                                                key=len)  # organiza a lista em ordem crescente do tamanho das strings

                        controle = False  # variável responsável por limitar que em cada iteração traga apenas um arquivo
                        for j in range(len(lista_arquivos)):
                            if not controle:
                                if data[i] in lista_arquivos[
                                    j]:  # traz o arquivo que contenha o mesmo nome antes do ponto
                                    name1 = lista_arquivos[j]  # nome de arquivo com o seu formato
                                    # No entanto esses arquivos podem ser dados de planilha eletrônica ou csv
                                    if name1.replace(data[i],
                                                     '') == '.xlsx':  # depois de encontrar o arquivo na pasta,a rotina agora tem nome e extensão para fazer a importação
                                        nomes_planilhas = ExcelFile(
                                            name1).sheet_names  # função que traz a lista com títulos das  planilhas
                                        if len(nomes_planilhas) > 1:  # caso que o arquivo .xlsx tem mais de uma planilha,assim será gerado um dataframe por planilha
                                            data_variable = read_excel(name1,
                                                                       sheet_name=nomes_planilhas)  # dicionário com os dataframes das planilhas contida no arquivo .xlsx
                                            for i2 in data_variable:
                                                lista_dataframe.append(data_variable[
                                                                           i2])  # adiciona todos os dataframes a lista de dataframes
                                                controle = True
                                        else:  # caso que o arquivo xlsx possui apenas uma planilha
                                            lista_dataframe.append(read_excel(name1))
                                            controle = True
                                    elif name1.replace(data[i], '') == '.csv' or name1.replace(data[i],
                                                                                               '') == '.CSV':  # caso que o arquivo é um .csv
                                        data_frame_csv = read_csv(name1, decimal=decimal, sep=separador)
                                        lista_dataframe.append(data_frame_csv)
                                        controle = True
                        if controle == False:  # se não houver nenhum arquivo com o nome passado pelo usuário
                            raise TypeError('There is no file with the name {} in the file folder'.format(data[i]))

            dataframe_geral = concat(lista_dataframe, axis=1)  # concatenates all dataframes into a single one
            dataframe_geral.drop([col for col in dataframe_geral.columns if "Unnamed" in col], axis=1,
                                 inplace=True)  # removes columns with unnamed title

            for nulos in dict(
                    dataframe_geral.isnull().sum()):  # dictionary that brings the symbols as keys and the number of empty lines as values
                if dict(dataframe_geral.isnull().sum())[nulos] != 0:
                    # Tests if there are empty rows in the columns of each symbol
                    aux_list.append(nulos)
            if len(aux_list) > 0:
                raise ValueError(
                    f"In quantity{'s'[:int(len(aux_list)) ^ 1]} {','.join(aux_list)} there are empty lines or the quantitity of data points are inconsistenty")

            # Test if the symbols passed in the object instantiation parameters of the MT_PEU class are all in the dataset
            list_names = self.x.simbolos + self.y.simbolos + self.y.simbolos_incertezas + self.x.simbolos_incertezas
            for symb in list_names:
                if symb not in dataframe_geral.columns.tolist():
                    raise ValueError("The symbol {} was not passed  in database".format(symb))

            # Creation of estimation and uncertainty matrices
            X = dataframe_geral[self.x.simbolos].to_numpy(dtype=float)
            Y = dataframe_geral[self.y.simbolos].to_numpy(dtype=float)
            uX = dataframe_geral[self.x.simbolos_incertezas].to_numpy(dtype=float)
            uY = dataframe_geral[self.y.simbolos_incertezas].to_numpy(dtype=float)

        else:
            raise TypeError(
                " The data input can be  a list or string or dictionary, check if the input follows any of these formats")

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