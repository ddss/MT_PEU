# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:05:02 2015

@author: danielsantana
"""
# Importação de pacotes de terceiros
from numpy import array, size, diag, linspace, min, max, \
    mean,  std, ndarray, insert, isfinite, arange, sqrt, zeros

from numpy.linalg import cond

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.correlation import plot_corr

from scipy.stats import normaltest, shapiro, ttest_1samp, kstest

from matplotlib.pyplot import savefig, close
    
from matplotlib.colors import LinearSegmentedColormap

from os import getcwd, sep

# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import Validacao_Diretorio, matrizcorrelacao

from Graficos import Grafico

class Grandeza:

    def __init__(self,simbolos, simbolos_incertezas, nomes=None, unidades=None, label_latex=None):
        u'''
        Classe para organizar as características das Grandezas:
                
        =======
        Entrada
        =======
        
        **OBRIGATÓRIO**:
        
        * ``simbolos`` (list)   : deve ser uma lista contendo os símbolos, na ordem de entrada de cada variável   

        **OPCIONAL**:
        
        * ``nomes``       (list) : deve ser uma lista contendo o nome das variáveis
        * ``unidades``    (list) : deve ser uma lista contendo as unidades das variáveis
        * ``label_latex`` (list) : deve ser uma lista contendo os símbolos em formato LATEX
        
        =======
        Métodos
        =======

        **DEFINICIONAIS** - Usado para criação de atributos:
        
        * ``_SETestimação``    : irá criar o atributo estimação. Deve ser usado se se tratar de dados experimentais
        * ``_SETmodelo``       : irá criar o atributo modelo. Deve ser usado se se tratar de dados do modelo
        * ``_SETvalidacao``    : irá criar o atributo validacao. Deve ser usado se se tratar de dados de validação
        * ``_SETparametro``    : irá criar os atributos estimativa, matriz_covariancia, regiao_abrangencia. Deve ser usado para os parâmetros
        * ``_SETresiduos``     : irá criar o atributo resíduos. Deve ser usado para os resíduos de gamma

        **OUTROS**:
        
        * ``labelGraficos``       : método que retorna os títulos de eixos de gráficos com base nas informações disponíveis
        * ``_testesEstatisticos`` : método para realizar testes estatísticos na variável
        * ``Graficos``            : método para criar gráficos que dependem exclusivamente da grandeza

        =========
        Atributos
        =========
        
        **ATRIBUTOS GERAIS**:
        
        * ``.simbolos``    (list): lista com os símbolos das variáveis (inclusive em código Latex)
        * ``.nomes``       (list): lista com os nomes das variáveis
        * ``.unidades``    (list): lista com as unidades das variáveis
        * ``.label_latex`` (list): lista com o label_latex das variáveis
        * ``.NV``         (float): número de variáveis na grandeza
        
        **GRANDEZAS DEPENDENTES E INDEPENDENTES**:

        * ``.estimação`` (objeto): objeto Organizador que armazena os valores e incertezas dos dados experimentais \
        (vide documentação do mesmo). **só exitirá após execução do método _SETexperimental**
        * ``.validacao``    (objeto): objeto Organizador que armazena os valores e incertezas dos dados de validação \
        (vide documentação do mesmo). **só exitirá após execução do método _SETvalidacao**
        * ``.evaluated``    (objeto): objeto Organizador que armazena os valores e incertezas dos dados evaluated pelo modelo \
        (vide documentação do mesmo). **só exitirá após execução do método _SETcalculado**
        * ``.residual``     (objeto): objeto Organizador que armazena os valores e incertezas dos resíduos \
        (vide documentação do mesmo). **só exitirá após execução do método _SETcalculado**

        **PARÂMETROS**
        (atributos só existirão após a execução do método _SETparametro)
        
        * ``.estimativa`` (list): lista com estimativas. 
        * ``.matriz_covariancia`` (array): array representando a matriz covariância. 
        * ``.matriz_correlcao``   (array): array representando a matriz dos coeficientes de correlação. 
        * ``.regiao_abrangencia`` (list): lista representando os pontos pertencentes à região de abrangência.
        '''

        # ------------------------------------------------------------------------------------
        # VALIDAÇÂO
        # -------------------------------------------------------------------------------------
        if simbolos is None:
            raise NameError('You must insert te symbols of the quantities.')


        self.__validacaoEntrada(simbolos,simbolos_incertezas,nomes,unidades,label_latex)

        # ------------------------------------------------------------------------------------
        # CRIAÇÃO DE ATRIBUTOS
        # -------------------------------------------------------------------------------------
        # simbolos: usado como referência para a quantidade de variáveis da grandeza

        self.simbolos           = simbolos
        self.simbolos_incertezas= simbolos_incertezas

        # nomes, unidades e label_latex: utilizados para plotagem
        # caso não definidos, eles serão uma lista de elementos None, para manter a consistência dimensional
        self.nomes       = nomes if nomes is not None else [None]*len(simbolos)
        self.unidades    = unidades if unidades is not None else [None]*len(simbolos)
        self.label_latex = label_latex if label_latex is not None else [None]*len(simbolos)

        # Número de pontos experimentais e de variáveis
        self.NV = len(simbolos)
        
        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------   
        self.__ID = [] # ID`s que a grandeza possui

    @property
    def __ID_available(self):
        # Todos os ID's disponíveis
        return ['observed', 'evaluated', 'parametros', 'residuo']

    @property
    def _available_dataType(self):
        return ['estimation', 'validation']

    @property
    def __configLabel(self):
        # Label para gráficos:
        #       observed                            evaluated
        return {self.__ID_available[0]: 'observed', self.__ID_available[1]: 'evaluated'}

    @property
    def __tipoGraficos(self):
        return ('regiaoAbrangencia', 'grandezas-entrada', 'grandezas-calculadas', 'optimization', 'analiseResiduos')

    def __validacaoEntrada(self, simbolos, simbolos_incertezas, nomes,unidades, label_latex):
        u'''
        Validação:
        
        * se os simbolos,simbolos das  incertezas , nome, unidades e label_latex são Listas.
        * se os elementos de simbolos,simbolos das  incertezas, nome, unidades e label_latex são strings (ou caracteres unicode)
        * se os elementos de simbolos possuem caracteres não permitidos (caracteres especiais)
        * se os simbolos são distintos
        * se os tamanhos dos atributos de simbologia, nome, unidades e label_latex são os mesmos.
        '''
        if simbolos_incertezas is not None :#Se simbolos_incertezas é nada não existe necessidade de validação
            # Verificação se os símbolos das incertezas  possuem caracteres especiais
            for simb1 in simbolos_incertezas:
                if not simb1.isalnum():
                    raise NameError(
                        'The symbols of uncertainty cannot have special characters. Incorrect Symbol: ' + simb1)
            # Verificação se os símbolos das incertezas são distintos
            # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
            if len(set(simbolos_incertezas)) != len(simbolos_incertezas):
                raise NameError('The symbols of each quantity must be different.')

                # Verificação se os símbolos apenas diferenciados por maiúsculo ou minúsculo
                # realização do teste
                results = []
                for sym in simbolos_incertezas:
                    test = [sym.lower() == sym2.lower() or sym.upper() == sym2.upper() for sym2 in
                            simbolos]  # Cria uma matriz com o resultados dos testes
                    if sum(test) != 1:  # Diferente de 1 significa que há mais de um True naquela linha
                        results.append(sym)  # busca o respectivo símbolo para aquele teste
                if len(
                        results) > 0:  # Maior que 0, pois, quando houver problemas, pelo menos um símbolo será identificado: maiúsculo e minúsculo
                    raise NameError(
                        'It is not possible to use the same symbols differentiated by upper or lower case. Please change these symbols: ' + str(
                            results))

        # Verificação se nomes, unidade e label_latex são listas
        for elemento in [simbolos, simbolos_incertezas, nomes, unidades, label_latex]:
            if elemento is not None:
                if not isinstance(elemento,list):
                    raise TypeError('For a quantity, the symbols, names, units, and label_latex must be informed in the form of a list.')
                # verificação se os elementos são strings
                for value in elemento:
                    if value is not None:
                        if not isinstance(value,str) and not isinstance(value,unicode):
                            raise TypeError('Symbols, names, units and label_latex must be STRINGS.')

        # Verificação se os símbolos possuem caracteres especiais
        for simb in simbolos:
            if not simb.isalnum():
                raise NameError('The symbols of quantities cannot have special characters. Incorrect Symbol: '+simb)

        # Verificação se os símbolos são distintos
        # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
        if len(set(simbolos)) != len(simbolos):
            raise NameError('The symbols of each quantity must be different.')

        # Verificação se os símbolos apenas diferenciados por maiúsculo ou minúsculo
        # realização do teste
        results = []
        for sym in simbolos:
            test = [sym.lower() == sym2.lower() or sym.upper() == sym2.upper() for sym2 in simbolos] # Cria uma matriz com o resultados dos testes
            if sum(test) != 1: # Diferente de 1 significa que há mais de um True naquela linha
                results.append(sym) # busca o respectivo símbolo para aquele teste
        if len(results) > 0: # Maior que 0, pois, quando houver problemas, pelo menos um símbolo será identificado: maiúsculo e minúsculo
            raise NameError('It is not possible to use the same symbols differentiated by upper or lower case. Please change these symbols: ' + str(results))

       # Verificação se nomes, unidade e label_latex possuem mesmo tamanho
        for elemento in [simbolos_incertezas,nomes,unidades,label_latex]:
            if elemento is not None:
                if len(elemento) != len(simbolos):
                    raise ValueError('Symbols, names, units and label_latex must be lists of the same size.')

    class Dados:

        def __init__(self,estimativa, NV, matriz_incerteza=None, matriz_covariancia=None, symbols=None, gL=[], NE=None, **kwargs):
            """
            Classe interna para organizar os dados das estimativas e suas respectivas incertezas, disponibilizando-os na forma de matriz, vetores e listas.
            ========
            Entradas
            ========
            * ``estimativa`` (array) : estimativas para as observações das variáveis (na forma de um vetor ou matriz). \
            Caso seja uma matriz, cada coluna contém as estimativas para uma variável. Se for um vetor, as estimativas estão \
            numa única coluna, sendo necessário fornecer a entrada NE.
            * ``NV`` (int): número de variáveis
            * ``matriz_incerteza``  (array) : uncertainty para os valores das estimativas. Cada coluna contém a uncertainty para os pontos de uma variável.
            * ``matriz_variancia`` (array)  : variância para os valores das estimativas. Deve ser a matriz de covariância.
            * ``gL''(lista)                 : graus de liberdade
            * ``NE`` (int): quantidade de pontos experimentais. Necessário apenas quanto a estimativa é um vetor.
            **AVISO:**
            * se estimativa for uma matriz, espera-se que ``matriz_incerteza`` seja uma matriz em que cada coluna seja as *INCERTEZAS* para cada observação de uma certa variável (ela será o atributo ``.matriz_incerteza`` )
            * se estimativa for um vetor, espera-se que seja informada a ``matriz_covariância``
            * se for informada a matriz_incerteza, a matriz de covariância assumirá que os elementos fora da diagonal principal são ZEROS.

            =========
            ATRIBUTOS
            =========

                * ``.matriz_estimativa`` (array): cada variável está alocada em uma coluna que contém suas observações.
                * ``.vetor_estimativa``  (array): todas as observações de todas as variáveis estão em um único vetor.
                * ``.matriz_incerteza``  (array): matriz em que cada coluna contém a uncertainty de cada ponto de uma certeza variável.
                * ``.matriz_covariancia`` (array): matriz de covariância.
                * ``matriz_correlacao`` (array): matriz de correlação
                * ``NE`` (float): número de observações (para cada grandeza)

            =======
            METODOS
            =======
                * GETListas que retorna lista_estimativa, lista_incerteza, lista_variancia.

            ======
            Kwargs
            ======
                * coluna_dumb (bool): possibilita lidar com uma coluna adicional no FINAL do conjunto de dados, que não faz parte
                dos dados experimentais. Exemplo: estimação de parâmetros linear -> coluna de 1
            """

            # ---------------------------------------------------------------------
            # VALIDAÇÃO INICIAL DAS ENTRADAS
            # ---------------------------------------------------------------------

            if not isinstance(estimativa, ndarray):
                raise TypeError(u'The input data must be arrays.')

            if matriz_covariancia is not None and matriz_incerteza is not None:
                raise SyntaxError(u'It is not possible to define the covariance matrix and the uncertainty matrix together. You have to choose between them.')

            if matriz_covariancia is not None:
                if not isinstance(matriz_covariancia, ndarray):
                    raise TypeError(u'The input data must be arrays.')

            if matriz_incerteza is not None:
                if not isinstance(matriz_incerteza, ndarray):
                    raise TypeError(u'The input data must be arrays.')

            if not isinstance(gL, list):
                raise TypeError(u'Freedom degrees must be informed as a list.')

            # ---------------------------------------------------------------------------
            # KEYWORD ARGUMENTS
            # ---------------------------------------------------------------------------
            # Indica se há uma coluna adicional na matriz de estimativas.
            self._coluna_dumb = kwargs.get('coluna_dumb') if kwargs.get('coluna_dumb') is not None else False
            # ---------------------------------------------------------------------------
            # CRIAÇÃO DA MATRIZ ESTIMATIVA E VETOR ESTIMATIVA (ARRAYS)
            # ---------------------------------------------------------------------------
            # Caso haja uma coluna_dumb, ao número de variáveis é somado 1, para lidar com essa coluna adicional
            if self._coluna_dumb:
                NV += 1
            if estimativa.shape[1] == NV: # Foi informado a matriz estimativa (NE , NV)
                self.matriz_estimativa = estimativa
                self.vetor_estimativa = self.matriz_estimativa.reshape(
                    (int(self.matriz_estimativa.shape[0] * self.matriz_estimativa.shape[1]), 1),
                    order='F')  # conversão de matriz para vetor
                self.lista_estimativa = self.vetor_estimativa.transpose().tolist()[0]

            elif NE is not None:

                if estimativa.shape[0] == NV*NE: # Foi informado o vetor estimativa (NExNV,1)
                    self.vetor_estimativa = estimativa
                    self.matriz_estimativa = self.vetor_estimativa.reshape((NE, int(self.vetor_estimativa.shape[0] / NE)),
                                                                           order='F')  # Conversão de vetor para uma matriz
                    self.lista_estimativa = self.vetor_estimativa.transpose().tolist()[0]
                else:
                    raise ValueError(u'The size of the array containing the estimates must be equal to the product between the number of variables and the number of data')
            else:
                raise ValueError(u'The estimate was informed in the form of an array. NE must be specified.')

            # ---------------------------------------------------------------------
            # Número de pontos experimentais
            # ---------------------------------------------------------------------
            self.NE = self.matriz_estimativa.shape[0]

            # ---------------------------------------------------------------------------
            # CRIAÇÃO DA MATRIZ COVARIÂNCIA E MATRIZ INCERTEZA (ARRAYS)
            # ---------------------------------------------------------------------------
            if matriz_incerteza is not None:
                self.matriz_incerteza = matriz_incerteza
                self.matriz_covariancia = diag(
                    (self.matriz_incerteza ** 2).reshape((self.NE * self.matriz_incerteza.shape[1], 1),
                                                         order='F').transpose().tolist()[0])
                self.matriz_correlacao = matrizcorrelacao(self.matriz_covariancia)

            elif matriz_covariancia is not None:
                if NE is not None:
                    self.matriz_covariancia = matriz_covariancia
                    self.matriz_incerteza = (diag(self.matriz_covariancia)**0.5).reshape(
                        (NE, self.matriz_estimativa.shape[1]), order='F')
                    self.matriz_correlacao = matrizcorrelacao(self.matriz_covariancia)
                else:
                    raise ValueError(u'It is necessary to define the argument NE .')
            else:
                self.matriz_covariancia = None
                self.matriz_incerteza = None
                self.matriz_correlacao = None

            self._validar() #validação das incertezas

            # ---------------------------------------------------------------------
            # Graus de liberdade
            # ---------------------------------------------------------------------
            self.gL = gL if len(gL) != 0 else [[100] * self.NE] * self.matriz_estimativa.shape[1]

        def GETListas(self):
            # ---------------------------------------------------------------------
            # Criação dos atributos na forma de LISTAS
            # ---------------------------------------------------------------------
            lista_estimativa = self.matriz_estimativa.transpose().tolist()

            if self.matriz_incerteza is not None:
                lista_incerteza = self.matriz_incerteza.transpose().tolist()
                lista_variancia = (self.matriz_incerteza ** 2).transpose().tolist()
            else:
                lista_incerteza = None
                lista_variancia = None

            return lista_estimativa, lista_incerteza, lista_variancia

        def _validar(self):
            # TODO: Corrigir este teste
            # if (len(gL) != size(estimativa)) and (len(gL) != 0) :
            #		raise ValueError(u'Os graus de liberdade devem ter o mesmo tamanho das estimativas')

            # ---------------------------------------------------------------------
            # VALIDAÇÃO: MATRIZ SINGULAR E INCERTEZA NEGATIVA E ZERO
            # ---------------------------------------------------------------------
            if self.matriz_incerteza is not None:

                for elemento in diag(self.matriz_covariancia):
                    if elemento <= 0.:
                        raise TypeError('The variance of a quantity must be not equal to zero or negative.')

                if not isfinite(cond(self.matriz_covariancia)):
                    raise TypeError('The covariance matrix of the quantity is singular.')

    def _SETdata(self, estimativa, matriz_incerteza=None, matriz_covariancia=None, gL=[], NE=None, dataType=None, **kwargs):

        if not self.__ID_available[0] in self.__ID:
            self.observed = {}
            if dataType is None:
                dataType = 'estimation'
        else:
            if dataType is None:
                dataType = 'validation'

        if not dataType in self._available_dataType:
            raise SyntaxError('The dataType should be:{}'.format(self._available_dataType))

        self.__ID.append(self.__ID_available[0]) #observed

        self.observed[dataType] = self.Dados(estimativa, self.NV,
                                             matriz_incerteza=matriz_incerteza, matriz_covariancia=matriz_covariancia, symbols=self.simbolos,
                                             gL=gL, NE=NE, **kwargs)

    def _SETevaluated(self, estimativa, matriz_incerteza=None, matriz_covariancia=None, gL=[], NE=None, dataType='estimation', **kwargs):

        if hasattr(self, self.__ID_available[0]):
            kwargs['coluna_dumb'] =  self.observed['estimation']._coluna_dumb

        if not self.__ID_available[1] in self.__ID:
            self.evaluated = {}

        self.__ID.append(self.__ID_available[1])
        #self.evaluated = Organizador(estimativa,variancia,gL,tipo,NE)
        self.evaluated[dataType] = self.Dados(estimativa, self.NV,
                                    matriz_incerteza=matriz_incerteza, matriz_covariancia=matriz_covariancia,
                                    gL=gL, NE=NE, **kwargs)

    def _SETresidual(self, estimativa, matriz_incerteza=None, matriz_covariancia=None, gL=[], NE=None, dataType='estimation', **kwargs):

        if hasattr(self, self.__ID_available[0]):
            kwargs['coluna_dumb'] =  self.observed['estimation']._coluna_dumb

        if not self.__ID_available[3] in self.__ID:
            self.residual = {}
            self.estatisticas = {}

        self.__ID.append(self.__ID_available[3])
        # self.residual = Organizador(estimativa,variancia,gL,tipo)

        self.residual[dataType] = self.Dados(estimativa, self.NV,
                                             matriz_incerteza=matriz_incerteza, matriz_covariancia=matriz_covariancia,
                                             gL=gL, NE=NE, **kwargs)

    def _SETparametro(self, estimativa, variancia, regiao, limite_inferior=None, limite_superior=None, **kwargs):

        # --------------------------------------
        # VALIDAÇÃO
        # --------------------------------------
        # estimative
        if not isinstance(estimativa,list):
            raise TypeError(u'The parameter estimative must be a list')

        for elemento in estimativa:
            if not isinstance(elemento,float):
                raise TypeError(u'The elements in the list must be the float type.')

        if len(estimativa) != self.NV:

            raise ValueError(u'It is necessary to inform estimates for all parameters that were defined.')

        # variância
        if variancia is not None:
            if not isinstance(variancia,ndarray):
                raise TypeError(u'The variance must be an array.')
            if not variancia.ndim == 2:
                raise TypeError(u'The variance must be an array with two dimensions.')

            if variancia.shape[0] != variancia.shape[1]:
                raise TypeError(u'The variance must be squared.')

            if variancia.shape[0] != self.NV:
                raise ValueError(u'The size of the covariance matrix must be consistent with the parameter symbols.')

            cont = 0
            for linha in variancia.tolist():
                if linha[cont] <= 0.:
                    raise TypeError('The variance of the parameters must be not equal to zero or negative.')
                cont+=1

        # regiao
        if regiao is not None:
            if not isinstance(regiao,list):
                raise TypeError(u'The region must be a list.')

        # --------------------------------------
        # EXECUÇÃO
        # --------------------------------------
        self.__ID.append(self.__ID_available[2])
        self.estimativa         = estimativa
        self.vetor_estimativa = array(estimativa,ndmin=2).transpose()
        self.matriz_covariancia = variancia
        # Cálculo da matriz de correlação
        if variancia is not None:
            self.matriz_correlacao  = matrizcorrelacao(self.matriz_covariancia)
            self.matriz_incerteza   = (diag(self.matriz_covariancia)**0.5).reshape((1,self.NV),order='F')
        else:
            self.matriz_correlacao  = None
            self.matriz_incerteza   = None

        self.regiao_abrangencia = regiao
        self.limite_superior = limite_superior
        self.limite_inferior = limite_inferior

        # --------------------------------------
        # VALIDAÇÃO
        # --------------------------------------
        if variancia is not None:

            if not isfinite(cond(self.matriz_covariancia)):
                raise TypeError('The covariance matrix of the parameters is singular.')

    def _updateParametro(self,**kwargs):
        u'''
        Método para fazer atualização de informações contidas em Parâmetros.

        =================
        Keyword Arguments
        =================
        Nome dos parâmetros que se deseja atualizar:

        * estimativa
        * matriz_covariancia
        * regiao_abrangencia
        * limite_superior
        * limite_inferior
        '''
        # Estimativa e variância
        estimativa = kwargs.get('estimativa') if kwargs.get('estimativa') is not None else self.estimativa
        variancia = kwargs.get('matriz_covariancia') if kwargs.get('matriz_covariancia') is not None else self.matriz_covariancia

        # limites dos parâmetros
        limite_superior = kwargs.get('limite_superior') if kwargs.get('limite_superior') is not None else self.limite_superior
        limite_inferior = kwargs.get('limite_inferior') if kwargs.get('limite_inferior') is not None else self.limite_inferior

        # região de abrangência
        if self.regiao_abrangencia is None:
            regiao = [] # Caso a região não esteja definida, será utilizado uma lista para permitir o extend
        else:
            regiao = self.regiao_abrangencia

        if kwargs.get('regiao_abrangencia') is not None:
            regiao.extend(kwargs.get('regiao_abrangencia'))
        else:
            regiao = self.regiao_abrangencia

        self._SETparametro(estimativa, variancia, regiao, limite_inferior, limite_superior)

    def labelGraficos(self,add=None, printunit=True):
        u'''
        Método para definição do label dos gráficos relacionado às grandezas.
        
        =======
        Entrada
        =======
        * add (string): texto que se deseja escrever antes da unidade. Deve ser um string
        * printunit (bool): se True, irá colocar a unidade no label
        '''

        # VALIDAÇÃO da variável add
        if (add is not None) and (not isinstance(add,str)):
            raise TypeError(u'The variable add must be a string')
            
        # Definição dos labels: latex ou nomes ou simbolos (nesta ordem)
        label = [None]*len(self.simbolos)

        for z in range(self.NV):

            if self.label_latex[z] is not None:
                label[z] = self.label_latex[z]
            elif self.nomes[z] is not None:
                label[z] = self.nomes[z]
            else:
                label[z] = self.simbolos[z]

            if add is not None:
                label[z] = label[z] +' '+ add

            # Caso seja definido uma unidade, esta será incluída no label
            if printunit:
                if self.unidades[z] is not None:
                    label[z] = label[z] + " / " + self.unidades[z]

        return label

    def _testesEstatisticos(self, Explic, dataType):
        u'''
        Subrotina para realizar testes estatísticos nos resíduos

        =======
        Entrada
        =======
        * Explic: variáveis para explicadores (Independentes/Regressores). Objetivo: avaliar homocedasticidade

        =================
        Testes realizados
        =================
        
        **NORMALIDADE**:
        
        * normaltest: Retorna o pvalor do teste de normalidade. Hipótese nula: a amostra vem de distribuição normal
        * shapiro   : Retorna o pvalor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal
        * anderson  : Retorna o pvalor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal
                      
        * kstest    : Retorna o pvalor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal

        **MÉDIA**:
        
        * ttest_1sam: Retorna o pvalor para média determinada. Hipótese nula: a amostra tem a média determinada
        * ztest : Retorna o pvalor para média determinada. Hipótese nula: a amostra tem a média determinada
       
       **AUTOCORRELAÇÃO**:
        
        *durbin_watson: Teste de autocorrelação Interpretativo. Há duas formas de analisar o resultado:
        1 Forma: Comparação com valores tabelados:
            Podemos tomar a decisão comparando o valor de dw (estatística) com os valores críticos
            dL e dU da Tabela de Durbin-Watson (https://www3.nd.edu/~wevans1/econ30331/Durbin_Watson_tables.pdf) .
            Assim,
            se 0 ≤ dw < dL então rejeitamos H0 (dependência);
            se dL ≤ dw ≤ dU então o teste é inconclusivo;
            se dU < dw < 4-dU então não rejeitamos H0 (independência);
            se 4-dU ≤ dw ≤ 4-dL então o teste é inconclusivo;
            se 4-dL < dw ≤ 4 então rejeitamos H0 (dependência).

            Quando 0<= dw < dL temos evidência de uma correlação positiva. Já quando 4-dL <= dw <= 4 ,
            a correlação é negativa.No caso em que não rejeitamos H0,
            temos que não existe autocorrelação, ou seja, os resíduos são independentes.

        2 Forma: Simplificada
            A estatística de teste é aproximadamente igual a 2 * (1-r) em que r é a autocorrelação das amostras residuais.
            Assim, por r == 0, indicando que não há correlação, a estatística de teste é igual a 2.
            Quanto mais próximo de 0 a estatística, o mais evidências para correlação serial positiva.
            Quanto mais próximo de 4, mais evidências de correlação serial negativa.

        **HOMOCEDÁSTICIDADE**:

        *het_white [1]: Testa se os residual são homocedásticos, foi proposto por Halbert White em 1980.
         Para este teste, a hipótese nula é de que todas as observações têm a mesma variância do erro, ou seja, os erros são homocedásticas.

        *Bresh Pagan:Testa a hipótese de os residual são homocedásticos, recomendado para funções lineares
        **Obs** :  O teste de bresh pagan não é indicado pra formas não lineares de heterocedasticidade
        =====
        SAÍDA
        =====
        Sob a forma de atributo:
        * estatisticas (dict):  p-valor das hipóteses testadas. Para a hipótese nula tida como verdadeira,
        um valor abaixo de (1-PA) nos diz que para PA de confiança pode-se rejeitar essa hipótese.

        OBS: para o teste de durbin_watson , é retornado uma estatística e não p-valores.

        =================
        Referências
        =================
        [1] White, H. (1980). "A Heteroskedasticity-Consistente Covariance Matrix Estimador e um teste direto para Heteroskedasticity". Econometrica 48 (4):. 817-838 JSTOR 1.912.934 . MR 575027 .

        '''
    
        if self.__ID_available[3] in self.__ID: # Testes para os resíduos
            # Variável para salvar os nomes dos testes estatísticos - consulta
            # identifica o nome do teste, e o tipo de resposta (1.0 - float, {} - dicionário, [] - lista)
            # É nessa variável que o Relatório se baseia para obter as informações
            self.__nomesTestes = {'residuo-Normalidade':{'normaltest':1.0,'shapiro':1.0, 'anderson':1.0,'kstest':1.0},
                                  'residuo-Media':{'ttest':1.0, 'ztest': 1.0},
                                  'residuo-Autocorrelacao':{'Durbin Watson':{'estatistica':1.0}, 'Ljung-Box':{'p-valor chi2':1.0,'p-valor Box-Pierce':1.0}},
                                  'residuo-Homocedasticidade':{'white test':{'p-valor multiplicador de Lagrange':1.0,'p-valor Teste F':1.0},'Bresh Pagan':{'p-valor multiplicador de Lagrange':1.0,'p-valor Teste F':1.0}}}
           
            self.__TestesInfo = {'residuo-Autocorrelacao':{'Ljung-Box':{'p-valor chi2':{'H0':'resíduos não são autocorrelacionados'},'p-valor Box-Pierce':{'H0':'resíduos não são autocorrelacionados'}}},'residuo-Normalidade':{'shapiro':{'H0':'resíduos normais'},'normaltest':{'H0':'resíduos normais'},'anderson':{'H0':'resíduos normais'},'kstest':{'H0':'resíduos normais'}}, 'residuo-Media':{'ttest':{'H0':'resíduos com média zero'}, 'ztest':{'H0':'resíduos com média zero'}}, 'residuo-Homocedasticidade':{'white test':{'p-valor multiplicador de Lagrange':{'H0':'resíduos são homocedásticos'},'p-valor Teste F':{'H0':'resíduos são homocedásticos'}}, 'Bresh Pagan':{'p-valor multiplicador de Lagrange':{'H0':'resíduos são homocedásticos'},'p-valor Teste F':{'H0':'resíduos são homocedásticos'}}}}
            pvalor = {}
            for nome in self.simbolos:
                pvalor[nome] = {}

            for i,nome in enumerate(self.simbolos):
                dados = self.residual[dataType].matriz_estimativa[:, i]
        
                # Lista que contém as chamadas das funções de teste:
                if size(dados) < 3: # Se for menor do que 3, não se pode executar o teste de shapiro
                    pnormal=[None, None, normal_ad(dados),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':None, 'shapiro':None, 'anderson':pnormal[2][1],'kstest':pnormal[3][1]}

                elif size(dados) < 20: # Se for menor do que 20 não será realizado no normaltest, pois ele só é válido a partir dste número de dados
                    pnormal=[None, shapiro(dados), normal_ad(dados),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':None, 'shapiro':pnormal[1][1], 'anderson':pnormal[2][1],'kstest':pnormal[3][1]}
                else:
                    pnormal=[normaltest(dados), shapiro(dados), normal_ad(dados),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':pnormal[0][1], 'shapiro':pnormal[1][1], 'anderson':pnormal[2][1],'kstest':pnormal[3][1]}

                # Testes para a média:
                pvalor[nome]['residuo-Media'] = {'ttest':float(ttest_1samp(dados,0.)[1]), 'ztest':ztest(dados, x2=None, value=0, alternative='two-sided', usevar='pooled', ddof=1.0)[1]}
             
                # Testes para a autocorrelação:
                ljungbox = acorr_ljungbox(dados, lags=1, boxpierce=True)
                pvalor[nome]['residuo-Autocorrelacao'] = {'Durbin Watson':{'estatistica':durbin_watson(dados)}, 'Ljung-Box':{'p-valor chi2':float(ljungbox['lb_pvalue'][1]),'p-valor Box-Pierce':float(ljungbox['bp_pvalue'][1])}}
                
                # Testes para a Homocedásticidade:
                pheter = [het_white(dados,insert(Explic, 0, 1, axis=1)),het_breuschpagan(dados,insert(Explic, 0, 1, axis=1))]
                pvalor[nome]['residuo-Homocedasticidade'] = {'white test':{'p-valor multiplicador de Lagrange':pheter[0][1], 'p-valor Teste F':pheter[0][3]},'Bresh Pagan':{'p-valor multiplicador de Lagrange':pheter[1][1],'p-valor Teste F':pheter[1][3]}}
        else:
            raise NameError(u'Statistical tests should be applied for residues only')

        self.estatisticas[dataType] = pvalor

    def Graficos(self, base_path=None, base_dir=None, ID=None, dataType = [], cmap=['k','r','0.75','w','0.75','r','k'], Fig=None):
        u'''
        Método para gerar os gráficos das grandezas, cujas informações só dependam dela.
        
        =======
        Entrada
        =======
        
        * ``base_path`` : caminho onde os gráficos deverão ser salvos
        * base_dir: diretório
        * ``ID``        : Identificação da grandeza.
        * dataType: estimation or validation
        Caso seja None, será feito os gráficos para TODOS os atributos disponíveis.
        * cmap : definição de cores para o pcolor:
         b: blue ;  g: green; r: red;    c: cyan;  m: magenta; z: yellow; k: black; w: white; 0.75: grey
       * Fig (objeto): objetivo Grafico (Graficos.Grafico)
       Funções: 
        * probplot  : Gera um gráfico de probabilidade de dados de exemplo contra os quantis de uma distribuição teórica especificado (a distribuição normal por padrão).
                      Calcula uma linha de melhor ajuste para os dados se "encaixar" é verdadeiro e traça os resultados usando Matplotlib.
        *BOXPLOT    : O boxplot (gráfico de caixa) é um gráfico utilizado para avaliar a distribuição empírica do dados. 
                      O boxplot é formado pelo primeiro e terceiro quartil e pela mediana.
        '''
        self._configFolder = {'plots-subfolder-estimation': 'Estimation',
                              'plots-subfolder-validation': 'Validation',
                              'plots-subfolder-matrizcorrelacao': 'Correlation Matrices',
                              'plots-subfolder-comparacaoresiduo':'Residuals'}
        # ---------------------------------------------------------------------
        # VALIDAÇÃO DAS ENTRADAS
        # ---------------------------------------------------------------------
        if ID is None:
            ID = self.__ID

        if False in [ele in self.__ID_available for ele in ID]:
            raise NameError(u'You inserted an unavailable ID. The available IDs are: ' +','.join(self.__ID_available) + '.')
        try:
            dataType_observed = self.observed.keys()
        except:
            dataType_observed = []
        try:
            dataType_evaluated = self.evaluated.keys()
        except:
            dataType_evaluated = []

        if base_path is None:
            base_path = getcwd()

        if Fig is None:
            Fig = Grafico(dpi=600)
        # ---------------------------------------------------------------------
        # CRIAÇÃO DOS GRÁFICOS
        # ---------------------------------------------------------------------

        base_dir  = sep + 'Grandezas' + sep if base_dir is None else sep + base_dir + sep
        Validacao_Diretorio(base_path,base_dir)

        #Gráfico Pcolor para auto correlação

        #Variável local para alterar a cor do cmap
        cores   = set(['b', 'g', 'r', 'c','m', 'z', 'k', 'w', '0.75'])
        setcmap = set(cmap)
        if not setcmap.issubset(cores):
            raise TypeError('The colors must belong to the list: {}'.format(cores))
           
        cm1 = LinearSegmentedColormap.from_list("Correlacao-cmap",cmap)

        if self.__ID_available[0] in ID: # Gráfico Pcolor para estimação
            #Pastas internas
            for type_data in dataType_observed:
                # ------------------------------------------------------------------------------------
                folder = sep + base_dir + sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)

                # --------------------------------------------------------------------------------------
                listalabel = []
                for elemento in self.labelGraficos(printunit=False):
                    for i in range(self.observed[type_data].NE):
                        listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')
                plot_corr(self.observed[type_data].matriz_correlacao[:self.observed[type_data].NE * self.NV, :self.observed[type_data].NE * self.NV, ], xnames=listalabel, ynames=listalabel,
                          normcolor=True, cmap=cm1)

                savefig(base_path+folder+'_'.join(self.simbolos) + '_' + self.__ID_available[0])
                close()

        if self.__ID_available[1] in ID: # Gráfico Pcolor para evaluated
            listalabel=[]
            for type_data in dataType_evaluated:
                if self.evaluated[type_data].matriz_correlacao is not None:
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    folder =  sep + base_dir + sep + self._configFolder['plots-subfolder-{}'.format(type_data)]+ sep+ self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                    Validacao_Diretorio(base_path, folder)

                    # --------------------------------------------------------------------------------------
                    listalabel = []
                    for elemento in self.labelGraficos(printunit=False):
                        for i in range(self.evaluated[type_data].NE):
                            listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')
                    plot_corr(self.evaluated[type_data].matriz_correlacao[:self.evaluated[type_data].NE * self.NV, :self.evaluated[type_data].NE * self.NV, ], xnames=listalabel, ynames=listalabel,
                              normcolor=True, cmap=cm1)
                    savefig(base_path + folder+'_'.join(self.simbolos) + '_' + self.__ID_available[1])
                    close()

        if (self.__ID_available[2] in ID) and (self.matriz_correlacao is not None): # Gráfico Pcolor para parâmetros
            # Pastas internas
            # ------------------------------------------------------------------------------------
            folder =  sep + base_dir + sep + self._configFolder['plots-subfolder-estimation'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
            Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            plot_corr(self.matriz_correlacao, xnames=self.labelGraficos(printunit=False), ynames=self.labelGraficos(printunit=False), title=u'Matriz de correlação ' + self.__ID_available[2], normcolor=True, cmap=cm1)
            savefig(base_path + folder + self.__ID_available[2])#+'_'+'pcolor')#_matriz-correlacao')
            close()

        if self.__ID_available[3] in ID:
            # BOXPLOT
            intersection = set(dataType).intersection(set(self.residual.keys()))
            for type_data in intersection:
                # Pastas internas
                # ------------------------------------------------------------------------------------
                folder = sep + base_dir + sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep + self._configFolder['plots-subfolder-comparacaoresiduo'] + sep
                Validacao_Diretorio(base_path, folder)

                # --------------------------------------------------------------------------------------
                # checa a variabilidade dos dados, assim como a existência de possíveis outliers
                Fig.boxplot(self.residual[type_data].matriz_estimativa, label_x=self.labelGraficos(printunit=False), label_y='Resíduos')
                Fig.salvar_e_fechar(base_path+folder+'boxplot_'+'residual.png')

                for i,nome in enumerate(self.simbolos):
                    # Gráficos da estimação
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    folder = sep + base_dir + sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep + self.simbolos[i] + sep
                    Validacao_Diretorio(base_path, folder)

                    # ------------------------------------------------------------------------------------
                    dados = self.residual[type_data].matriz_estimativa[:, i]
                    x = arange(1, dados.shape[0]+1, 1)

                    # TENDÊNCIA
                    #Testa a aleatoriedade dos dados, plotando os valores do residuo versus a ordem em que foram obtidos
                    #dessa forma verifica-se há alguma tendência
                    Fig.grafico_dispersao_sem_incerteza(array([min(x), max(x)]), array([mean(dados)] * 2),
                                                        linestyle='-.', color='r', linewidth=2,
                                                        add_legenda=True, corrigir_limites=False, config_axes=False)
                    Fig.grafico_dispersao_sem_incerteza(x, dados, label_x='sample', label_y=u'residual {}'.format(self.labelGraficos()[i]),
                                                        marker='o', linestyle='None')
                    Fig.axes.axhline(0, color='black', lw=1, zorder=1)
                    Fig.set_legenda(['mean'], loc = 'best')
                    Fig.salvar_e_fechar(base_path+folder+'trend_'+'residual.png')

                    # AUTO CORRELAÇÃO
                    #Gera um gráfico de barras que verifica a autocorrelação
                    Fig.autocorr(dados, label_x='lag', label_y=u'autorcorrelation {}'.format(self.labelGraficos(printunit=False)[i]),
                                 normed=True, maxlags=None)
                    Fig.salvar_e_fechar(base_path+folder+'autocorrelation_'+'residuo.png')

                    # HISTOGRAMA
                    #Gera um gráfico de histograma, importante na verificação da pdf
                    Fig.histograma(dados, label_x=u'residual {}'.format(self.labelGraficos()[i]), label_y=u'probability density',
                                   density=True,bins=int(sqrt(dados.shape[0])))
                    Fig.salvar_e_fechar(base_path+folder+'histogram'+'_residual.png')

                    # NORMALIDADE
                    #Verifica se os dados são oriundos de uma pdf normal, o indicativo disto é a obtenção de uma reta
                    Fig.probplot(dados, label_y=u'ordered residual {}'.format(self.labelGraficos(printunit=False)[i]))
                    Fig.salvar_e_fechar(base_path+folder+'probplot'+'_residual.png')

        if (self.__ID_available[0] in ID or self.__ID_available[1] in ID):

            if self.__ID_available[3] in ID:  # remover de ID o resíduo, pois foi tratado separadamente
                ID.remove(self.__ID_available[3])

            base_path = base_path + base_dir
            for type_data in dataType:
                for atributo in ID:
                    y  = eval('self.'+atributo+'["{}"]'.format(type_data)+'.matriz_estimativa')
                    NE = eval('self.'+atributo+'["{}"]'.format(type_data)+'.NE')

                    for i, symb in enumerate(self.simbolos):
                        # Gráficos da estimação
                        # Pastas internas
                        # ------------------------------------------------------------------------------------
                        folder = sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep  + symb + sep
                        Validacao_Diretorio(base_path, folder)

                        # ------------------------------------------------------------------------------------
                        dados = y[:,i]
                        x   = linspace(1,NE,num=NE)
                        #Gráfico em função do numero de observações
                        Fig.grafico_dispersao_sem_incerteza(x, dados, label_x='Amostra',
                                                            label_y=self.labelGraficos(self.__configLabel[atributo])[i],
                                                            marker='o', linestyle=' ')
                        Fig.salvar_e_fechar(base_path + folder + 'trend' + '_' + self.__configLabel[atributo] +'.png')

            if self.__ID_available[0] in ID:

                for type_data in dataType:
                    for i,nome in enumerate(self.simbolos):
                        # Gráficos da estimação
                        # Pastas internas
                        # ------------------------------------------------------------------------------------
                        folder = sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)

                        # ------------------------------------------------------------------------------------
                        dados = self.observed[type_data].matriz_estimativa[:, i]

                        # AUTO CORRELAÇÃO
                        # Gera um gráfico de barras que verifica a autocorrelação
                        Fig.autocorr(dados, label_x='Lag',
                                     label_y=u'autocorrelation {}'.format(self.labelGraficos(printunit=False)[i]),
                                     normed=True, maxlags=None)
                        Fig.salvar_e_fechar(base_path + folder + 'autocorrelacao' + '_observada.png')

            if self.__ID_available[1] in ID:
                for type_data in dataType:
                    for i, nome in enumerate(self.simbolos):
                        # Gráficos da estimação
                        # Pastas internas
                        # ------------------------------------------------------------------------------------
                        folder = sep + self._configFolder['plots-subfolder-{}'.format(type_data)] + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)

                        # ------------------------------------------------------------------------------------
                        dados = self.evaluated[type_data].matriz_estimativa[:, i]

                        # AUTO CORRELAÇÃO
                        # Gera um gráfico de barras que verifica a autocorrelação
                        Fig.autocorr(dados, label_x='Lag',
                                     label_y=u'autocorrelation {}'.format(self.labelGraficos(printunit=False)[i]),
                                     normed=True, maxlags=None)
                        Fig.salvar_e_fechar(base_path + folder + 'autocorrelation_' + 'evaluated.png')

class Grandeza_simplificada:

    def __init__(self, simbolos=None, estimativas=[], incertezas=[]):
        u'''

        '''

        # ------------------------------------------------------------------------------------
        # CRIAÇÃO DE ATRIBUTOS
        # -------------------------------------------------------------------------------------
        # simbolos: usado como referência para a quantidade de variáveis da grandeza

        self.simbolos = simbolos if simbolos is not None else [None]
        self.estimativas = estimativas
        self.incertezas = incertezas

        # Número de grandezas
        self.NV = 0 if simbolos is None else len(self.simbolos)