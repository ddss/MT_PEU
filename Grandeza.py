# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:05:02 2015

@author: danielsantana
"""
# Importação de pacotes de terceiros
from numpy import array, size, diag, linspace, min, max, \
    mean,  std, ndarray, insert, isfinite, arange, sqrt, concatenate

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

    def __init__(self,simbolos,simbolos_incertezas,nomes=None,unidades=None,label_latex=None):
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
        * ``_SETresiduos``     : irá criar o atributo resíduos. Deve ser usado para os resíduos de x

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
        * ``.calculado``    (objeto): objeto Organizador que armazena os valores e incertezas dos dados calculado pelo modelo \
        (vide documentação do mesmo). **só exitirá após execução do método _SETcalculado**
        * ``.residuos``     (objeto): objeto Organizador que armazena os valores e incertezas dos resíduos \
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
        if simbolos  is None:
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
    def __ID_disponivel(self):
        # Todos os ID's disponíveis
        return ['estimacao','predicao','calculado','parametro','residuo']

    @property
    def __configLabel(self):
        # Label para gráficos:
        #       estimacao                           predicao                            calculado
        return {self.__ID_disponivel[0]:'observado',self.__ID_disponivel[1]:'observado',self.__ID_disponivel[2]:'calculado'}

    @property
    def __tipoGraficos(self):
        return ('regiaoAbrangencia', 'grandezas-entrada', 'predicao', 'grandezas-calculadas', 'otimizacao', 'analiseResiduos')

    def __validacaoEntrada(self,simbolos,simbolos_incertezas,nomes,unidades,label_latex):
        u'''
        Validação:
        
        * se os simbolos,simbolos das  incertezas , nome, unidades e label_latex são Listas.
        * se os elementos de simbolos,simbolos das  incertezas, nome, unidades e label_latex são strings (ou caracteres unicode)
        * se os elementos de simbolos possuem caracteres não permitidos (caracteres especiais)
        * se os simbolos são distintos
        * se os tamanhos dos atributos de simbologia, nome, unidades e label_latex são os mesmos.
        '''
        if simbolos_incertezas is not  None :#Se simbolos_incertezas é nada não existe necessidade de validação
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
        for elemento in [simbolos,simbolos_incertezas,nomes,unidades,label_latex]:
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

        def __init__(self,estimativa,NV,matriz_incerteza=None,matriz_covariancia=None,symbols=None,gL=[],NE=None,**kwargs):
            """
            Classe interna para organizar os dados das estimativas e suas respectivas incertezas, disponibilizando-os na forma de matriz, vetores e listas.
            ========
            Entradas
            ========
            * ``estimativa`` (array) : estimativas para as observações das variáveis (na forma de um vetor ou matriz). \
            Caso seja uma matriz, cada coluna contém as estimativas para uma variável. Se for um vetor, as estimativas estão \
            numa única coluna, sendo necessário fornecer a entrada NE.
            * ``NV`` (int): número de variáveis
            * ``matriz_incerteza``  (array) : incerteza para os valores das estimativas. Cada coluna contém a incerteza para os pontos de uma variável.
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
                * ``.matriz_incerteza``  (array): matriz em que cada coluna contém a incerteza de cada ponto de uma certeza variável.
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

            elif NE is not None:

                if estimativa.shape[0] == NV*NE: # Foi informado o vetor estimativa (NExNV,1)
                    self.vetor_estimativa = estimativa
                    self.matriz_estimativa = self.vetor_estimativa.reshape((NE, int(self.vetor_estimativa.shape[0] / NE)),
                                                                           order='F')  # Conversão de vetor para uma matriz
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

    def _SETdadosestimacao(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        self.__ID.append(self.__ID_disponivel[0]) #estimacao

        self.estimacao = self.Dados(estimativa,self.NV,
                                       matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,symbols=self.simbolos,
                                       gL=gL,NE=NE,**kwargs)

        
    def _SETdadosvalidacao(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, self.__ID_disponivel[0]):#estimacao
            kwargs['coluna_dumb'] =  self.estimacao._coluna_dumb

        self.__ID.append(self.__ID_disponivel[1])
        # self.validacao = Organizador(estimativa,variancia,gL,tipo)
        self.predicao = self.Dados(estimativa,self.NV,
                                    matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia, symbols=self.simbolos,
                                    gL=gL,NE=NE,**kwargs)


    def _SETcalculado(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, self.__ID_disponivel[0]):
            kwargs['coluna_dumb'] =  self.estimacao._coluna_dumb

        self.__ID.append(self.__ID_disponivel[2])
        #self.calculado = Organizador(estimativa,variancia,gL,tipo,NE)
        self.calculado = self.Dados(estimativa,self.NV,
                                    matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                    gL=gL,NE=NE,**kwargs)

    def _SETresiduos(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, self.__ID_disponivel[0]):
            kwargs['coluna_dumb'] =  self.estimacao._coluna_dumb

        self.__ID.append( self.__ID_disponivel[4])
        # self.residuos = Organizador(estimativa,variancia,gL,tipo)
        self.residuos = self.Dados(estimativa,self.NV,
                                   matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                   gL=gL,NE=NE,**kwargs)

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
        self.__ID.append(self.__ID_disponivel[3])
        self.estimativa         = estimativa
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
        Evita repetição de uso do método _SETParametros.

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

    def _testesEstatisticos(self,Explic):
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

        *het_white [1]: Testa se os residuos são homocedásticos, foi proposto por Halbert White em 1980.
         Para este teste, a hipótese nula é de que todas as observações têm a mesma variância do erro, ou seja, os erros são homocedásticas.

        *Bresh Pagan:Testa a hipótese de os residuos são homocedásticos, recomendado para funções lineares  
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
    
        if  self.__ID_disponivel[4] in self.__ID: # Testes para os resíduos
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
                dados = self.residuos.matriz_estimativa[:,i]
        
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
                pheter = [het_white(dados,insert(Explic, 0, 1, axis=1)),het_breuschpagan(dados,Explic)]
                pvalor[nome]['residuo-Homocedasticidade'] = {'white test':{'p-valor multiplicador de Lagrange':pheter[0][1], 'p-valor Teste F':pheter[0][3]},'Bresh Pagan':{'p-valor multiplicador de Lagrange':pheter[1][1],'p-valor Teste F':pheter[1][3]}}
        else:
            raise NameError(u'Statistical tests should be applied for residues only')

        self.estatisticas = pvalor

    def Graficos(self,base_path=None,base_dir=None,ID=None,fluxo=None, cmap=['k','r','0.75','w','0.75','r','k'],Fig=None):
        u'''
        Método para gerar os gráficos das grandezas, cujas informações só dependam dela.
        
        =======
        Entrada
        =======
        
        * ``base_path`` : caminho onde os gráficos deverão ser salvos
        * ``ID``        : Identificação da grandeza. Este ID é útil apenas para as grandezas \
        dependentes e independentes, ele identifica para qual atributo os gráficos devem ser avaliados. \
        Caso seja None, será feito os gráficos para TODOS os atributos disponíveis.
        * Fluxo       : identificação do fluxo de trabalho
        * cmap : definição de cores para o pcolor:
         b: blue ;  g: green; r: red;    c: cyan;  m: magenta; y: yellow; k: black; w: white; 0.75: grey
       * Fig (objeto): objetivo Grafico (Graficos.Grafico)
       Funções: 
        * probplot  : Gera um gráfico de probabilidade de dados de exemplo contra os quantis de uma distribuição teórica especificado (a distribuição normal por padrão).
                      Calcula uma linha de melhor ajuste para os dados se "encaixar" é verdadeiro e traça os resultados usando Matplotlib.
        *BOXPLOT    : O boxplot (gráfico de caixa) é um gráfico utilizado para avaliar a distribuição empírica do dados. 
                      O boxplot é formado pelo primeiro e terceiro quartil e pela mediana.
        '''
        self._configFolder = {'plots-subfolder-DadosEstimacao': 'Dados Estimacao',
                              'plots-subfolder-Dadosvalidacao': 'Dados Validacao',
                              'plots-subfolder-matrizcorrelacao': 'Matrizes Correlacao',
                              'plots-subfolder-comparacaoresiduo':'Comparacao entre residuos'}
        # ---------------------------------------------------------------------
        # VALIDAÇÃO DAS ENTRADAS
        # ---------------------------------------------------------------------
        if ID is None:
            ID = self.__ID
        
        if False in [ele in self.__ID_disponivel for ele in ID]:
            raise NameError(u'You inserted an unavailable ID. The available IDs are: '+','.join(self.__ID_disponivel)+'.')

        if base_path is None:
            base_path = getcwd()

        if fluxo is None:
            fluxo = 0

        if Fig is None:
            Fig = Grafico(dpi=60)
        # ---------------------------------------------------------------------
        # CRIAÇÃO DOS GRÁFICOS
        # ---------------------------------------------------------------------

        base_dir  = sep + 'Grandezas' + sep if base_dir is None else sep + base_dir + sep
        Validacao_Diretorio(base_path,base_dir)

        #Gráfico Pcolor para auto correlação

        #Variável local para alterl a cor do cmap
        cores   = set(['b', 'g', 'r', 'c','m', 'y', 'k', 'w', '0.75'])
        setcmap = set(cmap)
        if not setcmap.issubset(cores):
            raise TypeError('The colors must belong to the list: {}'.format(cores))
           
        cm1 = LinearSegmentedColormap.from_list("Correlacao-cmap",cmap)

        if self.__ID_disponivel[0] in ID: # Gráfico Pcolor para estimação
            #Pastas internas
            # ------------------------------------------------------------------------------------
            if fluxo == 0:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            else:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo)+ sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            listalabel=[]
            for elemento in self.labelGraficos(printunit=False):
                for i in range(self.estimacao.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.estimacao.matriz_correlacao, xnames=listalabel,  ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[0],normcolor=True, cmap=cm1)
            savefig(base_path+folder+'_'.join(self.simbolos))#+'_pcolor')_Matriz_de_correlacao')
            close()

        if (self.__ID_disponivel[1] in ID) and (self.predicao.matriz_correlacao is not None): # Gráfico Pcolor para predição
            listalabel=[]
            # Pastas internas
            # ------------------------------------------------------------------------------------
            if fluxo == 0:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            else:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo)+ sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo)+ sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            for elemento in self.labelGraficos(printunit=False):
                for i in range(self.predicao.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.predicao.matriz_correlacao, xnames=listalabel, ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[1],normcolor=True,cmap=cm1)
            savefig(base_path+folder+'observado.png') # +'_'+'pcolor')_matriz-correlacao')
            close()

        if (self.__ID_disponivel[2] in ID) and (self.calculado.matriz_correlacao is not None): # Gráfico Pcolor para calculado
            listalabel=[]
            # Pastas internas
            # ------------------------------------------------------------------------------------
            if fluxo ==0:
                folder = sep + 'Grandezas' + sep+ self._configFolder['plots-subfolder-DadosEstimacao']+sep+ self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-DadosEstimacao']+ sep+ self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            else:
                folder = sep + 'Grandezas' + sep+ self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo)+sep+ self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo)+ sep+ self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            for elemento in self.labelGraficos(printunit=False):
                for i in range(self.calculado.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.calculado.matriz_correlacao, xnames=listalabel, ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[2],normcolor=True,cmap=cm1)
            savefig(base_path+folder+self.__ID_disponivel[2])#+'_'+'pcolor')#_matriz-correlacao')
            close()

        if (self.__ID_disponivel[3] in ID) and (self.matriz_correlacao is not None): # Gráfico Pcolor para parâmetros
            # Pastas internas
            # ------------------------------------------------------------------------------------
            if fluxo == 0:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            else:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep
                Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            plot_corr(self.matriz_correlacao, xnames=self.labelGraficos(printunit=False), ynames=self.labelGraficos(printunit=False), title=u'Matriz de correlação ' + self.__ID_disponivel[3],normcolor=True, cmap=cm1)
            savefig(base_path+folder+self.__ID_disponivel[3])#+'_'+'pcolor')#_matriz-correlacao')
            close()

        if self.__ID_disponivel[4] in ID:
            # BOXPLOT
            # Pastas internas
            # ------------------------------------------------------------------------------------
            if fluxo == 0:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self._configFolder['plots-subfolder-comparacaoresiduo'] + sep
                Validacao_Diretorio(base_path, folder)
            else:
                folder = sep + 'Grandezas' + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self._configFolder['plots-subfolder-matrizcorrelacao'] + sep if base_dir is None else sep + base_dir + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep +self._configFolder['plots-subfolder-comparacaoresiduo'] + sep
                Validacao_Diretorio(base_path, folder)
            # --------------------------------------------------------------------------------------
            # checa a variabilidade dos dados, assim como a existência de possíveis outliers
            Fig.boxplot(self.residuos.matriz_estimativa,label_x=self.labelGraficos(printunit=False), label_y='Resíduos')
            Fig.salvar_e_fechar(base_path+folder+'boxplot_'+'residuo.png')

            base_path = base_path + base_dir
            for i,nome in enumerate(self.simbolos):
                # Gráficos da estimação
                # Pastas internas
                # ------------------------------------------------------------------------------------
                if fluxo == 0:
                    folder = sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self.simbolos[i] + sep
                    Validacao_Diretorio(base_path, folder)
                else:
                    folder = sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self.simbolos[i] + sep
                    Validacao_Diretorio(base_path, folder)
                # ------------------------------------------------------------------------------------
                dados = self.residuos.matriz_estimativa[:,i]
                x = arange(1, dados.shape[0]+1, 1)
    
                # TENDÊNCIA
                #Testa a aleatoriedade dos dados, plotando os valores do residuo versus a ordem em que foram obtidos
                #dessa forma verifica-se há alguma tendência
                Fig.grafico_dispersao_sem_incerteza(array([min(x), max(x)]), array([mean(dados)] * 2),
                                                    linestyle='-.', color='r', linewidth=2,
                                                    add_legenda=True, corrigir_limites=False, config_axes=False)
                Fig.grafico_dispersao_sem_incerteza(x, dados, label_x='Amostra', label_y=u'Resíduos {}'.format(self.labelGraficos()[i]),
                                                    marker='o', linestyle='None')
                Fig.axes.axhline(0, color='black', lw=1, zorder=1)
                Fig.set_legenda(['Média dos resíduos'], loc = 'best')
                Fig.salvar_e_fechar(base_path+folder+'tendencia_'+'residuo.png')

                # AUTO CORRELAÇÃO
                #Gera um gráfico de barras que verifica a autocorrelação
                Fig.autocorr(dados, label_x='Lag', label_y=u'Autocorrelação resíduos {}'.format(self.labelGraficos(printunit=False)[i]),
                             normed=True, maxlags=None)
                Fig.salvar_e_fechar(base_path+folder+'autocorrelacao_'+'residuo.png')

                # HISTOGRAMA
                #Gera um gráfico de histograma, importante na verificação da pdf
                Fig.histograma(dados, label_x=u'Resíduos {}'.format(self.labelGraficos()[i]), label_y=u'Densidade de probabilidade',
                               density=True,bins=int(sqrt(dados.shape[0])))
                Fig.salvar_e_fechar(base_path+folder+'histograma'+'_residuo.png')

                # NORMALIDADE 
                #Verifica se os dados são oriundos de uma pdf normal, o indicativo disto é a obtenção de uma reta 
                Fig.probplot(dados, label_y=u'Valores ordenados resíduos {}'.format(self.labelGraficos(printunit=False)[i]))
                Fig.salvar_e_fechar(base_path+folder+'probplot'+'_residuo.png')

        if (self.__ID_disponivel[0] in ID or self.__ID_disponivel[1] in ID or self.__ID_disponivel[2] in ID):

            if self.__ID_disponivel[4] in ID:  # remover de ID o resíduo, pois foi tratado separadamente
                ID.remove(self.__ID_disponivel[4])

            base_path = base_path + base_dir
            for atributo in ID:
                y  = eval('self.'+atributo+'.matriz_estimativa')
                NE = eval('self.'+atributo+'.NE')

                for i, symb in enumerate(self.simbolos):
                    # Gráficos da estimação
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    if fluxo == 0:
                        folder = sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep  + symb + sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + symb + sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    dados = y[:,i]
                    x   = linspace(1,NE,num=NE)
                    #Gráfico em função do numero de observações
                    Fig.grafico_dispersao_sem_incerteza(x, dados, label_x='Amostra',
                                                        label_y=self.labelGraficos(self.__configLabel[atributo])[i],
                                                        marker='o', linestyle=' ')
                    Fig.salvar_e_fechar(base_path + folder + 'tendencia' + '_' + self.__configLabel[atributo] +'.png')

            if self.__ID_disponivel[0] in ID:

                for i,nome in enumerate(self.simbolos):
                    # Gráficos da estimação
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    if fluxo == 0:
                        folder = sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    dados = self.estimacao.matriz_estimativa[:,i]

                    # AUTO CORRELAÇÃO
                    # Gera um gráfico de barras que verifica a autocorrelação
                    Fig.autocorr(dados, label_x='Lag',
                                 label_y=u'Autocorrelação de {}'.format(self.labelGraficos(printunit=False)[i]),
                                 normed=True, maxlags=None)
                    Fig.salvar_e_fechar(base_path + folder + 'autocorrelacao_' + '_observada.png')

            if self.__ID_disponivel[1] in ID:

                for i, nome in enumerate(self.simbolos):
                    # Gráficos da estimação
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    if fluxo == 0:
                        folder = sep + self._configFolder['plots-subfolder-DadosEstimacao'] + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(fluxo) + sep + self.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    dados = self.predicao.matriz_estimativa[:,i]

                    # AUTO CORRELAÇÃO
                    # Gera um gráfico de barras que verifica a autocorrelação
                    Fig.autocorr(dados, label_x='Lag',
                                 label_y=u'Autocorrelação de {}'.format(self.labelGraficos(printunit=False)[i]),
                                 normed=True, maxlags=None)
                    Fig.salvar_e_fechar(base_path + folder + 'autocorrelacao_' + 'observado.png')
