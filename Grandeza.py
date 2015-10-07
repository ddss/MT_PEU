# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:05:02 2015

@author: danielsantana
"""
# Importação de pacotes de terceiros
from numpy import array, transpose ,size, diag, linspace, min, max, \
    mean,  std, amin, amax, ndarray, insert, nan, correlate, isfinite

from numpy.linalg import cond

from statsmodels.stats.weightstats import ztest
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breushpagan, het_white, normal_ad
from statsmodels.stats.stattools import durbin_watson
from statsmodels.graphics.correlation import plot_corr

from scipy.stats import normaltest, shapiro, ttest_1samp, kstest, probplot

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot, stem, acorr
    
from matplotlib.colors import LinearSegmentedColormap
    

from os import getcwd, sep

# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import Validacao_Diretorio, matrizcorrelacao

class Grandeza:

    def __init__(self,simbolos,nomes=None,unidades=None,label_latex=None):
        u'''
        Classe para organizar as características das Grandezas:
                
        =======
        Entrada
        =======
        
        **OBRIGATÓRIO**:
        
        * ``simbolos`` (list)   : deve ser uma lista contendo os símbolos, na ordem de entrada de cada variável   
        
        **OPCIONAL**:
        
        * ``nomes``       (list) : deve ser uma lisra contendo o nome das variáveis
        * ``unidades``    (list) : deve ser uma lista contendo as unidades das variáveis
        * ``label_latex`` (list) : deve ser uma lista contendo os símbolos em formato LATEX
        
        =======
        Métodos        
        =======

        **DEFINICIONAIS** - Usado para criação de atributos:
        
        * ``_SETexperimental`` : irá criar o atributo experimental. Deve ser usado se se tratar de dados experimentais
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

        * ``.experimental`` (objeto): objeto Organizador que armazena os valores e incertezas dos dados experimentais \
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
        if simbolos is None:
            raise NameError('Os símbolos das grandezas são obrigatórios')

        self.__validacaoEntrada(simbolos,nomes,unidades,label_latex)

        # ------------------------------------------------------------------------------------
        # CRIAÇÃO DE ATRIBUTOS
        # -------------------------------------------------------------------------------------
        # simbolos: usado como referência para a quantdade de variáveis da grandeza
        self.simbolos    = simbolos

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
        self.__ID_disponivel = ['experimental','validacao','calculado','parametro','residuo'] # Todos os ID's disponíveis

    def __validacaoEntrada(self,simbolos,nomes,unidades,label_latex):
        u'''
        Validação:
        
        * se os simbolos, nome, unidades e label_latex são Listas.
        * se os elementos de simbolos, nome, unidades e label_latex são strings (ou caracteres unicode)
        * se os elementos de simbolos possuem caracteres não permitidos (caracteres especiais)
        * se os simbolos são distintos
        * se os tamanhos dos atributos de simbologia, nome, unidades e label_latex são os mesmos.
        '''
        # Verificação se nomes, unidade e label_latex são listas
        for elemento in [simbolos,nomes,unidades,label_latex]:
            if elemento is not None:
                if not isinstance(elemento,list):
                    raise TypeError('A simbologia, nomes, unidades e label_latex de uma grandeza devem ser LISTAS.')
                # verificação se os elementos são strings
                for value in elemento:
                    if value is not None:
                        if not isinstance(value,str) and not isinstance(value,unicode):
                            raise TypeError('Os elementos de simbolos, nomes, unidades e label_latex devem ser STRINGS.')

        # Verificação se os símbolos possuem caracteres especiais
        for simb in simbolos:
            if set('[~!@#$%^&*()+{}":;\']+$').intersection(simb):
                raise NameError('Os simbolos das grandezas não podem ter caracteres especiais. Simbolo incorreto: '+simb)

        # Verificação se os símbolos são distintos
        # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
        if len(set(simbolos)) != len(simbolos):
            raise NameError('Os símbolos de cada grandeza devem ser distintos.')

       # Verificação se nomes, unidade e label_latex possuem mesmo tamanho
        for elemento in [nomes,unidades,label_latex]:
            if elemento is not None:
                if len(elemento) != len(simbolos):
                    raise ValueError('A simbologia, nomes, unidades e label_latex de uma grandeza devem ser listas de MESMO tamanho.')

    class Dados:

        def __init__(self,estimativa,NV,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):
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
                raise TypeError(u'Os dados de entrada precisam ser arrays.')

            if matriz_covariancia is not None and matriz_incerteza is not None:
                raise SyntaxError(u'Apenas uma entre a matriz_covariancia e matriz_incerteza deve ser definida')

            if matriz_covariancia is not None:
                if not isinstance(matriz_covariancia, ndarray):
                    raise TypeError(u'Os dados de entrada precisam ser arrays.')

            if matriz_incerteza is not None:
                if not isinstance(matriz_incerteza, ndarray):
                    raise TypeError(u'Os dados de entrada precisam ser arrays.')

            if not isinstance(gL, list):
                raise TypeError(u'os graus de liberdade precisam ser listas')

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
                    (self.matriz_estimativa.shape[0] * self.matriz_estimativa.shape[1], 1),
                    order='F')  # conversão de matriz para vetor

            elif NE is not None:

                if estimativa.shape[0] == NV*NE: # Foi informado o vetor estimativa (NExNV,1)
                    self.vetor_estimativa = estimativa
                    self.matriz_estimativa = self.vetor_estimativa.reshape((NE, self.vetor_estimativa.shape[0] / NE),
                                                                           order='F')  # Conversão de vetor para uma matriz
                else:
                    raise ValueError(u'O tamanho do vetor estimativa deve ser igual ao número de variáves vezes número de dados')
            else:
                raise ValueError(u'A estimativa foi fornecida na forma de um vetor. NE deve ser especificado.')
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
                    self.matriz_incerteza = diag(self.matriz_covariancia**0.5).reshape(
                        (NE, self.matriz_estimativa.shape[1]), order='F')
                    self.matriz_correlacao = matrizcorrelacao(self.matriz_covariancia)
                else:
                    raise ValueError(u'É necessário definir o valor de NE.')
            else:
                self.matriz_covariancia = None
                self.matriz_incerteza = None

            self._validar() # validação das incertezas

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

                if not isfinite(cond(self.matriz_covariancia)):
                    raise TypeError('A matriz de covariância da grandeza é singular.')

                for elemento in diag(self.matriz_covariancia):
                    if elemento <= 0.:
                        raise TypeError('A variância de uma grandeza não pode ser zero ou assumir valores negativos.')

    def _SETexperimental(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        self.__ID.append('experimental')
        # self.experimental = Organizador(estimativa,variancia,gL,tipo)
        self.experimental = self.Dados(estimativa,self.NV,
                                       matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                       gL=gL,NE=NE,**kwargs)

    def _SETvalidacao(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, 'experimental'):
            kwargs['coluna_dumb'] =  self.experimental._coluna_dumb

        self.__ID.append('validacao')
        # self.validacao = Organizador(estimativa,variancia,gL,tipo)
        self.validacao = self.Dados(estimativa,self.NV,
                                    matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                    gL=gL,NE=NE,**kwargs)

    def _SETcalculado(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, 'experimental'):
            kwargs['coluna_dumb'] =  self.experimental._coluna_dumb

        self.__ID.append('calculado')
        #self.calculado = Organizador(estimativa,variancia,gL,tipo,NE)
        self.calculado = self.Dados(estimativa,self.NV,
                                    matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                    gL=gL,NE=NE,**kwargs)

    def _SETresiduos(self,estimativa,matriz_incerteza=None,matriz_covariancia=None,gL=[],NE=None,**kwargs):

        if hasattr(self, 'experimental'):
            kwargs['coluna_dumb'] =  self.experimental._coluna_dumb

        self.__ID.append('residuo')
        # self.residuos = Organizador(estimativa,variancia,gL,tipo)
        self.residuos = self.Dados(estimativa,self.NV,
                                   matriz_incerteza=matriz_incerteza,matriz_covariancia=matriz_covariancia,
                                   gL=gL,NE=NE,**kwargs)

    def _SETparametro(self, estimativa, variancia, regiao,limite_inferior=None,limite_superior=None):

        # --------------------------------------
        # VALIDAÇÃO
        # --------------------------------------

        # estimativa
        if not isinstance(estimativa,list):
            raise TypeError(u'A estimativa para os parâmetros precisa ser uma lista')

        for elemento in estimativa:
            if not isinstance(elemento,float):
                raise TypeError(u'A estimativa precisa ser uma lista de floats.')

        if len(estimativa) != self.NV:
            raise ValueError(u'Devem ser fornecidas estimativas para todos os parãmetros definidos em símbolos')

        # variância
        if variancia is not None:
            if not isinstance(variancia,ndarray):
                raise TypeError(u'A variância precisa ser um array.')
            if not variancia.ndim == 2:
                raise TypeError(u'A variância precisa ser um array com duas dimensões.')

            if variancia.shape[0] != variancia.shape[1]:
                raise TypeError(u'A variância precisa ser quadrada.')

            if variancia.shape[0] != self.NV:
                raise ValueError(u'A dimensão da matriz de covariância deve ser coerente com os simbolos dos parâmetros')

            cont = 0
            for linha in variancia.tolist():
                if linha[cont] <= 0.:
                    raise TypeError('A variância dos parâmetros não pode ser zero ou assumir valores negativos')
                cont+=1

        # regiao
        if regiao is not None:
            if not isinstance(regiao,list):
                raise TypeError(u'A regiao precisa ser uma lista.')

        # --------------------------------------
        # EXECUÇÃO
        # --------------------------------------

        self.__ID.append('parametro')           
        self.estimativa         = estimativa
        self.matriz_covariancia = variancia
        # Cálculo da matriz de correlação
        if variancia is not None:
            self.matriz_correlacao  = matrizcorrelacao(self.matriz_covariancia)
            self.matriz_incerteza   = diag(self.matriz_covariancia**0.5).reshape((1,self.NV),order='F')
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
                raise TypeError('A matriz de covariância dos parâmetros é singular.')

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
        * add: texto que se deseja escrever antes da unidade. Deve ser um string
        '''
        # VALIDAÇÃO Da variável add
        if (add is not None) and (not isinstance(add,str)):
            raise TypeError(u'A variável add deve ser um string')
            
        # Definição dos labels: latex ou nomes ou simbolos (nesta ordem)
        label = [None]*len(self.nomes)
        if printunit is False:
            
            for z in xrange(len(self.nomes)):
    
                if self.label_latex[z] is not None:
                    label[z] = self.label_latex[z]
                elif self.nomes[z] is not None:
                    label[z] = self.nomes[z]
                elif self.simbolos[z] is not None:
                    label[z] = self.simbolos[z]
    
                if add is not None:
                    label[z] = label[z] +' '+ add
                    
        else:
        
            for z in xrange(len(self.nomes)):
    
                if self.label_latex[z] is not None:
                    label[z] = self.label_latex[z]
                elif self.nomes[z] is not None:
                    label[z] = self.nomes[z]
                elif self.simbolos[z] is not None:
                    label[z] = self.simbolos[z]
    
                if add is not None:
                    label[z] = label[z] +' '+ add
                
                # Caso seja definido uma unidade, esta será incluída no label
                if self.unidades[z] is not None:
                    label[z] = label[z]+' '+"/"+self.unidades[z]

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
    
        if 'residuo' in self.__ID: # Testes para os resíduos
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
                pvalor[nome]['residuo-Autocorrelacao'] = {'Durbin Watson':{'estatistica':durbin_watson(dados)}, 'Ljung-Box':{'p-valor chi2':float(ljungbox[1]),'p-valor Box-Pierce':float(ljungbox[3])}}
                
                # Testes para a Homocedásticidade:
                pheter = [het_white(dados,insert(Explic, 0, 1, axis=1)),het_breushpagan(dados,Explic)]
                pvalor[nome]['residuo-Homocedasticidade'] = {'white test':{'p-valor multiplicador de Lagrange':pheter[0][1], 'p-valor Teste F':pheter[0][3]},'Bresh Pagan':{'p-valor multiplicador de Lagrange':pheter[1][1],'p-valor Teste F':pheter[1][3]}}
        else:
            raise NameError(u'Os testes estatísticos são válidos apenas para o resíduos')

        self.estatisticas = pvalor
        
    def __construtor_graficos(self, Xlabel, Ylabel,x,y,base_path,base_dir,fluxo, ID, tipo, **kwargs):
        u'''
         Método para gerar os gráficos.
        =======
        Entrada
        =======
        * ``Xlabel`` : Label do eixo x \
        * ``Ylabel``  : Label do eixo y \
        * ``x`` : conjuntos de dados pro eixo x, caso tenha.\
        * ``y ``  : conjuntos de dados pro eixo y, caso tenha.  \
        *  ``base_path`` : caminho onde os gráficos deverão ser salvos    \
        *  ``ID``       : Identificação da grandeza. Este ID é útil apenas para as grandezas \
        dependentes e independentes, ele identifica para qual atributo os gráficos devem ser avaliados. \
        Caso seja None, será feito os gráficos para TODOS os atributos disponíveis. \
        *  ``Fluxo``     : identificação do fluxo de trabalho   \
        * ``Tipo``  : tipo de gráfico a ser feito, deve ser string, opções:'boxplot': usa boxplot, 'hist': usa hist, 'tendencia: usa plot normal', 'autocorrelacao': usa acorr, 'probplot'
        *`` kwargs`` : qualquer kwarg extra do gráfico pode ser passada para o construtor, desde que seja propriedade do gráfico e que esteja na lista de argumentos esperados
        '''
    
        kwargsdict = {}
        #obs: adicionar qualquer kwarg de um gráfico em especifico que possua tal propriedade nesta lista, exemplo: o plot normal possui a kwarg marker, logo marker teve que ser passado
        expected_args = ["legenda", "media","sym", "usevlines", "normed", "maxlags", 'label', 'marker', 'ls',\
        'xnames', 'ynames', 'title', 'normcolor', 'cmap']
        for key in kwargs.keys():
            if key in expected_args:
                kwargsdict[key] = kwargs[key]
            else:
                raise Exception("Argumentos esperados ".format(expected_args))
        fig = figure()
        ax=fig.add_subplot(1,1,1)
        legenda=False
        media=False
        if kwargs.has_key('media') or kwargs.has_key('legenda'):
             media=kwargs.pop('media')
             legenda= kwargs.pop('legenda')
        if tipo== 'boxplot':
           boxplot(y, **kwargs)
           #ylabel deve ser passado como uma lista com os nomes de cada conjunto de dados que vai construir o boxplot
           ax.set_xticklabels(Ylabel)
           ax.yaxis.grid(color='gray', linestyle='dashed')
           ax.xaxis.grid(color='gray', linestyle='dashed')
        if tipo=='autocorrelacao':
           acorr(y, **kwargs)
           xlim(0,len(y))
           ax.yaxis.grid(color='gray', linestyle='dashed')
           ax.xaxis.grid(color='gray', linestyle='dashed')
           xlabel(Xlabel)
           ylabel(Ylabel)
        if tipo=='tendencia':
#           plot(x,y, 'o',label=u'Ordem de coleta')
#           plot(x,y, **kwargs)
           plot(x, y,**kwargs)
           ax.axhline(0, color='black', lw=1)
           ax.axvline(0,color='black', lw=1)
           if media==True:
                  plot(linspace(0,len(y)+1,num=len(y+1)),[mean(y)]*len(y), 'r', label=u'Valor médio')
#                 plot(x,[mean(y)]*size(y), kwargs['label'][1])     
          
            # obtençao do tick do grafico
           # eixo x
           label_tick_x   = ax.get_xticks().tolist()
           tamanho_tick_x = (label_tick_x[1] - label_tick_x[0])/2
           # eixo y
           label_tick_y = ax.get_yticks().tolist()
           tamanho_tick_y = (label_tick_y[1] - label_tick_y[0])/2
           # Modificação do limite dos gráficos
           xmin   = min(x)     - tamanho_tick_x
           xmax   = max(x)     + tamanho_tick_x
           ymin   = min(y) - tamanho_tick_y
           ymax   = max(y) + tamanho_tick_y
           xlim(xmin,xmax)
           ylim(ymin,ymax)
           xlabel(Xlabel)
           ylabel(Ylabel)
           ax.yaxis.grid(color='gray', linestyle='dashed')
           ax.xaxis.grid(color='gray', linestyle='dashed')
        if tipo=='hist':
           hist(y, normed=True)
           xlabel(Xlabel)
           ylabel(Ylabel)
        if tipo=='probplot':
          res = probplot(y, dist='norm', sparams=(mean(y),std(y,ddof=1)))
          if not (nan in res[0][0].tolist() or nan in res[0][1].tolist() or nan in res[1]):
             plot(res[0][0], res[0][1], 'o', res[0][0], res[1][0]*res[0][0] + res[1][1])
             xlabel(Xlabel)
             ylabel(Ylabel)
             xmin = amin(res[0][0])
             xmax = amax(res[0][0])
             ymin = amin(y)
             ymax = amax(y)
             posx = xmin + 0.70 * (xmax - xmin)
             posy = ymin + 0.01 * (ymax - ymin)
             text(posx, posy, "$R^2$=%1.4f" % res[1][2])    
             ax.yaxis.grid(color='gray', linestyle='dashed')
             ax.xaxis.grid(color='gray', linestyle='dashed')  
#        if tipo== 'pcolor':
#           plot_corr(y,   ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[0],normcolor=True, cmap=cm1)

        if legenda==True:
              legend(loc='best')      
        fig.savefig(base_path+base_dir+'{}_fl'.format(ID[0])+str(fluxo)+'_{}'.format(tipo))        
        close()              
    def Graficos(self,base_path=None,base_dir=None,ID=None,fluxo=None, cmap=['k','r','0.75','w','0.75','r','k']):
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
       
       Funções: 
        * probplot  : Gera um gráfico de probabilidade de dados de exemplo contra os quantis de uma distribuição teórica especificado (a distribuição normal por padrão).
                      Calcula uma linha de melhor ajuste para os dados se "encaixar" é verdadeiro e traça os resultados usando Matplotlib.
        *BOXPLOT    : O boxplot (gráfico de caixa) é um gráfico utilizado para avaliar a distribuição empírica do dados. 
                      O boxplot é formado pelo primeiro e terceiro quartil e pela mediana.
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO DAS ENTRADAS
        # ---------------------------------------------------------------------
        if ID is None:
            ID = self.__ID
        
        if False in [ele in self.__ID_disponivel for ele in ID]:
            raise NameError(u'Foi inserido uma ID indiponível. IDs disponíveis: '+','.join(self.__ID_disponivel)+'.')

        if base_path is None:
            base_path = getcwd()

        if fluxo is None:
            fluxo = 0

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
            raise TypeError('As cores devem pertencer à lista: {}'.format(cores))
           
        cm1 = LinearSegmentedColormap.from_list("Correlacao-cmap",cmap)   

        if self.__ID_disponivel[0] in ID: # Gráfico Pcolor para experimental
            listalabel=[]
            for elemento in self.labelGraficos(printunit=False):
                for i in xrange(self.experimental.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.experimental.matriz_correlacao, xnames=listalabel,  ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[0],normcolor=True, cmap=cm1)
            savefig(base_path+base_dir+self.__ID_disponivel[0]+'_fl'+str(fluxo)+'_'+'pcolor_matriz-correlacao')
            close()

        if self.__ID_disponivel[1] in ID: # Gráfico Pcolor para validação
            listalabel=[]
            for elemento in self.labelGraficos(printunit=False):
                for i in xrange(self.validacao.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.validacao.matriz_correlacao, xnames=listalabel, ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[1],normcolor=True,cmap=cm1)
            savefig(base_path+base_dir+self.__ID_disponivel[1]+'_fl'+str(fluxo)+'_'+'pcolor_matriz-correlacao')
            close()

        if self.__ID_disponivel[2] in ID: # Gráfico Pcolor para calculado
            listalabel=[]
            for elemento in self.labelGraficos(printunit=False):
                for i in xrange(self.calculado.NE):
                    listalabel.append(elemento + r'$_{'+'{}'.format(i+1)+'}$')

            plot_corr(self.calculado.matriz_correlacao, xnames=listalabel, ynames=listalabel, title=u'Matriz de correlação ' + self.__ID_disponivel[2],normcolor=True,cmap=cm1)
            savefig(base_path+base_dir+self.__ID_disponivel[2]+'_fl'+str(fluxo)+'_'+'pcolor_matriz-correlacao')
            close()

        if self.__ID_disponivel[3] in ID: # Gráfico Pcolor para parâmetros

            plot_corr(self.matriz_correlacao, xnames=self.labelGraficos(printunit=False), ynames=self.labelGraficos(printunit=False), title=u'Matriz de correlação ' + self.__ID_disponivel[3],normcolor=True, cmap=cm1)
            savefig(base_path+base_dir+self.__ID_disponivel[3]+'_fl'+str(fluxo)+'_'+'pcolor_matriz-correlacao')
            close()

        if 'residuo' in ID:
            # BOXPLOT
            #checa a variabilidade dos dados, assim como a existência de possíveis outliers
            self.__construtor_graficos(None,self.labelGraficos(printunit=False),None, self.residuos.matriz_estimativa,base_path,base_dir,fluxo, ID, 'boxplot', sym='.k')
            
            base_path = base_path + base_dir
            for i,nome in enumerate(self.simbolos):
                # Gráficos da estimação
                base_dir = sep + self.simbolos[i] + sep
                Validacao_Diretorio(base_path,base_dir)
                dados = self.residuos.matriz_estimativa[:,i]
                x=linspace(0, len(dados), num=len(dados))
    
                # TENDÊNCIA
                #Testa a aleatoriedade dos dados, plotando os valores do residuo versus a ordem em que foram obtidos
                #dessa forma verifica-se há alguma tendência
                self.__construtor_graficos('Ordem de Coleta',self.labelGraficos()[i],x,dados,base_path,base_dir,fluxo, ID, 'tendencia', legenda=True, media=True, label='Pontos', marker='o', ls='None')
        
                # AUTO CORRELAÇÃO
                #Gera um gráfico de barras que verifica a autocorrelação
                self.__construtor_graficos('Lag', u'Autocorrelação de {}'.format(self.labelGraficos(printunit=False)[i]),None,dados,base_path,base_dir,fluxo, ID,'autocorrelacao' , usevlines=True,normed=True, maxlags=None)
                 
               
                # HISTOGRAMA                
                #Gera um gráfico de histograma, importante na verificação da pdf
                self.__construtor_graficos(self.labelGraficos()[i], u'Frequência',None,dados,base_path,base_dir,fluxo, ID,'hist')

                # NORMALIDADE 
                #Verifica se os dados são oriundos de uma pdf normal, o indicativo disto é a obtenção de uma reta 
                self.__construtor_graficos('Quantis', 'Valores ordenados de {}'.format(self.labelGraficos(printunit=False)[i]),None,dados,base_path,base_dir,fluxo, ID,'probplot')         
              
                
        if ('experimental' in ID or 'validacao' in ID or 'calculado' in ID):
                
            if 'residuo' in ID:  # remover de ID o resíduo, pois foi tratado separadamente
                ID.remove('residuo')

            base_path = base_path + base_dir
            for atributo in ID:
                y  = eval('self.'+atributo+'.matriz_estimativa')
                NE = eval('self.'+atributo+'.NE')

                for i,symb in enumerate(self.simbolos):
                    # Gráficos da estimação
                    base_dir = sep + symb + sep
                    Validacao_Diretorio(base_path,base_dir)
                    dados = y[:,i]
                    x   = linspace(1,NE,num=NE)
                    #Gráfico em função do numero de observações
                    self.__construtor_graficos(u'Número de pontos',self.labelGraficos(atributo)[i],x,dados,base_path,base_dir,fluxo, ID, 'tendencia', marker='o', ls='None')

            if 'experimental' in ID:

                for i,nome in enumerate(self.simbolos):
                    # Gráficos da estimação

                    base_dir = sep + self.simbolos[i] + sep
                    Validacao_Diretorio(base_path,base_dir)
                    dados = self.experimental.matriz_estimativa[:,i]
                    

                    # AUTO CORRELAÇÃO
                    # Gera um gráfico de barras que verifica a autocorrelação
                    self.__construtor_graficos('Lag', u'Autocorrelação de {}'.format(self.labelGraficos(printunit=False)[i]),None,dados,base_path,base_dir,fluxo, ID,'autocorrelacao', usevlines=True,normed=True, maxlags=None)
                    
            if 'validacao' in ID:

                for i,nome in enumerate(self.simbolos):
                    # Gráficos da estimação
                    base_dir = sep + self.simbolos[i] + sep
                    Validacao_Diretorio(base_path,base_dir)

                    dados = self.validacao.matriz_estimativa[:,i]

                    # AUTO CORRELAÇÃO
                    # Gera um gráfico de barras que verifica a autocorrelação
                    self.__construtor_graficos('Lag', u'Autocorrelação de {}'.format(self.labelGraficos(printunit=False)[i]),None,dados,base_path,base_dir,fluxo, ID,'autocorrelacao', usevlines=True,normed=True, maxlags=None)