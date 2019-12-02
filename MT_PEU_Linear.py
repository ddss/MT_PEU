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

# Fim da importação

def Model_1 (param,x, *args):

    # This model is used when there is no independent term calculation

    p = [] ; var = []
    for i in range(param.rows()):
        p = vertcat(p, param[i])

    var = horzcat(var, x)
    a = 1

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
    a=1
    return mtimes(var, p)

class EstimacaoLinear(EstimacaoNaoLinear):
    
    def __init__(self,simbolos_y,simbolos_x,simbolos_param,PA=0.95,projeto='Projeto',**kwargs):
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
        * ``simbolos_y`` (list)     : lista com os simbolos das variáveis y (Não podem haver caracteres especiais)
        * ``simbolos_x`` (list)     : lista com os simbolos das variáveis x (Não podem haver caracteres especiais)
        * ``simbolos_param`` (list) : lista com o simbolos dos parâmetros (Não podem haver caracteres especiais)

        ====================
        Entradas (opcionais)
        ====================
        * ``PA`` (float): probabilidade de abrangência da análise. Deve estar entre 0 e 1. Default: 0.95
        * ``projeto`` (string): nome do projeto (Náo podem haver caracteres especiais)

        **AVISO**:
        * Para cálculo do coeficiente linear, basta que o número de parâmetros seja igual ao número de grandezas
        independentes + 1.

        ==============================
        Keywords (Entradas opcionais):
        ==============================
        
        * ``nomes_x``        (list): lista com os nomes para x 
        * ``unidades_x``     (list): lista com as unidades para x (inclusive em formato LATEX)
        * ``label_latex_x``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``nomes_y``        (list): lista com os nomes para y
        * ``unidades_y``     (list): lista com as unidades para y (inclusive em formato LATEX)
        * ``label_latex_y``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``nomes_param``       (list): lista com os nomes para os parâmetros (inclusive em formato LATEX)
        * ``unidades_param``    (list): lista com as unidades para os parâmetros (inclusive em formato LATEX)
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
        * ``otimiza``              : método para realizar a otimização, com base nos dados fornecidos em setConjunto.
        * ``incertezaParametros``  : método que avalia a incerteza dos parâmetros (Vide documentação do método)
        * ``setConjunto``        : (é opcional para inclusão de dados de validação)
        * ``Predicao``             : método que avalia a predição do modelo e sua incerteza ou utilizando os pontos experimentais ou de \
        validação, se disponível (Vide documentação do método) 
        * ``analiseResiduos``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``graficos``             : método para criação dos gráficos (Vide documentação do método)
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

        if (len(simbolos_param) == len(simbolos_x) + 1):
            # The the independent term will be calculated and Model_2 should be used
            EstimacaoNaoLinear.__init__(self, Model_2, simbolos_y, simbolos_x, simbolos_param, PA, projeto, **kwargs)
        else:
            # The the independent term won't be calculated and Model_1 should be used
            EstimacaoNaoLinear.__init__(self, Model_1, simbolos_y, simbolos_x, simbolos_param, PA, projeto, **kwargs)

        self._EstimacaoNaoLinear__flag.setCaracteristica(['calc_termo_independente'])

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if self.y.NV != 1:
            raise ValueError(u'Está apenas implementado estimação de parâmetros de modelos lineares MISO')

        if (self.parametros.NV != self.x.NV) and (self.parametros.NV != self.x.NV+1):
            raise ValueError(u'O número de parâmetros deve ser: igual ao de grandezas independentes (não é efetuado cálculo do coeficiente linear)'+\
            'OU igual ao número de grandezas independentes + 1 (é calculado o coeficiente linear).')

        self.__coluna_dumb = False # this variable indicates that a column of ones has been added to independent quantities
        # ---------------------------------------------------------------------
        # Definindo se o b será calculado
        # ---------------------------------------------------------------------
        if (self.parametros.NV == self.x.NV+1):
            self._EstimacaoNaoLinear__flag.ToggleActive('calc_termo_independente')
            self.__coluna_dumb = True


    def setConjunto(self,glx=[],gly=[],tipo='estimacao',uxy=None):
        u'''
        Método para incluir os dados de entrada da estimação
        
        =======================
        Entradas (Obrigatórias)
        =======================
        
        * xe        : array com os dados experimentais das variáveis independentes na forma de colunas
        * ux        : array com as incertezas das variáveis independentes na forma de colunas
        * ye        : array com os dados experimentais das variáveis dependentes na forma de colunas
        * uy        : array com as incertezas das variáveis dependentes na forma de colunas
        * tipo      : string que define se os dados são experimentais ou de validação.
        **Aviso**:
        Caso não definidos dados de validação, será assumido os valores experimentais                    
        '''
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self._EstimacaoNaoLinear__controleFluxo.SET_ETAPA('setConjunto')

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------

        # Validação da sintaxe
        if not set([tipo]).issubset(self._EstimacaoNaoLinear__tiposDisponiveisEntrada):
            raise ValueError('A(s) entrada(s) ' + ','.join(
                set([tipo]).difference(self._EstimacaoNaoLinear__tiposDisponiveisEntrada)) + ' não estão disponíveis. Usar: ' + ','.join(
                self._EstimacaoNaoLinear__tiposDisponiveisEntrada) + '.')

        # Validação do número de dados experimentais
        if self._EstimacaoNaoLinear__xtemp.shape[0] != self._EstimacaoNaoLinear__ytemp.shape[0]:
            raise ValueError('Foram inseridos {:d} dados para as grandezas dependentes, mas {:d} para as independentes'.format(self.__ytemp.shape[0],self.__xtemp.shape[0]))

        # ---------------------------------------------------------------------
        # MODIFICAÇÕES DAS MATRIZES DE DADOS
        # ---------------------------------------------------------------------
        coluna_dumb = False
        if self._EstimacaoNaoLinear__flag.info['calc_termo_independente']:
            self._EstimacaoNaoLinear__xtemp = hstack((self._EstimacaoNaoLinear__xtemp, ones((shape(self._EstimacaoNaoLinear__xtemp)[0], 1))))
            self._EstimacaoNaoLinear__uxtemp = hstack((self._EstimacaoNaoLinear__uxtemp, ones((shape(self._EstimacaoNaoLinear__uxtemp)[0], 1))))
            coluna_dumb = True

        if tipo == 'estimacao':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadosestimacao')
            if self._EstimacaoNaoLinear__controleFluxo.FLUXO_ID != 0:
                self._EstimacaoNaoLinear__controleFluxo.reiniciar()
                if self.__flag.info['dadospredicao']:
                    warn('O fluxo foi reiniciado, faz-se necessário incluir novos dados de validação.')
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados experimentais nas variáveis.
            try:
                self.x._SETdadosestimacao(estimativa=self._EstimacaoNaoLinear__xtemp,matriz_incerteza=self._EstimacaoNaoLinear__uxtemp,gL=glx,coluna_dumb=self.__coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto de estimação da grandeza X: {}'.format(erro))

            try:
                self.y._SETdadosestimacao(estimativa=self._EstimacaoNaoLinear__ytemp,matriz_incerteza=self._EstimacaoNaoLinear__uytemp,gL=gly)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto de estimação da grandeza Y: {}'.format(erro))

        if tipo == 'predicao':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadospredicao')
            self._EstimacaoNaoLinear__controleFluxo.reiniciarParcial()

            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            try:
                self.x._SETdadosvalidacao(estimativa=self._EstimacaoNaoLinear__xtemp,matriz_incerteza=self._EstimacaoNaoLinear__uxtemp,gL=glx,coluna_dumb=coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto validação de X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=self._EstimacaoNaoLinear__ytemp,matriz_incerteza=self._EstimacaoNaoLinear__uytemp,gL=gly)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto validação de Y: {}'.format(erro))

        if self._EstimacaoNaoLinear__flag.info['dadospredicao'] == False:
            # Caso gerarEntradas seja executado somente para os dados experimentais,
            # será assumido que estes são os dados de validação, pois todos os cálculos 
            # de predição são realizados para os dados de validação.
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            try:
                self.x._SETdadosvalidacao(estimativa=self._EstimacaoNaoLinear__xtemp,matriz_incerteza=self._EstimacaoNaoLinear__uxtemp,gL=glx,coluna_dumb=self.__coluna_dumb)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto validação de X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=self._EstimacaoNaoLinear__ytemp,matriz_incerteza=self._EstimacaoNaoLinear__uytemp,gL=gly)
            except Exception as erro:
                raise RuntimeError('Erro na criação do conjunto validação de Y: {}'.format(erro))

        # Transformando variáveis temporárias ( xtemp, uxtemp, ytemp, uytemp) em listas vazias
        self.EstimacaoNaoLinear__xtemp = None
        self.EstimacaoNaoLinear__uxtemp = None
        self.EstimacaoNaoLinear__ytemp = None
        self.EstimacaoNaoLinear__uytemp = None


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
            raise TypeError(u'Para executar a otimização, faz-se necessário dados experimentais.')

        if self._EstimacaoNaoLinear__controleFluxo.SETparametro:
            raise TypeError(u'O método otimizacao não pode ser executado com SETparametro.')

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

    def incertezaParametros(self, preencherregiao=True, parametersReport = True, **kwargs):
        u'''
        Método para avaliar a região de abrangência dos parâmetros.

        **Observação**:
        A matriz de covariância dos parâmetros é calculada juntamente com a otimização, por ser parte constituinte da solução analítica. Entretanto,
        caso o método SETparametros seja executado e neste não seja definida a matriz de covariância, ela é calculada.

        ==================
        Entradas opcionais
        ==================

        * preencherregiao (bool): identifica se será executado algoritmo para preenchimento da região de abrangência.
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
        if preencherregiao and self.parametros.NV != 1:
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