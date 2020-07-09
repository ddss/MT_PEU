 # -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author(es): Daniel, Francisco, Anderson, Victor, Leonardo, Regiane
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# --------------------------------------------------------------------
# IMPORTAÇÃO DE PACOTES DE TERCEIROS
# ---------------------------------------------------------------------
# Cálculos científicos
from numpy import array, size, linspace, min, max, copy,\
    mean, ones, ndarray, nanmax, nanmin, arange, transpose, delete, concatenate, linalg,inf
from numpy.core.multiarray import ndarray
from numpy.random import uniform, triangular
from scipy.stats import f, t, chi2
from scipy.special import factorial
from numpy.linalg import inv
from math import floor, log10
#from threading import Thread
from scipy import transpose, dot, concatenate, matrix
from scipy.optimize import  minimize, rosen, rosen_der
# Pacotes do sistema operacional
from os import getcwd, sep
from casadi import MX,DM,vertcat,horzcat,nlpsol,sum1,jacobian,hessian,mtimes,inv as inv_cas, diag,Function
# Exception Handling
from warnings import warn

# Sistema
#TODO: CORRIGIR ENCONDING
#import sys
#sys.setdefaultencoding("utf-8") # Forçar o sistema utilizar o coding utf-8

# ---------------------------------------------------------------------------
# IMPORTAÇÃO DE SUBROTINAS PRÓPRIAS E ADAPTAÇÕES (DESENVOLVIDAS PELO GI-UFBA)
# ---------------------------------------------------------------------------
from Grandeza import Grandeza
from subrotinas import Validacao_Diretorio, eval_cov_ellipse, vetor_delta,\
    matriz2vetor, WLS
from Graficos import Grafico
from Relatorio import Report
from Flag import flag

class EstimacaoNaoLinear:

    class Fluxo:

        def __init__(self):
            u"""
            Classe voltada para controlar o fluxo de etapas de EstimacaoNaoLinear

            ==========
            ATRIBUTOS:
            ==========

            * Cada atributo representa uma etapa da classe EstimacaoNaoLinear e assumem 2 valores:

                * 0 : o método NÃO foi executado
                * 1 : o método foi executado

            ========
            MÉTODOS:
            ========

            * .SET_ETAPA: método para, na execução da Estimacao, indicar qual etapa está sendo avaliada.O método irá avaliar
            se as etapas predecessoras foram executadas.
            * .reiniciar: reinicia o fluxo. Atribui 0 a todos os atributos.
            * .reinicicarParcial: reiniciar parcialmente o fluxo. Exemplo: quando dados de validação forem inseridos.

            =============
            PROPRIEDADES:
            =============

            * definem as etapas predecessoras e/ou sucessoras de cada etapa (atributo)
            """
            self.setDados = 0
            self.setConjunto = 0
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
            Método voltado para definição de uma etapa e suas validações

            =========
            ENTRADAS:
            =========

            * etapa (string): define a etapa que está em execução e se deseja validar. Deve ter mesmo nome que um
            dos atributos (em __init__)
            * ignoreValidacao (bool): irá ignorar a validação e atribuir 1 à etapa

            Se o método ocorrer sem erros, atribuirá 1 a etapa.
            """
            if not ignoreValidacao:
                # Teste para verificar se as etapas predecessoras foram executadas
                teste = [getattr(self,elemento) for elemento in getattr(self, '_predecessora_'+etapa)]
                # Caso não haja predecessora, o valor é atribuído a True
                teste = teste if teste != [] else [True]
                # Caso nenhuma predecessora tenha sido executada, retorna um erro
                if not any(teste):
                    raise SyntaxError('To run the {} method you must first run {}.'.format(etapa, ' or '.join(
                        getattr(self, '_predecessora_' + etapa))))
            # atribuindo o valor 1 (executado) ao atributo referente à etapa, atualmente em execução
            setattr(self, etapa, 1)

        def reiniciar(self,manter='setConjunto'):
            u"""
            Método utilizado para reiniciar o fluxo. (IMPACTA todas as etapas)

            Entende-se por reinicialização de fluxo, atribuir o valor de todos os atributos a zero, ou seja,
            como se os métodos de EstimacaoNaoLinear não tivessem sido executados.

            =========
            ENTRADAS:
            =========
            * manter (string): etapa que será mantida como 1.

            =========
            Filosofia
            =========
            * Todas as vezes que dados experimentais são adionados, o fluxo é reiniciado, inclusive definindo que não foram inseridos
             dados de validação
            """
            for atributo in vars(self).keys():
                if atributo != '_Fluxo__fluxo':
                    setattr(self, atributo, 0)

            self.__fluxoID = 0
            setattr(self, manter, 1)

        def reiniciarParcial(self, etapas=None):
            u"""
            Método utilizado para reiniciar apenas etapas específicas

            ========
            ENTRADAS
            ========

            * etapas (list): lista com as etapas que serão reiniciadas.

            =========
            Filosofia
            =========
            * Toda vez que é adicionado dados de validação é iniciado um novo fluxo de trabalho, para que a predição, analise de residuos
             e os respectivos gráficos e relatórios destas etapas sejam corretamente criados.
            """

            etapas = etapas if etapas is not None else self._sucessoresValidacao

            for atributo in etapas:
                setattr(self, atributo, 0)

            self.__fluxoID += 1

        @property
        def FLUXO_ID(self):
            u"""
            Obtém o número de identificação do fluxo
            """
            return self.__fluxoID

        @property
        def _predecessora_setDados(self):
            return []

        @property
        def _predecessora_setConjunto(self):
            return ['setDados']

        @property
        def _predecessora_otimizacao(self):
            return ['setConjunto']

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
            return ['setConjunto']

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

    def __init__(self, Model, symbols_y, symbols_x, symbols_param, PA=0.95, Folder='Projeto', **kwargs):
        u"""
        Classe para executar a estimação de parâmetros de modelos não lineares.

        Esta classe conta com um conjunto de métodos para obtenção do ótimo, utilizando a função objetivo WLS
        (weighted least squares), avaliação da incerteza dos parâmetros (com região de abrangência), estimativa da
        predição, cálculo da incerteza da predição e análise de resíduos.

        Classes auxiliares:
        * Grandeza
        * Flag
        * Graficos
        * Relatorio

        ======================
        Bibliotecas requeridas
        ======================
        * Numpy
        * Scipy
        * Matplotlib
        * Math
        * statsmodels

        * É recomendado uso da distribuição Anaconda Python 3

        =======================
        Entradas (obrigatórias)
        =======================
        * ``Model`` (Thread)       : objeto modelo. O modelo deve retornar um array com número de colunas igual ao número de grandezas dependentes.
        * ``symbols_y`` (list)     : lista com os simbolos das grandezas dependentes (Não podem haver caracteres especiais)
        * ``symbols_x`` (list)     : lista com os simbolos das grandezas independentes (Não podem haver caracteres especiais)
        * ``symbols_param`` (list) : lista com o simbolos dos parâmetros  (Não podem haver caracteres especiais)

        ====================
        Entradas (opcionais)
        ====================
        * ``PA`` (float): probabilidade de abrangência da análise. Deve estar entre 0 e 1. Default: 0.95.
        * ``projeto`` (string): nome do projeto (Náo podem haver caracteres especiais)

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
        alguns métodos, na ordem indicada (vide observação):

        **ESTIMAÇÂO DE PARÂMETROS**

        * ``setDados'': método para incluir os dados experimentais. Deve ser executado uma vez para as grandezas dependentes e outra
        para as grandezas independentes. (Vide documentação do método)
        * ``setConjunto``        : método para definir se os dados experimentais incluídos serão usados para estimação de parâmetros ou para
        validação. (Vide documentação do método)
        * ``optimize``              : método para realizar a otimização, com base no conjunto de dados definido em setConjunto. (Vide documentação do método)
        * ``parametersUncertainty``  : método que avalia a incerteza dos parâmetros (Vide documentação do método)
        * ``setDados'': (é opcional para inclusão de dados de validação)
        * ``setConjunto``        : (é opcional para inclusão de dados de validação)
        * ``Prediction``             : método que avalia a predição do modelo e sua incerteza ou utilizando os dados de validação. Caso estes \
        não estejam disponíveis, será utilizado os mesmos dados de estimação (Vide documentação do método)
        * ``residualAnalysis``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``plots``             : método para criação dos gráficos (Vide documentação do método)
        * ``_armazenarDicionario`` : método que retorna as grandezas sob a forma de um dicionário (Vide documentação do método)

        **PREDIÇÃO**

        * ``setConjunto``        : método para incluir dados obtidos de experimentos. Neste há a opção de determinar \
        se estes dados serão utilizados como dados para estimar os parâmetros ou para validação. (Vide documentação do método)
        * ``SETparametro``         : método adicionar manualmente valores das estimativas dos parâmetros e sua matriz covarãncia. É assumido \
        que os parâmetros foram estimados para o conjunto de dados fornecidos para estimação.
        * ``setConjunto``        : (é opcional para inclusão de dados de validação)
        * parametersUncertainty      : (é opcional para avaliação da incerteza, caso não incluído em SETparametro). Entretanto, este estará limitado a \
        calcular a matriz de covariância dos parâmetros. Não será avaliada a região de abrangẽncia (esta deve ser incluída via SETparametro)
        * ``Prediction``             : método que avalia a predição do modelo e sua incerteza ou utilizando os dados de validação. Caso estes \
        não estejam disponíveis, será utilizado os mesmos dados de estimação (Vide documentação do método)
        * ``residualAnalysis``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``plots``             : método para criação dos gráficos (Vide documentação do método)
        * ``_armazenarDicionario`` : método que retorna as grandezas sob a forma de um dicionário (Vide documentação do método)


        **OBSERVAÇÃO**: A ordem de execução dos métodos é importante. Esta classe só permite a execução de métodos, caso as etapas predescessoras tenham sido
        executadas. Entretanto, alguns métodos possuem flexibilidade. Segue abaixo algumas exemplos:
        * setConjunto para definir os dados de estimação deve ser sempre executado antes de optimize
        * setConjunto para definir os dados de validação deve ser sempre executado antes de predicao
        * plots é um método que pode ser executado em diferentes momentos:
            * se for solicitado os gráficos das grandezas-entrada, o método pode ser executado logo após setDados
            * se for solicitado os gráficos da otimização, o método pode ser executado logo após otimização

        =================
        Fluxo de trabalho
        =================

        Esta classe possui uma classe interna, Fluxo, que valida a correta ordem de execução dos métodos. É importante
        salientar que cada vez que o método ``setConjunto`` é utilizado, é criado um novo ``Fluxo de trabalho`` ou ele
        ``Reinicia`` todos.

        **Observação 1**: Se forem adicionados diferentes dados de validação (execuções do método setDados para incluir tais dados), \
        são iniciado novos fluxos.

        **Observação 2**: Se forem adicionados novos dados para estimacao, todo o histórico de fluxos é apagado e reniciado.

        Esta característica permite a avaliação de diferentes dados de valiação consecutivamente (uso dos métodos Prediction, residualAnalysis, plots),
        após a estimação dos parâmetros (optimize, parametersUncertainty)

        ======
        Saídas
        ======

        As saídas deste motor de cálculo estão, principalmente, sob a forma de atributos e gráficos.
        Os principais atributos de uma variável Estimacao, são:

        * ``x`` : objeto Grandeza que contém todas as informações referentes às grandezas \
        independentes sob a forma de atributos:
            * ``estimação`` : referente aos dados experimentais. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``calculado``    : referente aos dados calculados pelo modelo. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``predicao``    : referente aos dados de validação. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``residuos``     : referente aos resíduos de regressão. Principais atributos: ``matriz_estimativa``, ``estatisticas``

        * ``y``          : objeto Grandeza que contém todas as informações referentes às grandezas \
        dependentes sob a forma de atributos. Os atributos são os mesmos de x.

        * ``parametros`` : objeto Grandeza que contém todas as informações referentes aos parâmetros sob a forma de atributos.
            * ``estimativa``         : estimativa para os parâmetros
            * ``matriz_covariancia`` : matriz de covariância
            * ``matriz_correlacao``   : matriz de correlação
            * ``regiao_abrangencia`` : pontos contidos na região de abrangência

        Obs.: Para informações mais detalhadas, consultar os Atributos da classe Grandeza.

        * ``PA``: probabilidade de abrangência da análise

        ===============
        Função objetivo
        ===============

        A função objetivo deve ser um objeto com uma estrutura específica. Consulte arquivo Funcao_Objetivo.

        OBSERVAÇÃO:
        O Motor de cálculo sempre irá enviar como uma lista argumentos para a função objetivo \
        nesta ordem:

        * vetor com os pontos experimentais das grandezas dependentes
        * matriz com os pontos experimentais das grandezas independentes (cada coluna representa uma grandeza independente)
        * matriz covariância das grandezas dependentes
        * matriz covariância das grandezas independentes
        * argumentos extras a serem passados para o modelo (entrada de usuário)
        * o modelo
        * lista com os símbolos das grandezas independentes
        * lista com os símbolos das grandezas dependentes
        * lista com os símbolos dos parâmetros

        =======
        Modelo
        ======
        Deve ter a seguinte estrutura

        def model(parametros,x,*args):

            ...

            return y

        onde y é uma matriz onde cada coluna representa os valores calculados para cada grandeza de saída, dado o vetor
        de grandezas independente x.

        =====================
        Atributos em destaque
        =====================

        CONFIGURAÇÕES:

        * ._configFolder: variável que contém o nome de todas as pastas criadas pelo algoritmo nas etapas de Gráficos e
         relatórios. Alterando o conteúdo de uma chave, altera-se o nomes das pastas. É permitido alterar o conteúdo das
         chaves (nomes das pastas), mas alterando as chaves ocasionará erros.

        * .__args_user: variável que contém os argumentos extras a serem passados para o modelo. Equivale ao argumento
        args em otimiza.

        * .__flag: classe flag que controla o comportamento do algoritmo. Flags disponíveis:
            *'dadosestimação': identifica se dados para a estimação foram inseridos
            * 'dadospredicao'   : identifica se dados para validação foram inseridos
            * 'reconciliacao'    : identifica se foi solicitada reconciliação de dados (HOLD: aguarda implementação da
            reconciliação)
            * 'graficootimizacao': identifica se o algoritmo de otimização tem gráficos de desempenho
            * 'relatoriootimizacao': identifica se o algoritmo de otimização possui relatório na forma de um arquivo

        * .__base_path: identifica o caminho raiz nos quais todos gráficos e relatórios serão salvos
        * .__controleFluxo: objeto Fluxo que controla a validação das etapas, e o fluxo de execução do motor de cálculo.
           * .__controleFLuxo.FLUXO_ID: 0 se só houverem dados experimentais. >0 foram inseridos dados de validação (Isto
           é usado na plotagem de gráficos, para evitar que o uso de dados de validação sucessivos sobrescrevam os gráficos)

        OUTROS:

        * .Otimizacao: salva todas as informações do algorimo de otimização. [Só existe após execução do método otimização].
        * ._deltaHessiana: incremento a ser utilizado para avaliar a matriz Hessiana (pode ser definido via kwargs no método
        incertezaParametros e/ou Predicao)
        * ._deltaGy: incremento a ser utilizado para avaliar a matriz Gy (derivadas segundas da função objetivo em relação
        a dados estimação de y e parâmetros) (pode ser definido via kwargs no método incertezaParametros e/ou Predicao)
        * ._deltaS: incremento a ser utilizado para avalair a transposta da matriz jacobiana do modelo em relação aos
        parâmetros (pode ser definido via kwargs no método incertezaParametros e/ou Predicao)
        * .Hessiana: salva a matriz Hessiana (somente avaliada após incertezaParametros ou Predicao - a depender do método solicitado)
        * .Gy: salva matriz Gy (somente avaliada após incertezaParametros ou Predicao - a depender do método solicitado)
        * .S: salva matriz S (somente avaliada após incertezaParametros ou Predicao - a depender do método solicitado)
        * .estatisticas: dicionário que contém aguns testes estatístcicos (Para outras estatísticas aqui não incluídas
        consulte Grandezas)
        * FOotimo: valor da função objetivo no ponto ótimo
        """
        # ---------------------------------------------------------------------
        # CONTROLE DO FLUXO DE INFORMAÇÕES DO ALGORITMO
        # ---------------------------------------------------------------------
        # FLUXO DE INFORMAÇÕES -> conjunto de etapas do algoritmo
        self.__controleFluxo = self.Fluxo()

        # ---------------------------------------------------------------------
        # VALIDAÇÕES GERAIS DE KEYWORDS
        # ---------------------------------------------------------------------
        # Keywords disponíveis para o método de entrada
        self.__keywordsEntrada = ('names_x', 'units_x', 'label_latex_x', 'names_y', 'units_y', 'label_latex_y',
                                  'names_param','units_param', 'label_latex_param', 'base_path')

        # Validação se houve keywords digitadas incorretamente:
        keyincorreta = [key for key in kwargs.keys() if not key in self.__keywordsEntrada]

        if len(keyincorreta) != 0:
            raise NameError('keyword(s) incorreta(s): ' + ', '.join(keyincorreta) + '.' +
                            ' Keywords disponíveis: ' + ', '.join(self.__keywordsEntrada) + '.')

        # Verificação de PA está entre 0 e 1
        if not 0 < PA < 1:
            raise ValueError('The coverage probability must be between 0 and 1.')

        # Verificação se o nome do projeto é um string
        if not isinstance(Folder, str):
            raise TypeError('The Folder name must be a string.')

        # Verificação se o nome do projeto possui caracteres especiais
        if not Folder.isalnum():
            raise NameError('The folder name must not contain special characters')

        # Verificação se o base_path é uma string
        if kwargs.get(self.__keywordsEntrada[9]) is not None and not isinstance(kwargs.get(self.__keywordsEntrada[9]),
                                                                                  str):
            raise TypeError('The keyword {} must be a string.'.format(self.__keywordsEntrada[9]))

        # ---------------------------------------------------------------------
        # INICIALIZAÇÃO DAS GRANDEZAS
        # ---------------------------------------------------------------------
        # Variável      = Grandeza(simbolos      ,nomes                                ,unidades                             ,label_latex                          )
        self.x          = Grandeza(symbols_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        self.y          = Grandeza(symbols_y    ,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        self.parametros = Grandeza(symbols_param,kwargs.get(self.__keywordsEntrada[6]),kwargs.get(self.__keywordsEntrada[7]),kwargs.get(self.__keywordsEntrada[8]))

        # Verificação se os símbolos são distintos
        # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
        if len(set(self.y.simbolos).intersection(self.x.simbolos)) != 0 or len(set(self.y.simbolos).intersection(self.parametros.simbolos)) != 0 or len(set(self.x.simbolos).intersection(self.parametros.simbolos)) != 0:
            raise NameError('The symbols of the quantities must be different.')


        # ---------------------------------------------------------------------
        # OUTRAS VARIÁVEIS
        # ---------------------------------------------------------------------
        # Propabilidade de abrangência
        self.PA = PA

        # Incremento das derivadas numéricas
        self._deltaHessiana = 1e-5  # Hessiana da função objetivo
        self._deltaGy = 1e-5        # Gy (derivada parcial segunda da função objetivo em relação aos parâmetros e dados experimentais)
        self._deltaS = 1e-5         # S (transposto do jacobiano do modelo)

        # ---------------------------------------------------------------------
        # CRIAÇÃO DAS VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------
        # Modelo
        self.__modelo    = Model
        # Argumentos extras a serem passados para o modelo definidos pelo usuário.
        self.__args_user = None # Aqui iniciado para que possa existir na herança
        # Optimization algorithm position history (parameters) (used in optimizes and / or objective function mapping
        self.__decisonVariablesMapped = []
        # Fitness history (objective function value) of the optimization algorithm (used in optimizing and / or objective function mapping)
        self.__OFMapped = []
        # Caminho base para os arquivos, caso seja definido a keyword base_path ela será utilizada.
        if kwargs.get(self.__keywordsEntrada[9]) is None:
            self.__base_path = getcwd()+ sep +str(Folder)+sep
        else:
            self.__base_path = kwargs.get(self.__keywordsEntrada[9])

        # Flags para controle de informações
        self.__flag = flag()
        self.__flag.setCaracteristica(['dadosestimacao','dadospredicao',
                                       'reconciliacao','mapeamentoFO',
                                       'graficootimizacao','relatoriootimizacao','Linear'])
        # uso das caracterśiticas:
        # dadosestimacao: indicar se dadosestimacao foram inseridos
        # dadospredicao: indicar se dadospredicao foram inseridos
        # reconciliacao: indicar se reconciliacao está sendo executada
        # graficootimizacao: indicar se na etapa de otimização são utilizados algoritmos de otimização que possuem
        #                    gráficos de desempenho
        # relatoriootimizacao: indicar se o algoritmo de otimização possui relatório

        # Variável que controla o nome das pastas criadas pelos métodos gráficos e relatórios
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

        # variáveis auxiliares para definição de conjunto de dados
        self.__xtemp = None
        self.__uxtemp = None
        self.__ytemp = None
        self.__uytemp = None

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
        # métodos para avaliação da incerteza
        return ('2InvHessiana', 'Geral', 'SensibilidadeModelo')

    @property
    def __keywordsDerivadas(self):
        # keywords disponíveis para avaliação das derivadas
        return ('deltaHess', 'deltaGy', 'deltaS', 'delta')

    @property
    def __tipoObjectiveFunctionMapping(self):
        # types of objective function mapping algorithms available
        return ('MonteCarlo',)

    @property
    def __graph_flux_association(self):
        return {'setConjunto':[self.__tipoGraficos[1]],'incertezaParametros':[self.__tipoGraficos[0],self.__tipoGraficos[3]],
                'predicao':[self.__tipoGraficos[2],self.__tipoGraficos[3]],'analiseResiduos':[self.__tipoGraficos[5]]}

    @property
    def _args_model(self):
        """
        Método que retorna argumentos extras a serem passados para o modelo

        :return: lista (list) com argumentos extras
        """
        # ---------------------------------------------------------------------
        # LISTA DE ATRIBUTOS A SEREM INSERIDOS NO MODELO
        # ---------------------------------------------------------------------

        return [self.__args_user,self.x.simbolos,self.y.simbolos,self.parametros.simbolos]

    def __validacaoDadosEntrada(self,dados,udados,NV):
        u"""
        Validação dos dados de entrada

        * verificar se as colunas dos arrays de entrada tem o mesmo número dos símbolos das variáveis definidas (y, x)
        * verificar se o número de pontos é o mesmo
        * verificar se os graus de liberdade são suficientes para realizar a estimação
        """
        if dados.shape[1] != NV:
            raise ValueError('The number of variables defined was {:d}, but data was entered for {:d} variables.'.format(NV,dados.shape[1]))

        if udados.shape[1] != NV:
            raise ValueError('The number of variables defined was {:d}, but uncertainties were inserted for {:d}.'.format(NV,udados.shape[1]))

        if dados.shape[0] != udados.shape[0]:
            raise ValueError('Data vectors and their uncertainties must have the same number of points.')

        if udados.shape[0]*self.y.NV-float(self.parametros.NV) <= 0: # Verificar se há graus de liberdade suficiente
            warn('Insufficient degrees of freedom. Your experimental data set is not enough to estimate the parameters!',UserWarning)

    def setDados(self, type, *data):
        u"""
        Método para tratar os dados de entrada de grandezas dependentes e independentes e organizá-los em formato adequado.
        Este método deve ser executado para cada grupo de grandezas envolvidas na estimação: (i) grandezas independentes e (ii) grandezas
        independentes.
        Após inclusão dos grupos de grandezas é necessário executar o método setConjunto para definir se s grupos de grandezas (dependentes-indepentes) serão
        usados para a estimação (avaliação dos parâmetros) ou predição (validação do modelo).

        =======================
        Entradas (Obrigatórias)
        =======================

        type (bool): 0 (grandeza independente) ou 1 (grandeza dependente)
        *dados: tuplas contém os dados experimentais e incertezas das grandezas. Cada tupla deve
        conter duas listas: (i) uma com os dados experimentais de uma grandeza e (ii) outra lista contendo as incertezas
        associada à cada dado. Formato: ([dados grandeza],[incertezas grandeza]). Podem ser inseridas uma tupla para cada grandeza.

        ========
        Exemplo
        ========
        # Considere o grupo de dados:
        # Grandezas independentes:
        x1 = [1,2,3] # dados
        ux1 = [1,1,1] # incerteza
        x2 = [ 4,5,6 ] # dados
        ux2 = [1,1,1] # incerteza
        # Grandezas dependentes
        y1  = [5,7,8]
        uy1 = [1,1,1]

        Estime = EstimacaoNaoLinear(Modelo,simbolos_x=['x1','x2'], simbolos_y=['y1'], simbolos_param=['A','B'])
        Estime.setDados(0,(x1,ux1),(x2,ux2))
        Estime.setDados(1,(y1,uy1))

        Para consultar variáveis, faz-se necessário indicar que os dados inseridos constituem um conjunto:

        Estime.setConjunto()

        As variáveis, então, podem ser consultadas através:

        # Grandezas independentes:

        Estime.x.estimacao.matriz_estimativa # Dados
        Estime.x.estimacao.matriz_covariancia # Incertezas foram convertidas em uma matriz de covariância

        # Grandeza dependentes

        Estime.y.estimacao.matriz_estimativa  # Dados
        Estime.y.estimacao.matriz_covariancia # Incertezas foram convertidas em uma matriz de covariância

        ============
        Detalhamento
        ============
        * Converte pares de listas de dados e suas respectivas incertezas em arrays de duas dimensões
        * Estes arrays são armazenados como variáveis temporárias, pondendo ser consultados após método setConjunto.
        """

        # VALIDATION

        if len(data) == 0:
            raise TypeError('It is necessary to include at least data for one quantity: ([data],[uncertainty])')

        for ele in data:
            if not (isinstance(ele, list) or isinstance(ele, tuple)):
                raise TypeError('Each quantity pair (data and uncertainty) must be a tuple or list: ([data],[uncertainty]).')

            if len(ele) != 2:
                raise TypeError('Each tuple must contain only 2 lists: ([data],[uncertainty])')

            for ele_i in ele:
                if not (isinstance(ele, list) or isinstance(ele, tuple)):
                    raise TypeError('Each quantity pair (data and uncertainty) must be a tuple or list: ([data],[uncertainty]).')

        # EXECUTION

        self.__controleFluxo.SET_ETAPA('setDados')

        if type == 0:

            X  = transpose(array([data[i][0] for i in range(len(data))], ndmin=2, dtype=float))
            uX = transpose(array([data[i][1] for i in range(len(data))], ndmin=2, dtype=float))

            self.__validacaoDadosEntrada(X, uX, self.x.NV)

            self.__xtemp   = X
            self.__uxtemp = uX

        else:

            Y  = transpose(array([data[i][0] for i in range(len(data))], ndmin=2, dtype=float))
            uY = transpose(array([data[i][1] for i in range(len(data))], ndmin=2, dtype=float))

            self.__validacaoDadosEntrada(Y, uY, self.y.NV)

            self.__ytemp = Y
            self.__uytemp = uY


        # TODO:
        # validação de args
        # graus de liberdade devem ser passados por aqui


    def setConjunto(self,glx=[],gly=[],dataType=None,uxy=None):
        u"""
        setConjunto(self,glx=[],gly=[],dataType=None,uxy=None)
        
        ===================================================================================
        Method for including the estimation data. It must be run after the setDados method.
        ===================================================================================
            - Parameters
            ----------
            glx : list, optional
                list with the freedom degrees for the input quantities.
            gly : list, optional
                list with the freedom degrees for the output quantities.
            dataType : string, optionial
                defines the purpose of the informed data set.

                ============ =================================================================
                dataType     purpose
                ============ =================================================================
                estimacao    dataset to perform the parameter estimation
                predicao     dataset to perform the prediction of the output model estimates.
                             In this case, the parameters are already known.
                ============ =================================================================

            uxy : not in use

            - Notes
             -----
             If the dataType argument was not defined the method defines automatically as 'estimacao' or 'predicao'.
             If the prediction data was not defined the method define dataType as 'estimacao'.
             If the quantities freedom degrees was not defined it will be assumed constant and equal to 100

        """
        # ---------------------------------------------------------------------
        # FLUX
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('setConjunto')
        # ---------------------------------------------------------------------
        # VALIDATON
        # ---------------------------------------------------------------------
        if (self.__xtemp is None) or (self.__ytemp is None) or (self.__uxtemp is None) or (self.__uytemp is None):
            raise ValueError('It is necessary to run the setDados method to define data for dependent (y) and independent (x) quantities.')

        # dataType validation
        if dataType is not None:
            if not set([dataType]).issubset(self.__tiposDisponiveisEntrada):
                raise ValueError('The input(s) ' + ','.join(
                    set([dataType]).difference(self.__tiposDisponiveisEntrada)) + ' are not available. You should use: ' + ','.join(
                    self.__tiposDisponiveisEntrada) + '.')

        # validation of the amount of experimental data

        if self.__xtemp.shape[0] != self.__ytemp.shape[0]:
            raise ValueError('{:d} data were entered for dependent quantities, but {:d} for independent quantities'.format(self.__ytemp.shape[0],self.__xtemp.shape[0]))

        # ---------------------------------------------------------------------
        # EXECUTION
        # ---------------------------------------------------------------------

        if dataType is None:
            if not self.__flag.info['dadosestimacao']:
                dataType = self.__tiposDisponiveisEntrada[0]
            else:
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
                self.x._SETdadosestimacao(estimativa=self.__xtemp,matriz_incerteza=self.__uxtemp,gL=glx)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the estimation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosestimacao(estimativa=self.__ytemp,matriz_incerteza=self.__uytemp,gL=gly)
            except Exception as erro:
                raise RuntimeError('EError in the creation of the estimation set of the quantity Y: {}'.format(erro))

        # prediction data
        if type == self.__tiposDisponiveisEntrada[1]:
            self.__flag.ToggleActive('dadospredicao')

            self.__controleFluxo.reiniciarParcial()
            # ---------------------------------------------------------------------
            # ASSIGNMENT OF VALUES TO QUANTITIES
            # ---------------------------------------------------------------------
            # Saving the validation data.
            try:
                self.x._SETdadosvalidacao(estimativa=self.__xtemp,matriz_incerteza=self.__uxtemp,gL=glx)
            except Exception as error:
                raise RuntimeError('Error in the creation of the validation set of the quantity X: {}'.format(error))

            try:
                self.y._SETdadosvalidacao(estimativa=self.__ytemp,matriz_incerteza=self.__uytemp,gL=gly)
            except Exception as error:
                raise RuntimeError('Error in the creation of the validation set of the quantity Y: {}'.format(error))

        if not self.__flag.info['dadospredicao']:
            # If setConjunto method is only performed for experimental data,
            # it will be assumed that also are validation data because all prediction
            # calculation is performed to the validation data.
            # ---------------------------------------------------------------------
            # ASSIGNMENT OF VALUES TO QUANTITIES
            # ---------------------------------------------------------------------
            # Saving validation data.
            try:
                self.x._SETdadosvalidacao(estimativa=self.__xtemp,matriz_incerteza=self.__uxtemp,gL=glx)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity X: {}'.format(erro))

            try:
                self.y._SETdadosvalidacao(estimativa=self.__ytemp,matriz_incerteza=self.__uytemp,gL=gly)
            except Exception as erro:
                raise RuntimeError('Error in the creation of the validation set of the quantity Y: {}'.format(erro))

        # Transforming the temporary variables ( xtemp, uxtemp, ytemp, uytemp) in empty lists.
        self.__xtemp = None
        self.__uxtemp = None
        self.__ytemp = None
        self.__uytemp = None

        # initialization of casadi's variables
        self._constructionCasadiVariables()

    def _constructionCasadiVariables(self): # construction of the casadi variables
        u"""

        When MT_PEU is working with estimation data the symbolics variables should be created with this data.
        But if the data is for validation, so validation data should be used for to create the symbolic variables.
        This is necessary because the data size is considered in the symbolic variables creation.

        """
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------



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

            #for i in range(self.x.estimacao.NE):
             #   self.__symUxo = vertcat(self.__symUxo, MX.sym('Uxo' + str(i)))
             #   self.__symVariables = vertcat(self.__symVariables, self.__symUxo[i])
              #  self._values = vertcat(self._values, self.y.estimacao.matriz_incerteza[:, i:i + 1])

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
             ------
             Before executing the optimize method it's necessary to execute the "setConjunto" method
             and define the estimation data.
             Every time the optimization method is run, the information about the parameters is lost.
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

        # parameters report creation
        if parametersReport is True:
            self._out.Parametros(self.parametros,self.FOotimo)

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
        Método para atribuir uma estimativa aos parâmetros e, opcionamente, sua matriz de covarância e região de abrangência.
        Substitui o métodos otimiza e, opcionalmente, parametersUncertainty.

        Caso seja incluída somente uma estimativa para os parâmetros, o método parametersUncertainty deve ser executado.

        ========
        Entradas
        ========

        * estimative (list)         : estimativa para os parâmetros
        * variance (array,ndmin=2) : matriz de covariância dos parâmetros
        * region (list[list])       : lista contendo listas com os parâmetros que pertencem á região de abrangência

        ===
        USO
        ===
        * Inclusion of parameter estimation: will replace the optimization method. You will need to execute the uncertaintyParameter method.
        * Inclusion of parameter estimation and variance: will replace the optimization method and a part of the uncertainty method.
        For objective Function Mapping the region by the likelihood method, the uncertaintyParameter method must be performed (will override the uncertainty inseparated).
        * Inclusion of parameter estimation, variance and region: will replace optimization and uncertaintyParameter method.

        =================
        Keyword Arguments
        =================
        * limite_superior: limite superior dos parâmetros
        * limtie_inferior: limite_inferior dos parâmetros
        * args           : argumentos extras a serem passados para o modelo

        =========
        ATRIBUTOS
        =========
        O método irá incluir estas informações no atributo parâmetros
        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('SETparametro')
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        # Caso não haja dados de estimação -> erro
        if not self.__flag.info['dadosestimacao']:
            raise SyntaxError('It is necessary to add estimation data.')

        # SETparameter não pode ser executado em conjunto com optimize
        if self.__controleFluxo.otimizacao:
            raise SyntaxError('The SETparameter method cannot be executed with optimize method')

        # ---------------------------------------------------------------------
        # ARGUMENTOS EXTRAS A SEREM PASSADOS PARA O MODELO
        # ---------------------------------------------------------------------
        # Obtenção de args_user
        if kwargs.get('args') is not None:
            self.__args_user = kwargs.pop('args')

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------
        # Atribuindo o valores para a estimativa dos parâmetros e sua matriz de 
        # covariância
        self.parametros._SETparametro(estimative, variance, region, **kwargs)

        # ---------------------------------------------------------------------
        # AVALIAÇÃO DO MODELO
        # ---------------------------------------------------------------------
        # Avaliação do modelo no ponto ótimo informado
        try:
            aux = self.__excModel(self.parametros.estimativa,self._values)
        except Exception as erro:
            raise SyntaxError(u'Error in the model when evaluated in the informed parameters estimative. Error identified: "{}"'.format(erro))

        # ---------------------------------------------------------------------
        # OBTENÇÃO DO PONTO ÓTIMO
        # ---------------------------------------------------------------------

        self.FOotimo = float(self._excObjectiveFunction(self.parametros.estimativa, self._values))

        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------

        # Caso seja definida a variância, é assumido que o método parametersUncertainty
        # foi executado, mesmo que a inclusão de abrangência seja opcional.
        if variance is not None:
            self.__controleFluxo.SET_ETAPA('incertezaParametros')

        # Caso seja definida a região, é assumido que o método regiaoAbrangencia
        # foi executado.
        if region is not None:
            self.__controleFluxo.SET_ETAPA('regiaoAbrangencia', ignoreValidacao=True)

        # parameters report creation
        if parametersReport is True:
            self._out.Parametros(self.parametros, self.FOotimo)

    def parametersUncertainty(self,uncertaintyMethod ='Geral', parametersReport = True, objectiveFunctionMapping=True, **kwargs):
        u"""

        Método para avaliação da matriz de covariãncia dos parâmetros e região de abrangência.

        ===================
        Método predescessor
        ===================

        É necessário executar a otimização ou SETparametro.

        =======================
        Entradas (opcionais)
        =======================

        * uncertainty method (string): method for calculating the covariance matrix
        of the parameters. Available methods: 2InvHessian, Geral, SensibilidadeModelo
        * objectiveFunctionMapping (bool): Identifies if an algorithm for mapping the objective function will be executed.

        ======
        Saídas
        ======
        * a matriz de covariância dos parâmetros é salva na Grandeza parâmetros

        ==========
        Observação
        ==========
        * A região de abrangência só é executada caso haja histórico da otimização (ETAPA: mapeamentoFO) e o atributo regiao_abrangencia
        de parâmetros não esteja definido.
        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('incertezaParametros')
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         

        if uncertaintyMethod not in self.__metodosIncerteza:
            raise NameError('The method requested to calculate the uncertainty of the parameters {}'.format(uncertaintyMethod)
                            + ' is not available. Available methods ' + ', '.join(self.__metodosIncerteza) + '.')

        if not isinstance(objectiveFunctionMapping, bool):
            raise TypeError('The argument objectiveFunctionMapping must be boolean (True ou False).')

        # ---------------------------------------------------------------------
        # MATRIZ DE COVARIÂNCIA DOS PARÂMETROS
        # ---------------------------------------------------------------------

        # Avaliação de matrizes auxiliares
        # Matriz Hessiana da função objetivo em relação aos parâmetros
        # somente avaliada se o método é 2InvHess ou Geral
        if uncertaintyMethod == self.__metodosIncerteza[0] or uncertaintyMethod == self.__metodosIncerteza[1]:
            self.__Hessiana_FO_Param()

            # Inversa da matriz hessiana a função objetivo em relação aos parâmetros
            invHess = inv(self.Hessiana)

        # Gy: derivadas parciais segundas da função objetivo em relação aos parâmetros e
        # dados experimentais
        # Somente avaliada caso o método seja Geral
        if uncertaintyMethod == self.__metodosIncerteza[1]:
            self.__Matriz_Gy()

        # Matriz de sensibilidade do modelo em relação aos parâmetros
        # Somente avaliada caso o método seja o simplificado
        if uncertaintyMethod == self.__metodosIncerteza[2]:
            self.__Matriz_S()

        # ---------------------------------------------------------------------
        # AVALIAÇÃO DA INCERTEZA DOS PARÂMETROS
        # ---------------------------------------------------------------------

        # MATRIZ DE COVARIÂNCIA
        # Método: 2InvHessiana ->  2*inv(Hess)
        if uncertaintyMethod == self.__metodosIncerteza[0]:
            matriz_covariancia = 2*invHess

        # Método: geral - > inv(H)*Gy*Uyy*GyT*inv(H)
        elif uncertaintyMethod == self.__metodosIncerteza[1]:
            matriz_covariancia  = invHess.dot(self.Gy).dot(self.y.estimacao.matriz_covariancia).dot(self.Gy.transpose()).dot(invHess)

        # Método: simplificado -> inv(trans(S)*inv(Uyy)*S)
        elif uncertaintyMethod == self.__metodosIncerteza[2]:
            matriz_covariancia = inv(self.S.transpose().dot(inv(self.y.estimacao.matriz_covariancia)).dot(self.S))

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZA
        # ---------------------------------------------------------------------
        self.parametros._updateParametro(matriz_covariancia=matriz_covariancia)

        # ---------------------------------------------------------------------
        # REGIÃO DE ABRANGÊNCIA
        # ---------------------------------------------------------------------
        # MAPPING OF OBJECTIVE FUNCTION:
        if objectiveFunctionMapping and self.parametros.NV != 1:
            self.__objectiveFunctionMapping(**kwargs)
            self.__flag.ToggleActive('mapeamentoFO')

        # A região de abrangência só é executada caso haja histórico de posicoes e fitness
        if self.__controleFluxo.mapeamentoFO and self.parametros.NV != 1:
            # OBTENÇÃO DA REGIÃO:
            regiao = self.regiaoAbrangencia()
            # ATRIBUIÇÃO A GRANDEZA
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
         Method used for objective function mapping

        =================
        Keyword arguments
        =================
            * MethodObjectivefunctionmapping  ('string'): defines which method is used in objective function mapping. Available: MonteCarlo
        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('mapeamentoFO')

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------

        if kwargs.get('MethodObjectivefunctionmapping') is not None:
            tipo = kwargs.pop('MethodObjectivefunctionmapping')
        else:
            tipo = 'MonteCarlo'

        # evaluating whether the objective function mapping type is available
        if tipo not in self.__tipoObjectiveFunctionMapping:
            raise NameError('O método solicitado para mapeamento da função objetivo {}'.format(
                tipo) + ' não está disponível. Métodos disponíveis ' + ', '.join(self.__tipoObjectiveFunctionMapping) + '.')

        # caso seja MonteCarlo:
        if tipo == self.__tipoObjectiveFunctionMapping[0]:
            kwargsdisponiveis = ('iteracoes', 'limite_superior', 'limite_inferior', 'fatorlimitebusca', 'distribuicao')

            # avaliando se as keywords estão disponíveis
            if not set(kwargs.keys()).issubset(kwargsdisponiveis):
                raise NameError('O(s) keyword(s) argument digitado(s) está(ão) incorreto(s). Keyword disponíveis: ' +
                                ', '.join(kwargsdisponiveis) + '.')
            # avaliando o número de iterações -> dever ser maior do que 1
            if kwargs.get(kwargsdisponiveis[0]) is not None:
                if kwargs.get(kwargsdisponiveis[0]) < 1:
                    raise ValueError('O número de iterações deve ser inteiro e positivo.')
            # avaliando o fatorlimite -> deve ser positivo
            if kwargs.get(kwargsdisponiveis[3]) is not None:
                if kwargs.get(kwargsdisponiveis[3]) < 0:
                    raise ValueError('O fator limite busca deve positivo.')
            # avaliando a distribuição
            if kwargs.get(kwargsdisponiveis[4]) is not None:
                if kwargs.get(kwargsdisponiveis[4]) not in ['uniforme', 'triangular']:
                    raise ValueError('As distribuições disponíveis para o Método de MonteCarlo são: {}.'.format(['uniforme', 'triangular']))

        # ---------------------------------------------------------------------
        # LIMTES DE BUSCA
        # ---------------------------------------------------------------------
        limite_superior = kwargs.get('limite_superior')
        limite_inferior = kwargs.get('limite_inferior')
        fatorlimitebusca = kwargs.get('fatorlimitebusca') if kwargs.get('fatorlimitebusca') is not None else 1/10.

        #Validação
        if (((not isinstance(limite_superior, list) and not isinstance(limite_superior, tuple)) and limite_superior is not None) or (((not isinstance(limite_inferior, list)) and (not isinstance(limite_inferior, tuple))) and limite_inferior is not None)):
            raise TypeError('The upper_limit and the lower_limit must be lists or tuples.')
        if (limite_superior is not None and len(limite_superior) != self.parametros.NV) or (limite_inferior is not None and len(limite_inferior) != self.parametros.NV):
            raise TypeError('Upper_limits and lower_limits must be lists or tuples of the same size as self.paramtros.NV')

        if limite_superior is None or limite_inferior is None:
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

        if limite_superior is None:
            limite_superior = [extremo_elipse_superior[i] + (extremo_elipse_superior[i]-extremo_elipse_inferior[i])*fatorlimitebusca for i in range(self.parametros.NV)]
        else:
            kwargs.pop('limite_superior') # retira limite_superior dos argumentos extras

        if limite_inferior is None:
            limite_inferior = [extremo_elipse_inferior[i] - (extremo_elipse_superior[i]-extremo_elipse_inferior[i])*fatorlimitebusca for i in range(self.parametros.NV)]
        else:
            kwargs.pop('limite_inferior') # retira limite_inferior dos argumentos extras

        # Validating limits
        # Verificar se limite superior é maior do que inferior
        test_bounds = [0]*self.parametros.NV
        for i in arange(self.parametros.NV):
            if (limite_inferior[i]>limite_superior[i]) or (limite_inferior[i]>self.parametros.estimativa[i] or self.parametros.estimativa[i]>limite_superior[i]):
                test_bounds[i] = 1

        index_test_bounds = [i for i, ele in enumerate(test_bounds) if ele]

        if any(test_bounds):
            raise TypeError(('The parameter estimate of '+'{} '*len(index_test_bounds)+' ​​must be between the lower_limit and the upper_limit. Parameter estimate: {}').format(*[self.parametros.simbolos[i] for i in index_test_bounds],self.parametros.estimativa))

        # ---------------------------------------------------------------------
        # MÉTODO MONTE CARLO
        # ---------------------------------------------------------------------
        if tipo == self.__tipoObjectiveFunctionMapping[0]:
            iteracoes = int(kwargs.get('iteracoes') if kwargs.get('iteracoes') is not None else 10000)

            for cont in range(iteracoes):

                amostra_total_uni = [uniform(limite_inferior[i], limite_superior[i], 1)[0] for i in range(self.parametros.NV)]

                amostra_total = [triangular(limite_inferior[i], self.parametros.estimativa[i], limite_superior[i], 1)[0] for i in range(self.parametros.NV)]

                amostra_inf = [triangular(limite_inferior[i], (limite_inferior[i]+self.parametros.estimativa[i])/2, self.parametros.estimativa[i], 1)[0] for i in range(self.parametros.NV)]

                amostra_sup = [triangular(self.parametros.estimativa[i], (limite_superior[i] + self.parametros.estimativa[i]) / 2, limite_superior[i], 1)[0] for i in range(self.parametros.NV)]

                amostra = [amostra_total,amostra_inf,amostra_sup, amostra_total_uni]

                FO = [float(self._excObjectiveFunction(amo_i, self._values)) for amo_i in amostra] #self._excFO returns a DM object, it's necessary convert to float object

                for i,FO_i in enumerate(FO):
                    self.__decisonVariablesMapped.append(amostra[i])
                    self.__OFMapped.append(FO_i)

    def __criteriosAbrangencia(self):
        u"""
        Método que retorna os valores das distribuições de Fisher, chi2 e o valor limite da função objetivo.
        Utilizado para avaliar a região de abrangẽncia
        """

        # TesteF = F(PA,NP,NE*NY-NP)
        fisher = f.ppf(self.PA,self.parametros.NV,(self.y.estimacao.NE*self.y.NV-self.parametros.NV))

        # Valor para a ellipse de abrangência:
        ellipseComparacao = self.FOotimo*(float(self.parametros.NV)/(self.y.estimacao.NE*self.y.NV-float(self.parametros.NV))*fisher)

        return fisher, ellipseComparacao

    def regiaoAbrangencia(self):
        u"""
        Método para avaliação da região de abrangência pelo critério de Fisher, conhecidas
        como região de verossimilhança [1].

        ==========
        Referência
        ==========
        [1] SCHWAAB, M. et al. Nonlinear parameter estimation through particle swarm optimization. Chemical Engineering Science, v. 63, n. 6, p. 1542–1552, mar. 2008.
        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('regiaoAbrangencia')

        # ---------------------------------------------------------------------
        # DETERMINAÇÃO DA REGIÃO DE ABRANGÊNCIA PELO CRITÉRIO DE FISHER
        # ---------------------------------------------------------------------
        fisher, ellipseComparacao = self.__criteriosAbrangencia()

        # Comparação dos valores da função objetivo avaliados na etapa de otimização com FOcomparacao, caso
        # sejam menores, os respectivos parâmetros estarão contidos da região de abrangência.
        regiao = []
        for pos,OFMapped in enumerate(self.__OFMapped):
            if OFMapped <= ellipseComparacao+self.FOotimo:
                regiao.append(self.__decisonVariablesMapped[pos])

        # ---------------------------------------------------------------------
        # AVALIAÇÃO SE A REGIÃO DE ABRANGÊNCIA NÃO ESTÁ VAZIA (Warning)
        # ---------------------------------------------------------------------
        if regiao == []:
            warn('The coverage region evaluated by the likelihood method contains no points. Review the parameters of the algorithm used..',UserWarning)

        return regiao

    def residualAnalysis(self, report=True, **kwargs):
        u"""
        Método para realização da análise de resíduos.
        A análise da sempre preferência aos dados de validação.

        ======
        Input
        ======
        * When report is true the prediction report includes statistical tests

        ======
        Saídas
        ======
        * Cálculo do R2 e R2 ajustado (atributos: R2 e R2ajustado)
        * Aplicação de testes estatísticos para as grandezas. Criação do atributo estatisticas para cada grandeza x e y. (Vide documentação de Grandeza)
        * Teste estatítico para avaliar se a função objetivo segue uma chi2 (atributo estatisticas)
              * Se FO pertence ao intervalo chi2min < FO < chi2max, tem uma situação ideal, então o modelo representa bem os dados

               Caso contrário há duas situações possíveis ao se analisar a FO com a chi2:

              * FO < chi2min, O modelo representa os dados esperimentais muito melhor que o esperado,
              o que pode indicar que há super parametrização do modelo ou que os erros esperimentais estão superestimados:
              * FO > chi2max: o modelo não é capaz de explicar os erros experimentais
              ou pode haver subestimação dos erros esperimentais

        ========
        Keywords
        ========
        * See documentation of Relatorio.Predicao

        """
        # ---------------------------------------------------------------------
        # FLUXO
        # ---------------------------------------------------------------------
        self.__controleFluxo.SET_ETAPA('analiseResiduos')
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         

        # Tamanho dos vetores:
        if self.y.predicao.NE != self.y.calculado.NE:
            raise TypeError(u'The length of the validation and calculated vectors are not consistent. Evaluate the need to perform the prediction method.')
        # ---------------------------------------------------------------------
        # CÁLCULO DOS RESÍDUOS
        # ---------------------------------------------------------------------          
        # Calculos dos residuos (ou desvios) - estão baseados nos dados de validação
        residuo_y = self.y.predicao.matriz_estimativa - self.y.calculado.matriz_estimativa
        residuo_x = self.x.predicao.matriz_estimativa - self.x.calculado.matriz_estimativa

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------       
        # Attribuição dos valores nos objetos
        self.x._SETresiduos(estimativa=residuo_x)
        self.y._SETresiduos(estimativa=residuo_y)

        # ---------------------------------------------------------------------
        # CÁLCULO DE R2 e R2 ajustado
        # ---------------------------------------------------------------------   
        self.estatisticas = {'R2': {}, 'R2ajustado': {}, 'FuncaoObjetivo': {}}
        # Para y:
        for i,symb in enumerate(self.y.simbolos):
            SSE = sum(self.y.residuos.matriz_estimativa[:,i]**2)
            SST = sum((self.y.predicao.matriz_estimativa[:,i]-\
                  mean(self.y.predicao.matriz_estimativa[:,i]))**2)
            self.estatisticas['R2'][symb]         = 1 - SSE/SST
            self.estatisticas['R2ajustado'][symb] = 1 - (SSE/(self.y.predicao.NE-self.parametros.NV))\
                                       /(SST/(self.y.predicao.NE - 1))
        # Para x:                                           
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
        # EXECUÇÃO DE TESTES ESTATÍSTICOS
        # ---------------------------------------------------------------------             
        # Grandezas independentes
        if self.__flag.info['reconciliacao']:
            self.x._testesEstatisticos(self.y.predicao.matriz_estimativa)

        # Grandezas dependentes            
        self.y._testesEstatisticos(self.x.predicao.matriz_estimativa)

        # ---------------------------------------------------------------------
        # VALIDAÇÃO DO VALOR DA FUNÇÃO OBJETIVO COMO UMA CHI 2
        # ---------------------------------------------------------------------
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
        Métodos para gerar e salvar os gráficos

        =======================
        Gráficos diponíveis
        =======================
            * 'regiaoAbrangencia': gráficos da região de abrangência dos parâmetros
            * 'grandezas-entrada': gráficos referentes aos dados de entrada e de validação
            * 'predicao': gráficos da predição
            * 'grandezas-calculadas': gráficos dos valores calculados de cada grandeza
            * 'otimizacao': gráficos referentes à otimização (depende do algoritmo utilizado)
            * 'analiseResiduos': gráficos referentes à análise de resíduos.
        """
        if kwargs.get('tipos') is None:
            tipos = []
            for fl_key in self.__graph_flux_association.keys():
                if getattr(self.__controleFluxo, fl_key):
                    tipos.extend(self.__graph_flux_association[fl_key])

        # Início da Figura que conterá os gráficos -> objeto
        Fig = Grafico(dpi=300)

        # ---------------------------------------------------------------------
        # CAMINHO BASE
        # ---------------------------------------------------------------------         
        base_path = self.__base_path + sep + self._configFolder['plots'] + sep

        # ---------------------------------------------------------------------
        # GRÁFICOS
        # ---------------------------------------------------------------------
        if (self.__tipoGraficos[1] in tipos):
            # se setConjunto foi executado alguma vez:
            if self.__controleFluxo.setConjunto:
                base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[1])] + sep
                Validacao_Diretorio(base_path,base_dir)
                # Pastas internas
                # ------------------------------------------------------------------------------------
                folder = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[1])] + sep + self._configFolder['plots-subfolder-DadosEstimacao']+ sep + self._configFolder['plots-subfolder-grandezatendencia']+sep
                Validacao_Diretorio(base_path, folder)
                # -----------------------------------------------------------------------------------
                # gráficos gerados para os dados experimentais
                if self.__flag.info['dadosestimacao'] == True:
                    self.x.Graficos(base_path, base_dir, ID=['estimacao'], fluxo=0, Fig=Fig)
                    self.y.Graficos(base_path, base_dir, ID=['estimacao'], fluxo=0, Fig=Fig)

                    # Gráficos das grandezas y em função de x
                    for iy in range(self.y.NV):
                        for ix in range(self.x.NV):
                            # Gráficos sem a incerteza
                            Fig.grafico_dispersao_sem_incerteza(self.x.estimacao.matriz_estimativa[:,ix],
                                                                self.y.estimacao.matriz_estimativa[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                marker='o', linestyle='None')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_em_funcao_de_'+self.x.simbolos[ix]+'_sem_incerteza')
                            # Gráficos com a incerteza
                            Fig.grafico_dispersao_com_incerteza(self.x.estimacao.matriz_estimativa[:,ix],
                                                                self.y.estimacao.matriz_estimativa[:,iy],
                                                                self.x.estimacao.matriz_incerteza[:,ix],
                                                                self.y.estimacao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                fator_abrangencia_x=2., fator_abrangencia_y=2., fmt='o')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_em_funcao_de_'+' '+self.x.simbolos[ix]+'_com_incerteza')
                # gráficos gerados para os dados de validação, apenas se estes forem diferentes dos experimentais,
                # apesar dos atributos de validação sempre existirem
                if self.__flag.info['dadospredicao'] == True:
                    # Pastas internas
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

                    # Gráficos das grandezas y em função de x
                    for iy in range(self.y.NV):
                        for ix in range(self.x.NV):
                            # Gráficos sem a incerteza
                            Fig.grafico_dispersao_sem_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                marker='o', linestyle='None')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_sem_incerteza')
                            # Gráficos com a incerteza
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.predicao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos('observado')[ix],
                                                                label_y=self.y.labelGraficos('observado')[iy],
                                                                fator_abrangencia_x=2., fator_abrangencia_y=2., fmt= 'o')
                            Fig.salvar_e_fechar(base_path+folder+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')
            else:
                warn('The input graphs could not be created because the setConjunto method was not executed.',UserWarning)

        # Gráficos referentes aos dados de saída (calculados)
        # grandezas-calculado
        if self.__tipoGraficos[3] in tipos:
            base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[3])] + sep
            Validacao_Diretorio(base_path, base_dir)

            # a incerteza dos parâmetros foi alguma vez executada
            if self.__controleFluxo.incertezaParametros:
                self.parametros.Graficos(base_path, base_dir, ID=['parametro'], fluxo=self.__controleFluxo.FLUXO_ID)
            else:
                warn('Graphs involving only calculated quantities (PARAMETERS) could not be created because the parametersUncertainty method was not executed.',UserWarning)

            # Predição deve ter sido executada no fluxo de trabalho
            if self.__controleFluxo.predicao:
                self.x.Graficos(base_path, base_dir, ID=['calculado'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)
                self.y.Graficos(base_path, base_dir, ID=['calculado'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

            else:
                warn('The graphs involving only the calculated quantities (X and Y) could not be created, because the prediction method was not executed.',UserWarning)

        # regiaoAbrangencia
        if self.__tipoGraficos[0] in tipos:
            # os gráficos da região de abrangência só são executados se a matriz de covariância
            # dos parâmetros existir.
            if self.__controleFluxo.incertezaParametros:
                # Gráficos da estimação
                if self.parametros.NV >1:
                    base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[0])] + sep
                    Validacao_Diretorio(base_path, base_dir)
                # os gráficos só podem ser executado se o número de parâmetros for
                # maior do que 1
                if self.parametros.NV != 1:
                    # numéro de combinações não repetidas para os parâmetros
                    Combinacoes = int(factorial(self.parametros.NV)/(factorial(self.parametros.NV-2)*factorial(2)))
                    p1 = 0; p2 = 1; cont = 0; passo = 1 # inicialiação dos contadores (pi e p2 são indinces dos parâmetros
                    # passo: contabiliza o número de parâmetros avaliados
                    # cont: contador que contabiliza (param.NV - passo1)+(param.NV - passo2)

                    for pos in range(Combinacoes):
                        if pos == (self.parametros.NV-1)+cont:
                            p1 +=1; p2 = p1+1; passo +=1
                            cont += self.parametros.NV-passo

                        # PLOT de região de abrangência pelo método da verossimilhança
                        if self.__controleFluxo.regiaoAbrangencia and self.parametros.regiao_abrangencia != []:
                            aux1 = [] # lista auxiliar -> região de abrangência para o parâmetro p1
                            aux2 = [] # lista auxiliar -> região de abrangência para o parâmetro p2
                            for it in range(int(size(self.parametros.regiao_abrangencia)/self.parametros.NV)):
                                aux1.append(self.parametros.regiao_abrangencia[it][p1])
                                aux2.append(self.parametros.regiao_abrangencia[it][p2])
                            Fig.grafico_dispersao_sem_incerteza(array(aux1), array(aux2),
                                                                add_legenda=True, corrigir_limites=False,
                                                                marker='o', linestyle='None', color='b', linewidth=2.0, zorder=1)
                        # PLOT da região de abrangência pelo método da linearização (elipse)
                        fisher, ellipseComparacao = self.__criteriosAbrangencia()

                        cov = array([[self.parametros.matriz_covariancia[p1,p1], self.parametros.matriz_covariancia[p1,p2]],
                                     [self.parametros.matriz_covariancia[p2,p1], self.parametros.matriz_covariancia[p2,p2]]])

                        Fig.elipse_covariancia(cov,[self.parametros.estimativa[p1],self.parametros.estimativa[p2]],ellipseComparacao)

                        if self.__controleFluxo.regiaoAbrangencia and self.parametros.regiao_abrangencia != []:
                            Fig.set_legenda([u'Verossimilhança','Elipse'], loc='best')
                        else:
                            Fig.set_legenda(['Elipse'], loc='best')

                        Fig.set_label(self.parametros.labelGraficos()[p1], self.parametros.labelGraficos()[p2])

                        # SALVA O GRÁFICO
                        Fig.salvar_e_fechar(base_path+base_dir+'regiao_verossimilhanca'+'_'+
                                    str(self.parametros.simbolos[p1])+'_'+str(self.parametros.simbolos[p2])+'.png',
                                            config_axes=True)
                        p2+=1
                else:
                    warn('The coverage region graphs could not be created, because there is only one parameter.',UserWarning)

            else:
                warn('The coverage region graphs could not be created because the uncertaintyParameters method was not run OR in the SETparameter method the parameters variance was not defined',UserWarning)

        # predição
        if self.__tipoGraficos[2] in tipos:
            # Predição deve ter sido executada neste fluxo
            if self.__controleFluxo.predicao:
                # Pastas internas
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
                #gráficos de y em função de y
                for iy in range(self.y.NV):
                    for ix in range(self.x.NV):
                        # Gráficos sem a incerteza
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

                        # Gráficos com a incerteza
                        if self.y.calculado.matriz_correlacao is not None:
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.calculado.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.calculado.matriz_incerteza[:,iy],
                                                                fator_abrangencia_x=2., fator_abrangencia_y=2., fmt='o',
                                                                color='b',add_legenda=True)
                            Fig.grafico_dispersao_com_incerteza(self.x.predicao.matriz_estimativa[:,ix],
                                                                self.y.predicao.matriz_estimativa[:,iy],
                                                                self.x.predicao.matriz_incerteza[:,ix],
                                                                self.y.predicao.matriz_incerteza[:,iy],
                                                                label_x=self.x.labelGraficos()[ix],
                                                                label_y=self.y.labelGraficos()[iy],
                                                                fator_abrangencia_x=2., fator_abrangencia_y=2., fmt='o',
                                                                color='r',add_legenda=True)
                            Fig.set_legenda(['calculado','observado'],loc='best', fontsize=12)
                            Fig.salvar_e_fechar(base_path+folderone+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')
                            #Fig.salvar_e_fechar(base_path+folderone+'calculado' +'_'+self.y.simbolos[iy]+'_funcao_'+self.x.simbolos[ix]+'_com_incerteza')


                #incerteza_expandida_Yc=ones((self.y.calculado.NE,self.y.NV))
                #incerteza_expandida_Ye=ones((self.y.validacao.NE,self.y.NV))
                # Fatores de abrangência para y de validação e o y calculado
                t_cal = -t.ppf((1-self.PA)/2, self.y.calculado.gL[0][0])
                t_val = -t.ppf((1-self.PA)/2, self.y.predicao.gL[0][0])

                for iy in range(self.y.NV):
                    y  = self.y.predicao.matriz_estimativa[:,iy]
                    ym = self.y.calculado.matriz_estimativa[:,iy]
                    amostras = arange(1,self.y.predicao.NE+1,1)

                    diagonal = linspace(min(y), max(y))
                    # Gráfico comparativo entre valores experimentais e calculados pelo modelo, sem variância
                    Fig.grafico_dispersao_sem_incerteza(y, ym, marker='o', linestyle='None',
                                                        corrigir_limites=False, config_axes=False)
                    Fig.grafico_dispersao_sem_incerteza(diagonal, diagonal, linestyle='-', color='k', linewidth = 2.0,
                                                        corrigir_limites=True, config_axes=False)
                    # Set_label possui o fontsize (tamanho das fontes nos eixos X x Y) definidos de acordo ao valor estabelecdo em Gráficos.
                    Fig.set_label(self.y.labelGraficos('observado')[iy] \
                                  if self.__flag.info['dadospredicao'] else self.y.labelGraficos('observado')[iy],
                                  self.y.labelGraficos('calculado')[iy])


                    Fig.salvar_e_fechar((base_path+foldertwo+'observado' if self.__flag.info['dadospredicao'] else base_path+foldertwo+'observado')+'_' + str(self.y.simbolos[iy])+'_funcao_'+str(self.y.simbolos[iy])+'_calculado_sem_incerteza.png',config_axes=True)


                    # Gráfico comparativo entre valores experimentais e calculados pelo modelo, sem variância em função
                    # das amostras
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

                    # Gráfico comparativo entre valores experimentais e calculados pelo modelo, com variância
                    if self.y.calculado.matriz_incerteza is not None:
                        yerr_calculado = self.y.calculado.matriz_incerteza[:,iy]

                        yerr_validacao = self.y.predicao.matriz_incerteza[:,iy]

                        # Gráfico comparativo entre valores experimentais (validação) e calculados pelo modelo, com variância
                        # Em função do número da amostra
                        Fig.grafico_dispersao_com_incerteza(amostras, y, None, yerr_validacao, fator_abrangencia_y=t_val,
                                                            fmt="o", color = 'b', config_axes=False, corrigir_limites=False,
                                                            add_legenda=True)
                        Fig.grafico_dispersao_com_incerteza(amostras, ym, None, yerr_calculado, fator_abrangencia_y=t_cal,
                                                            fmt="o", color = 'r', config_axes=False, add_legenda=True)
                        Fig.set_label('Amostras', self.y.labelGraficos()[iy])
                        Fig.set_legenda(['dados para predicao' if self.__flag.info['dadospredicao'] else 'dados para estimacao', 'calculado'],fontsize=12, loc='best')
                        Fig.salvar_e_fechar((base_path+foldertwo+'observado' if self.__flag.info['dadospredicao'] else base_path + foldertwo+'observado') + '_' + str(self.y.simbolos[iy]) +'_funcao_amostras_calculado_com_incerteza.png', config_axes=True)

                        # ycalculado em função de yexperimental
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
                                            reiniciar=(False if not self.__flag.info['dadospredicao'] else True)) # caso não tenha dados
                                            # de validação é aplicado um teste baseado no teste F que usará este gráfico.
                        # gráfico comparativo entre os valores experimentais (validação) e calculados pelo modelo, com variância
                        # em função das amostras

                        # gráficos do teste F
                        if not self.__flag.info['dadospredicao']:
                            # Gráfico do Teste F
                            ycalc_inferior_F = []
                            ycalc_superior_F = []
                            for iNE in range(self.y.calculado.NE):

                                ycalc_inferior_F.append(self.y.calculado.matriz_estimativa[iNE,iy]+\
                                            t_val\
                                            *(f.ppf((self.PA+(1-self.PA)/2),self.y.calculado.gL[iy][iNE],\
                                            self.y.predicao.gL[iy][iNE])*self.y.predicao.matriz_covariancia[iNE,iNE])**0.5)

                                ycalc_superior_F.append(self.y.calculado.matriz_estimativa[iNE,iy]-t_val\
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

        # AnáliseResiduos
        if (self.__tipoGraficos[5] in tipos):
            # o método análise de resíduos deve ter sido executado
            if self.__controleFluxo.analiseResiduos:
                base_dir = sep + self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] + sep
                Validacao_Diretorio(base_path,base_dir)
                # Gráficos relacionados aos resíduos das grandezas independentes, caso
                # seja realizada a reconciliação
                if self.__flag.info['reconciliacao'] == True:
                    self.x.Graficos(base_path, base_dir, ID=['residuo'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

                # Gráficos relacionados aos resíduos das grandezas dependentes
                self.y.Graficos(base_path, base_dir, ID=['residuo'], fluxo=self.__controleFluxo.FLUXO_ID, Fig=Fig)

                # Grafico dos resíduos em função dos dados de validação (ou experimentais) e calculados
                for i,simb in enumerate(self.y.simbolos):
                    # Pastas internas
                    # ------------------------------------------------------------------------------------
                    if self.__controleFluxo.FLUXO_ID == 0:
                        folder = self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] +  sep +self._configFolder['plots-subfolder-DadosEstimacao']+ sep + self.y.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    else:
                        folder = self._configFolder['plots-{}'.format(self.__tipoGraficos[5])] + sep + self._configFolder['plots-subfolder-Dadosvalidacao']+' '+str(self.__controleFluxo.FLUXO_ID)+ sep + self.y.simbolos[i] + sep
                        Validacao_Diretorio(base_path, folder)
                    # ------------------------------------------------------------------------------------
                    # Resíduos vs ycalculado
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

                    # Resíduos vs yvalidacao
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
                        #Resíduos vs. X estimacao/validacao
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
                warn('Graphs involving residue analysis could not be created because the residualAnalysis method was not executed.',UserWarning)

    def reports(self,**kwargs):
        u"""
        Método para criação do(s) relatório(s) com os principais resultados.

        ========
        Keywords
        ========
        * Vide documentação de Relatorio.Predicao
        """
        # ---------------------------------------------------------------------
        # DEFINIÇÃO DA CLASSE
        # ---------------------------------------------------------------------
        #out = Report(str(self.__controleFluxo.FLUXO_ID),self.__base_path,sep + self._configFolder['report']+ sep, **kwargs)

        # ---------------------------------------------------------------------
        # RELATÓRIO DOS PARÂMETROS
        # ---------------------------------------------------------------------
        # Caso a otimização ou SETParametros tenha sido executado, pode-se fazer um relatório sobre os parâmetros
        if self.__controleFluxo.otimizacao or self.__controleFluxo.SETparametro:
            self._out.Parametros(self.parametros,self.FOotimo)
        else:
            warn('The parameter report was not created because the optimize method or SETparameter method was not executed')
        # ---------------------------------------------------------------------
        # RELATÓRIO DA PREDIÇÃO E ANÁLISE DE RESÍDUOS
        # ---------------------------------------------------------------------
        # Caso a Predição tenha sido executada, pode-se fazer um relatório sobre a predição
        if self.__controleFluxo.predicao:
            # Caso a Análise de resíduos tenha sido executada, pode-se fazer um relatório completo
            kwargs['PA'] = self.PA
            if self.__controleFluxo.analiseResiduos:
                self._out.Predicao(self.x,self.y,self.estatisticas,**kwargs)
            else:
                self._out.Predicao(self.x,self.y,None,**kwargs)
                warn('The residue analysis report has not been created because the residualAnalysis method has not been carried out. However, you can still export the prediction')
        else:
            warn('The report on the prediction and residual analysis was not created because the prediction method was not executed')

