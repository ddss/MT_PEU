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

from numpy.linalg import inv

# Rotinas Internas
from MT_PEU import EstimacaoNaoLinear

from PSO.PSO import PSO # Deve haver uma pasta de nome PSO com os código fonte

from Funcao_Objetivo import WLS

# Fim da importação
class Modelo(Thread):
    result = 0
    def __init__(self,param,x,args,**kwargs):
        Thread.__init__(self)

        self.param  = array(param,ndmin=2).transpose()

        self.x      = x

        self.args  = args

        # LIDAR COM EXCEPTIONS THREAD
        self.bucket = kwargs.get('bucket')


    def runEquacoes(self):

        self.result = self.x.dot(self.param)

    def run(self):

        if self.bucket == None:
            self.runEquacoes()
        else:
            try:
                self.runEquacoes()
            except:
                self.bucket.put(exc_info())


class EstimacaoLinear(EstimacaoNaoLinear):
    
    def __init__(self,simbolos_y,simbolos_x,simbolos_param,projeto='Projeto',**kwargs):
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
        * ``projeto`` (string)      : nome do projeto (Náo podem haver caracteres especiais)
        
        **AVISO**:
        Para cálculo do coeficiente linear, basta que o número de parâmetros seja igual ao número de grandezas
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
        
        * ``gerarEntradas``        : método para incluir dados obtidos de experimentos. Neste há a opção de determinar \
        se estes dados serão utilizados como dados para estimar os parâmetros ou para validação. (Vide documentação do método)
        * ``otimiza``              : método para realizar a otimização, com base nos dados fornecidos em gerarEntradas.
        * ``incertezaParametros``  : método que avalia a incerteza dos parâmetros (Vide documentação do método)   
        * ``gerarEntradas``        : (é opcional para inclusão de dados de validação)
        * ``Predicao``             : método que avalia a predição do modelo e sua incerteza ou utilizando os pontos experimentais ou de \
        validação, se disponível (Vide documentação do método) 
        * ``analiseResiduos``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``graficos``             : método para criação dos gráficos (Vide documentação do método)
        * ``_armazenarDicionario`` : método que returna as grandezas sob a forma de um dicionário (Vide documentação do método)
        
        ====================
        Fluxo de trabalho        
        ====================
        
        Esta classe valida a correta ordem de execução dos métodos. É importante salientar que cada vez que o método ``gerarEntradas`` \
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
            * ``experimental`` : referente aos dados experimentais. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``calculado``    : referente aos dados calculados pelo modelo. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
            * ``validacao``    : referente aos dados de validação. Principais atributos: ``matriz_estimativa``, ``matriz_covariancia``
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

        EstimacaoNaoLinear.__init__(self,WLS,Modelo,simbolos_y,simbolos_x,simbolos_param,projeto,**kwargs)

        self._EstimacaoNaoLinear__flag.setCaracteristica(['calc_termo_independente'])

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if self.y.NV != 1:
            raise ValueError(u'Está apenas implementado estimação de parâmetros de modelos lineares MISO')

        if (self.parametros.NV != self.x.NV) and (self.parametros.NV != self.x.NV+1):
            raise ValueError(u'O número de parâmetros deve ser: igual ao de grandezas independentes (não é efetuado cálculo do coeficiente linear)'+\
            'OU igual ao número de grandezas independentes + 1 (é calculado o coeficiente linear).')

        # ---------------------------------------------------------------------
        # Definindo se o b será calculado
        # ---------------------------------------------------------------------
        if (self.parametros.NV == self.x.NV+1):
            self._EstimacaoNaoLinear__flag.ToggleActive('calc_termo_independente')

    def gerarEntradas(self,x,y,ux,uy,glx=[],gly=[],tipo='experimental',uxy=None):
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
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        # Validação dos dados de entrada x, y, xval e yval - É tomado como referência
        # a quantidade de observações das variáveis x.
        self._EstimacaoNaoLinear__validacaoDadosEntrada(x  ,ux   ,self.x.NV,size(x,0)) 
        self._EstimacaoNaoLinear__validacaoDadosEntrada(y  ,uy   ,self.y.NV,size(x,0))
        
        self._EstimacaoNaoLinear__validacaoArgumentosEntrada('gerarEntradas',None,tipo)       
         
        # ---------------------------------------------------------------------
        # MODIFICAÇÕES DAS MATRIZES DE DADOS 
        # ---------------------------------------------------------------------
        # Adiciona uma coluna de zeros nos dados experimentais para possibilitar
        # o cálculo do termo independente
        if self._EstimacaoNaoLinear__flag.info['calc_termo_independente']:
            x               = hstack((x,ones((shape(x)[0],1))))
            ux              = hstack((ux,ones((shape(x)[0],1))))


        if tipo == 'experimental':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadosexperimentais')
            if not self._EstimacaoNaoLinear__etapasID == 0: # Se a execução do motor de Cálculo não for a primeira, é
                self._EstimacaoNaoLinear__novoFluxo(reiniciar=True) # Inclusão de novo fluxo

            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados experimentais nas variáveis.
            self.x._SETexperimental(x,ux,glx,{'estimativa':'matriz','incerteza':'incerteza'})
            self.y._SETexperimental(y,uy,gly,{'estimativa':'matriz','incerteza':'incerteza'})

            # ---------------------------------------------------------------------
            # LISTA DE ATRIBUTOS A SEREM INSERIDOS NA FUNÇÃO OBJETIVO
            # ---------------------------------------------------------------------

            self._EstimacaoNaoLinear__args_model = [self.y.experimental.vetor_estimativa, self.x.experimental.matriz_estimativa,
                                                    self.y.experimental.matriz_covariancia, self.x.experimental.matriz_covariancia,
                                                    None, self._EstimacaoNaoLinear__modelo,
                                                    self.x.simbolos, self.y.simbolos, self.parametros.simbolos]

        
        if tipo == 'validacao':
            self._EstimacaoNaoLinear__flag.ToggleActive('dadosvalidacao')
            self._EstimacaoNaoLinear__novoFluxo() # Variável para controlar a execução dos métodos PEU
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            self.x._SETvalidacao(x,ux,glx,{'estimativa':'matriz','incerteza':'incerteza'})
            self.y._SETvalidacao(y,uy,gly,{'estimativa':'matriz','incerteza':'incerteza'})


        if self._EstimacaoNaoLinear__flag.info['dadosvalidacao'] == False:
            # Caso gerarEntradas seja executado somente para os dados experimentais,
            # será assumido que estes são os dados de validação, pois todos os cálculos 
            # de predição são realizados para os dados de validação.
            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZAS
            # ---------------------------------------------------------------------
            # Salvando os dados de validação.
            self.x._SETvalidacao(x,ux,glx,{'estimativa':'matriz','incerteza':'incerteza'})
            self.y._SETvalidacao(y,uy,gly,{'estimativa':'matriz','incerteza':'incerteza'})
            
        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------         
        # Inclusão desta etapa da lista de etapas
        self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID].append(self._EstimacaoNaoLinear__etapasdisponiveis[1]) 
        
    def otimiza(self):
        u'''
        Método para obtenção da estimativa dos parâmetros e sua matriz de covariância.
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if (self._EstimacaoNaoLinear__etapasdisponiveis[1] not in self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID]) or (self._EstimacaoNaoLinear__flag.info['dadosexperimentais']==False):
                raise TypeError(u'Para executar a otimização, faz-se necessário primeiro executar método %s informando os dados experimentais.'%(self._EstimacaoNaoLinear__etapasdisponiveis[1],))

        if self._EstimacaoNaoLinear__etapasdisponiveis[8] in self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID]:
            raise TypeError(u'O método %s não pode ser executado com %s.'%(self._EstimacaoNaoLinear__etapasdisponiveis[2], self._EstimacaoNaoLinear__etapasdisponiveis[8]))

        # ---------------------------------------------------------------------
        # RESOLUÇÃO
        # ---------------------------------------------------------------------
        X   = self.x.experimental.matriz_estimativa
        Uyy = self.y.experimental.matriz_covariancia
        y   = self.y.experimental.vetor_estimativa
        variancia = inv(X.transpose().dot(inv(Uyy)).dot(X))
        parametros = variancia.dot(X.transpose().dot(inv(Uyy))).dot(y)

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZA
        # ---------------------------------------------------------------------

        self.parametros._SETparametro(parametros.transpose()[0].tolist(),variancia,None)

        # ---------------------------------------------------------------------
        # FUNÇÃO OBJETIVO NO PONTO ÓTIMO
        # ---------------------------------------------------------------------

        self._EstimacaoNaoLinear__GETFOotimo()

        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------         
        # Inclusão desta etapa da lista de etapas
        self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID].append(self._EstimacaoNaoLinear__etapasdisponiveis[2])
        # Inclusão da incertezaParametros na lista de etapas
        self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID].append(self._EstimacaoNaoLinear__etapasdisponiveis[3])

    def incertezaParametros(self,PA=0.95,**kwargs):
        u'''
        Método para avaliar a região de abrangência dos parâmetros.

        **Observação**:
        A matriz de covariância dos parâmetros é calculada juntamente com a otimização, por ser parte constituinte da solução analítica. Entretanto,
        caso o método SETparametros seja executado e neste não seja definida a matriz de covariância, ela é calculada.

        =======
        Entrada
        =======

        PA: Probabilidade de abragência (deve ser um número entre 0 e 1)


        ========
        Keywords
        ========
        Keywords para o algoritmo de PSO responsável por avaliar a região de abrangência
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if (self._EstimacaoNaoLinear__etapasdisponiveis[2] not in self._EstimacaoNaoLinear__etapasGlobal()) and (self._EstimacaoNaoLinear__etapasdisponiveis[8] not in self._EstimacaoNaoLinear__etapasGlobal()):
                raise TypeError(u'Para executar a avaliação da incerteza dos parâmetros, faz-se necessário primeiro executar método %s ou %s.'%(self._EstimacaoNaoLinear__etapasdisponiveis[2],self._EstimacaoNaoLinear__etapasdisponiveis[8]))

        # ---------------------------------------------------------------------
        # CÁLCULO DA MATRIZ DE COVARIÂNCIA
        # ---------------------------------------------------------------------
        # Caso a matriz de covariância não seja calculada, ela será aqui calculada
        if self.parametros.matriz_covariancia is None:
            X   = self.x.experimental.matriz_estimativa
            Uyy = self.y.experimental.matriz_covariancia
            variancia = inv(X.transpose().dot(inv(Uyy)).dot(X))
            self.parametros._SETparametro(self.parametros.estimativa, variancia, self.parametros.regiao_abrangencia)

            # ---------------------------------------------------------------------
            # VARIÁVEIS INTERNAS
            # ---------------------------------------------------------------------
            self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID].append(self._EstimacaoNaoLinear__etapasdisponiveis[3])

        # ---------------------------------------------------------------------
        # CÁLCULO DA REGIÃO DE ABRANGÊNCIA
        # ---------------------------------------------------------------------
        # A região de abrangência só é calculada caso não esteja definida
        if self.parametros.regiao_abrangencia is None:

            regiao = self.regiaoAbrangencia(PA,**kwargs)

            # ---------------------------------------------------------------------
            # ATRIBUIÇÃO A GRANDEZA
            # ---------------------------------------------------------------------
            self.parametros._SETparametro(self.parametros.estimativa, self.parametros.matriz_covariancia, regiao)


    def regiaoAbrangencia(self,PA=0.95,**kwargs):
        u'''
        Método para cálculo da região de abrangência de verossimilhança. 

        PA: probabilidade de abrangência
        kwargs: argumentos para o algoritmo de PSO

        '''

        # ---------------------------------------------------------------------
        # KEYWORDS
        # ---------------------------------------------------------------------
        # Atributos obrigatórios
        sup = kwargs.get('sup')
        inf = kwargs.get('inf')

        if sup == None:
            sup = (self.parametros.estimativa + 5*t.ppf(PA+(1-PA)/2,100)*self.parametros.matriz_incerteza).transpose().tolist()[0]
        else:
            del kwargs['sup'] # retira sup dos argumentos extras

        if inf == None:
            inf = (self.parametros.estimativa - 5*t.ppf(PA+(1-PA)/2,100)*self.parametros.matriz_incerteza).transpose().tolist()[0]
        else:
            del kwargs['inf'] # retira inf dos argumentos extras

        if kwargs.get('itmax') == None:
            kwargs['itmax'] = 300

        # Separação de keywords para os diferentes métodos
        # keywarg para a etapa de busca:
        kwargsbusca = {}
        if kwargs.get('printit')  != None:
            kwargsbusca['printit'] = kwargs.get('printit')
            del kwargs['printit']

        self.Otimizacao = PSO(sup,inf,args_model=self._EstimacaoNaoLinear__args_model,**kwargs)
        self.Otimizacao.Busca(self._EstimacaoNaoLinear__FO,**kwargsbusca)

        # ---------------------------------------------------------------------
        # HISTÓRICO DA OTIMIZAÇÃO
        # ---------------------------------------------------------------------
        self._EstimacaoNaoLinear__hist_Posicoes = []; self._EstimacaoNaoLinear__hist_Fitness = []

        for it in xrange(self.Otimizacao.itmax):
            for ID_particula in xrange(self.Otimizacao.Num_particulas):
                self._EstimacaoNaoLinear__hist_Posicoes.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                self._EstimacaoNaoLinear__hist_Fitness.append(self.Otimizacao.historico_fitness[it][ID_particula])

        # Como o histórico da otimização foi avaliado nesta função e ele é requisito para o cálculo da região
        # de abrangência, ele foi adicinado à lista de etapas.

        self._EstimacaoNaoLinear__etapas[self._EstimacaoNaoLinear__etapasID].append(self._EstimacaoNaoLinear__etapasdisponiveis[12])

        return EstimacaoNaoLinear.regiaoAbrangencia(self, PA)
