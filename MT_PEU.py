# -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# Importação de pacotes de terceiros
from numpy import array, transpose, concatenate,size, diag, linspace, min, max, \
sort, argsort, mean,  std, amin, amax, copy, cos, sin, radians, mean

from scipy.stats import f, chi2

from scipy.misc import factorial
from numpy.linalg import inv
from math import floor, log10
from statsmodels.graphics.regressionplots import influence_plot

from matplotlib import use
use('Agg')

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot

from os import getcwd, sep

# Subrotinas próprias e adaptações (desenvolvidas pelo GI-UFBA)
from Grandeza import Grandeza
from subrotinas import Validacao_Diretorio, plot_cov_ellipse, vetor_delta
from PSO import PSO


class Estimacao:
    
    def __init__(self,FO,Modelo,simbolos_y,simbolos_x,simbolos_param,projeto='Projeto',**kwargs):
        u'''
        Classe para executar a estimação de parâmetros        
        
        =======================
        Entradas (obrigatórias)
        =======================
        * ``FO`` (Thread)           : objeto função objetivo
        * ``Modelo`` (Thread)       : objeto modelo. O modelo deve retornar um array com número de colunas igual ao número de y.
        * ``simbolos_y`` (list)     : lista com os simbolos das variáveis y (Não podem haver caracteres especiais)
        * ``simbolos_x`` (list)     : lista com os simbolos das variáveis x (Não podem haver caracteres especiais)
        * ``simbolos_param`` (list) : lista com o simbolos dos parâmetros (Não podem haver caracteres especiais)
        * ``projeto`` (string)      : nome do projeto (Náo podem haver caracteres especiais)
        
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
        
        =======        
        Métodos
        =======
        
        Para a realização da estimação de parâmetros de um certo modelo faz-se necessário executar \
        alguns métodos, na ordem indicada:
        
        * ``gerarEntradas``        : método para incluir os dados experimentais (Vide documentação do método)
        * ``otimiza``              : método para realizar a otimização, com base nos dados fornecidos em gerarEntradas. \
        Após a execução deste método, o valor as grandezas de saída são avaliadas para os pontos de validação. (Vide documentação do método)
        * ``incertezaParametros``  : método que avalia a incerteza dos parâmetros (Vide documentação do método)   
        * ``analiseResiduos``      : método para executar a análise de resíduos (Vide documentação do método)
        * ``graficos``             : método para criação dos gráficos (Vide documentação do método)
        * ``_armazenarDicionario`` : método que returna as grandezas sob a forma de um dicionário (Vide documentação do método)
        
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
        
        ===============
        Função objetivo
        ===============
        
        A função objetivo deve ser um objeto com uma estrutura específica, conforme detalha \
        o exemplo: ::
                
            class WLS(Thread):
                result = 0
                def __init__(self,p,argumentos):
                    Thread.__init__(self)
            
                    self.param  = p
                    
                    self.y      = argumentos[0]
                    self.x      = argumentos[1]
                    self.Vy     = argumentos[2]
                    self.Vx     = argumentos[3]        
                    self.args   = argumentos[4]
                    
                def run(self):
            
                    ym = Modelo(self.param,self.x,self.args)
                    ym.start()
                    ym.join()
                    ym = matriz2vetor(ym.result)
                    d     = self.y - ym
                    self.result =  float(dot(dot(transpose(d),linalg.inv(self.Vy)),d))            
        
        O Motor de cálculo sempre irá enviar como uma lista argumentos para a função objetivo \
        nesta ordem: 
        
        * vetor com as variáveis dependentes
        * vetor com as variáveis independentes
        * incerteza das variáveis dependentes
        * incerteza das variáveis independentes
        * argumentos extras para o modelo
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÕES GERAIS DE KEYWORDS
        # ---------------------------------------------------------------------
        self.__etapasdisponiveis = ['__init__','gerarEntradas','otimizacao',\
                                    'incertezaParametros','regiaoAbrangencia',\
                                    'analiseResiduos','armazenarDicionario'] # Lista de etapas que o algoritmo irá executar
        self.__validacaoArgumentosEntrada('__init__',kwargs,projeto)
        
        # ---------------------------------------------------------------------
        # INICIALIZAÇÃO DAS GRANDEZAS
        # ---------------------------------------------------------------------
        # Variável      = Grandeza(simbolos      ,nomes                                ,unidades                             ,label_latex)
        self.x          = Grandeza(simbolos_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        self.y          = Grandeza(simbolos_y    ,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        self.parametros = Grandeza(simbolos_param,kwargs.get(self.__keywordsEntrada[6]),kwargs.get(self.__keywordsEntrada[7]),kwargs.get(self.__keywordsEntrada[8]))
        
        # Verificação se os símbolos são distintos
        # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
        if len(set(self.y.simbolos).intersection(self.x.simbolos)) != 0 or len(set(self.y.simbolos).intersection(self.parametros.simbolos)) != 0 or len(set(self.x.simbolos).intersection(self.parametros.simbolos)) != 0:
            raise NameError('Os símbolos das grandezas devem ser diferentes.')
            
        # ---------------------------------------------------------------------
        # CRIAÇÃO DAS VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------
        # Função objetivo
        self.__FO        = FO
        # Modelo
        self.__modelo    = Modelo
        # Caminho base para os arquivos, caso seja definido a keyword base_path ela será utilizada.
        if kwargs.get(self.__keywordsEntrada[9]) == None:
            self.__base_path = getcwd()+ sep +str(projeto)+sep
        else:
            self.__base_path = kwargs.get(self.__keywordsEntrada[9]) 
        
        # Controle interno das etapas do algoritmo (métodos executados)
        self.__etapas            = [self.__etapasdisponiveis[0]] # Variável de armazenamento das etapas realizadas pelo algoritmo
            
             
    def __validacaoArgumentosEntrada(self,etapa,keywargs,args=None):
        u'''
        Método para verificação dos argumentos de entrada dos métodos de inicialização, otimização, incerteza dos parâmetros, \
        análise de resíduos, armazenar dicionário:
        
        * verificar se keywords foram corretamente definidas
        * verficar se keywords obtigatórias foram definidas
        * verificar se o método pode ser executado, validando as etapas predecessoras 
        * outras verificações
        '''
        # ---------------------------------------------------------------------
        # INICIALIZAÇÃO
        # --------------------------------------------------------------------- 
        # Keywords disponíveis        
        self.__keywordsEntrada  = ['nomes_x','unidades_x','label_latex_x','nomes_y','unidades_y','label_latex_y','nomes_param','unidades_param','label_latex_param','base_path'] # Keywords disponíveis para a entrada
        if etapa == self.__etapasdisponiveis[0]:
            # Validação se houve keywords digitadas incorretamente:
            keyincorreta  = [key for key in keywargs.keys() if not key in self.__keywordsEntrada]
        
            if len(keyincorreta) != 0:
                raise NameError('keyword(s) incorretas: '+', '.join(keyincorreta)+'.'+' Keywords disponíveis: '+', '.join(self.__keywordsEntrada)+'.')
    
            # Verificação se o nome do projeto é um string
            # args[0] = projeto
            if not isinstance(args,str):
                raise TypeError('O nome do projeto deve ser um string.')
            
            # Verificação se o nome do projeto possui caracteres especiais
            # set: conjunto de elementos distintos não ordenados (trabalha com teoria de conjuntos)
            if set('[~!@#$%^&*()_+{}":;\']+$').intersection(args):
                raise NameError('O nome do projeto não pode conter caracteres especiais')  
            
        # ---------------------------------------------------------------------
        # OTIMIZAÇÃO
        # --------------------------------------------------------------------- 
        # Keywords disponíveis        
        self.__AlgoritmosOtimizacao = ['PSO']        
        self.__keywordsOtimizacaoObrigatorias = {'PSO':['sup','inf'],'Nelder_Mead':[]}  

        if etapa == self.__etapasdisponiveis[2]:
            
            if self.__etapasdisponiveis[1] not in self.__etapas:
                raise TypeError(u'Para executar a otimização, faz-se necessário primeiro executar método %s.'%(self.__etapasdisponiveis[1],))
                
            # verificação se o algoritmo está disponível
            if (not args in self.__AlgoritmosOtimizacao) and  args != None:
                raise NameError(u'A opção de algoritmo não está correta. Algoritmos disponíveis: '+', '.join(self.__AlgoritmosOtimizacao)+'.')
            
            # Validação das keywords obrigatórias por algoritmo
            keyobrigatoria = [key for key in self.__keywordsOtimizacaoObrigatorias[args] if not key in keywargs.keys()]
                
            if len(keyobrigatoria) != 0:
                raise NameError(u'Para o método de %s a(s) keyword(s) obrigatória(s) não foram (foi) definida(s): '%(args,)+', '.join(keyobrigatoria)+'.')
        
            # validação se as keywords foram corretamente definidas
            if args == self.__AlgoritmosOtimizacao[0]:
                # verificação de os tamanhos das listas sup e inf são iguais ao número de parâmetros
                if (not isinstance(keywargs.get('sup'),list)) or (not isinstance(keywargs.get('inf'),list)):
                    raise TypeError(u'As keywords sup e inf devem ser LISTAS.')
                    
                if (len(keywargs.get('sup')) != self.parametros.NV) or (len(keywargs.get('inf')) != self.parametros.NV):
                    raise ValueError(u'As keywords sup e inf devem ter o mesmo tamanho do número de parâmetros, definido pelos símbolos. Número de parâmetros: %d'%(self.parametros.NV,))
                    
        # ---------------------------------------------------------------------
        # INCERTEZA DOS PARÂMETROS
        # --------------------------------------------------------------------- 
        self.__metodosIncerteza = ['2InvHessiana','geral']
        if etapa == self.__etapasdisponiveis[3]:
            if self.__etapasdisponiveis[2] not in self.__etapas:
                raise TypeError(u'Para executar a avaliação da incerteza dos parâmetros, faz-se necessário primeiro executar método %s.'%(self.__etapasdisponiveis[2],))
        
            if args not in self.__metodosIncerteza:
                raise NameError(u'O método solicitado para cálculo da incerteza dos parâmetros %s'%(args,)+' não está disponível. Métodos disponíveis '+', '.join(self.__metodosIncerteza)+'.')
        # ---------------------------------------------------------------------
        # ANÁLISE RESÍDUOS
        # --------------------------------------------------------------------- 
        if etapa == self.__etapasdisponiveis[5]:
            if self.__etapasdisponiveis[2] not in self.__etapas:
                raise TypeError(u'Para executar o método de análise de resíduos, faz-se necessário primeiro executar método %s.'%(self.__etapasdisponiveis[2],))
        
        # ---------------------------------------------------------------------
        # ARMAZENAR DICIONÁRIO
        # --------------------------------------------------------------------- 
        if etapa == self.__etapasdisponiveis[6]:
            if self.__etapasdisponiveis[1] not in self.__etapas:
                raise TypeError(u'Para executar o método armazenarDicionario, faz-se necessário primeiro executar método %s.'%(self.__etapasdisponiveis[1],))
           
    def __validacaoDadosEntrada(self,dados,udados,Ndados,NE):
        u'''
        Validação dos dados de entrada 
        
        * verificar se as colunas dos arrays de entrada são iguais aos nomes das variáveis definidas (y, x)
        * verificar se as grandezas têm o mesmo número de dados experimentais
        '''
        if size(dados,0) != NE:
            raise ValueError(u'Foram inseridos %d dados experimentais para uma grandeza e %d para outra'%(NE,size(dados,0)))
        
        if size(dados,1) != Ndados: 
            raise ValueError(u'O número de variáveis definidas foi %s, mas foram inseridos dados para %s variáveis.'%(Ndados,size(dados,1)))
            
        if size(udados,0) != NE:
            raise ValueError(u'Foram inseridos %d dados experimentais, mas incertezas para %d dados'%(NE,size(udados,0)))
        
        if size(udados,1) != Ndados: 
            raise ValueError(u'O número de variáveis definidas foi %s, mas foram inseridas incertezas para %s.'%(Ndados,size(udados,1)))
 

    def gerarEntradas(self,xe,ye,ux,uy,xval=None,yval=None,uxval=None,uyval=None,uxy=None):
        u'''
        Método para incluir os dados de entrada da estimação
        
        =======================
        Entradas (Obrigatórias)
        =======================        
        
        * xe        : array com os dados experimentais das variáveis independentes na forma de colunas
        * ux        : array com as incertezas das variáveis independentes na forma de colunas
        * ye        : array com os dados experimentais das variáveis dependentes na forma de colunas
        * uy        : array com as incertezas das variáveis dependentes na forma de colunas
        * xval      : array com os dados de validação para as variáveis independentes na forma de colunas
        * yval      : array com os dados de validação para as variáveis dependentes na forma de colunas
        * uxy       : covariância entre x e y
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO DOS DADOS DE VALIDAÇÃO
        # ---------------------------------------------------------------------
        # Se forem definidos dados de validação, eles devem estar completos com xval ,yval, uxval, uyval
        if (xval != None or yval != None)   and (uxval == None or uyval == None):
            raise ValueError(u'Caso seja definido dados de validação, é necessário fazê-lo para todas as grandezas. Assim, xval, yval, uxval e uyval devem ser definidos')

        if  (uxval != None or uyval != None) and (xval == None or yval == None):
            raise ValueError(u'Caso seja definido dados de validação, é necessário fazê-lo para todas as grandezas. Assim, xval, yval, uxval e uyval devem ser definidos')

        # Caso não definidos dados de validação, será assumido os valores experimentais                    
        if xval == None and yval == None:
            xval  = xe
            yval  = ye
            uxval = ux
            uyval = uy    
        
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        # Validação dos dados de entrada x, y, xval e yval - É tomado como referência
        # a quantidade de observações das variáveis x.
        self.__validacaoDadosEntrada(xe  ,ux   ,self.x.NV,size(xe,0)) 
        self.__validacaoDadosEntrada(ye  ,uy   ,self.y.NV,size(xe,0))
        self.__validacaoDadosEntrada(xval,uxval,self.x.NV,size(xval,0)) 
        self.__validacaoDadosEntrada(yval,uyval,self.y.NV,size(xval,0))

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------
        # Salvando os dados experimentais nas variáveis.
        self.x._SETexperimental(xe,ux,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETexperimental(ye,uy,{'estimativa':'matriz','incerteza':'incerteza'}) 
        
        # Salvando os dados de validação.
        self.x._SETvalidacao(xval,uxval,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETvalidacao(yval,uyval,{'estimativa':'matriz','incerteza':'incerteza'}) 

        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------         
        # Inclusão desta etapa da lista de etapas
        self.__etapas.append(self.__etapasdisponiveis[1]) 
        
    def _armazenarDicionario(self):
        u'''
        Método opcional para armazenar as Grandezas (x,y e parãmetros) na
        forma de um dicionário, cujas chaves são os símbolos.
        
        ======
        Saídas
        ======
        
        * grandeza: dicionário cujas chaves são os símbolos das grandezas e respectivos
        conteúdos objetos da classe Grandezas.
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         
        self.__validacaoArgumentosEntrada('armazenarDicionario',None,None)       

        # ---------------------------------------------------------------------
        # GERANDO O DICIONÁRIO
        # ---------------------------------------------------------------------    
     
        grandeza = {}        
        for j,simbolo in enumerate(self.y.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.y.nomes[j]],[self.y.unidades[j]],[self.y.label_latex[j]])
            if self.__etapasdisponiveis[1] in self.__etapas:
                # Salvando dados experimentais
                grandeza[simbolo]._SETexperimental(self.y.experimental.matriz_estimativa[:,j:j+1],self.y.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:
                # Salvando dados calculados
                grandeza[simbolo]._SETcalculado(self.y.calculado.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)
            if self.__etapasdisponiveis[5] in self.__etapas:
                # Salvando os resíduos
                grandeza[simbolo]._SETresiduos(self.y.residuos.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'})

        for j,simbolo in enumerate(self.x.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.x.nomes[j]],[self.x.unidades[j]],[self.x.label_latex[j]])
            if self.__etapasdisponiveis[1] in self.__etapas:
                # Salvando dados experimentais
                grandeza[simbolo]._SETexperimental(self.x.experimental.matriz_estimativa[:,j:j+1],self.x.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:
                # Salvando dados calculados
                grandeza[simbolo]._SETcalculado(self.x.calculado.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)
            if self.__etapasdisponiveis[5] in self.__etapas:
                # Salvando os resíduos
                grandeza[simbolo]._SETresiduos(self.x.residuos.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'})

        for j,simbolo in enumerate(self.parametros.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.parametros.nomes[j]],[self.parametros.unidades[j]],[self.parametros.label_latex[j]])
            if self.__etapasdisponiveis[2] in self.__etapas:
                # Salvando as informações dos parãmetros
                if self.parametros.matriz_covariancia == None:
                    grandeza[simbolo]._SETparametro(self.parametros.estimativa[j],None,None)
                else:
                    grandeza[simbolo]._SETparametro(self.parametros.estimativa[j],array([self.parametros.matriz_covariancia[j,j]],ndmin=2),None)

        return grandeza
    

    def otimiza(self,algoritmo='PSO',args=None,**kwargs):
        u'''
        Método para realização da otimização        
        
        =======================
        Entradas (Obrigatórias)
        =======================

        * algoritmo : string informando o algoritmo de otimização a ser utilizado. Cada algoritmo tem suas próprias keywords
        * args      : argumentos extras a serem passados para o modelo
        
        =======================
        Keywords (Obrigatórias)
        =======================
        
        algoritmo = PSO
        
        * sup           : limite superior de busca
        * inf           : limite inferior de busca
        
        Obs.: Para outros argumentos de entrada do PSO e keywords, verifique a documentação do método
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if not isinstance(algoritmo,str):
            raise TypeError(u'O algoritmo precisa ser uma string.')
            
        # Validação das keywords obrigatórias para o método de otimização
        self.__validacaoArgumentosEntrada('otimizacao',kwargs,algoritmo)       
 
        # ---------------------------------------------------------------------
        # DEFINIÇÃO DOS ARGUMENTOS EXTRAS PARA FUNÇÃO OBJETIVO
        # ---------------------------------------------------------------------
        self.__args_model = [self.y.experimental.vetor_estimativa,self.x.experimental.matriz_estimativa,\
        self.y.experimental.matriz_covariancia,self.x.experimental.matriz_covariancia,\
        args,self.x.simbolos,self.y.simbolos,self.parametros.simbolos]
        
        # ---------------------------------------------------------------------
        # ALGORITMOS DE OTIMIZAÇÃO
        # ---------------------------------------------------------------------        
        if algoritmo == 'PSO':
            # ---------------------------------------------------------------------
            # KEYWORDS
            # ---------------------------------------------------------------------
            # Atributos obrigatórios
            sup = kwargs.get('sup')
            inf = kwargs.get('inf')
            # Exclusão dos atributos obrigatórios da listas de keywords, mantendo somente
            # os opcionais
            del kwargs['sup']
            del kwargs['inf']
            # Separação de keywords para os diferentes métodos
            # keywarg para a etapa de busca:
            kwargsbusca = {}
            if kwargs.get('printit')  != None:
                kwargsbusca['printit'] = kwargs.get('printit')
                del kwargs['printit']
            # ---------------------------------------------------------------------
            # EXECUÇÃO OTIMIZAÇÃO
            # ---------------------------------------------------------------------
            # OS argumentos extras (kwargs e kwrsbusca) são passados diretamente para o algoritmo
            self.Otimizacao = PSO(sup,inf,args_model=self.__args_model,**kwargs)
            self.Otimizacao.Busca(self.__FO,**kwargsbusca)
            self.parametros._SETparametro(self.Otimizacao.gbest,None,None) # Atribuindo o valor ótimo dos parâemetros

        # ---------------------------------------------------------------------
        # PREDIÇÃO
        # ---------------------------------------------------------------------     
        # A predição é calculada com base nos dados de validação   
        aux = self.__modelo(self.parametros.estimativa,self.x.validacao.matriz_estimativa,[args,self.x.simbolos,self.y.simbolos,self.parametros.simbolos])
        aux.start()
        aux.join()
                
        # Salvando os resultados
        self.y._SETcalculado(aux.result,None,{'estimativa':'matriz','incerteza':'variancia'},None)
        self.x._SETcalculado(self.x.experimental.matriz_estimativa,self.x.experimental.matriz_incerteza,{'estimativa':'matriz','incerteza':'incerteza'},None)

        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------         
        # Inclusão desta etapa da lista de etapas
        self.__etapas.append(self.__etapasdisponiveis[2]) # Inclusão desta etapa da lista de etapas
         
    def incertezaParametros(self,PA=0.95,delta=1e-5,metodo='2InvHessiana'):       
        u'''
        Método para avaliação da matriz covariãncia dos parâmetros.
        
        =======================
        Entradas (opcionais)
        =======================
        * PA         : probabilidade de abrangência para gerar a região de abrangência
        * delta      : incremento para o cálculo das derivadas (derivada numérica)
        * Métodos disponíveis: 2InvHessiana, geral
        ======
        Saídas
        ======
        * a matriz de covariância é salva na Grandeza parâmetros
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         
        self.__validacaoArgumentosEntrada('incertezaParametros',None,metodo)       

        # ---------------------------------------------------------------------
        # CÁLCULO DA MATRIZ DE COVARIÂNCIA
        # ---------------------------------------------------------------------     
        if metodo == self.__metodosIncerteza[0]:      
            # Método: 2InvHessiana - >  2*inv(Hess)
            matriz_covariancia = 2*inv(self.__Hessiana_FO_Param(delta))
        
        elif metodo == self.__metodosIncerteza[1]:
            # Método: geral - > inv(H)*Gy*Uyy*GyT*inv(H)
            # Inversa da matriz hessiana
            invHess             = inv(self.__Hessiana_FO_Param(delta))
            # Gy: derivadas parciais segundas da função objetivo em relação aos parâmetros e 
            # dados experimentais
            Gy                  = self.__Matriz_Gy(delta) 
            # cálculo da matriz de covariância
            matriz_covariancia  = invHess.dot(Gy).dot(self.y.experimental.matriz_covariancia).dot(Gy.transpose()).dot(invHess)


        Regiao, Hist_Posicoes, Hist_Fitness = self.regiaoAbrangencia(PA) # método para avaliação da região de abrangência

        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------
        self.parametros._SETparametro(self.parametros.estimativa,matriz_covariancia,Regiao)
        
        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------
        self.parametros._SETparametro(self.parametros.estimativa,self.parametros.matriz_covariancia,Regiao)

        # ---------------------------------------------------------------------
        # VARIÁVEIS INTERNAS
        # ---------------------------------------------------------------------         
        # Inclusão desta etapa da lista de etapas                
        self.__etapas.append(self.__etapasdisponiveis[3])
        
    def __Hessiana_FO_Param(self,delta=1e-5):
        u'''
        Método para calcular a matriz Hessiana da função objetivo em relaçao aos parâmetros.
        
        Está disponível o método de derivada central.
        
        ========
        Entradas
        ========
        * delta: valor do incremento relativo para o cálculo da derivada. Incremento relativo à ordem de grandeza do parâmetro.

        =====
        Saída
        =====
        
        Retorna a matriz Hessiana
        '''
            
        matriz_hessiana=[[1. for col in range(self.parametros.NV)] for row in range(self.parametros.NV)]
        
        FO_otimo = self.Otimizacao.best_fitness # Valor da função objetivo no ponto ótimo

        for i in range(self.parametros.NV): 
            for j in range(self.parametros.NV):
                
                delta1 = (10**(floor(log10(abs(self.parametros.estimativa[i])))))*delta
                delta2 = (10**(floor(log10(abs(self.parametros.estimativa[j])))))*delta
                
                if i==j:
                    # Incrementos
                    vetor_parametro_delta_positivo = vetor_delta(self.parametros.estimativa,i,delta1) # Vetor que irá conter o incremento no parâmetro i
                    vetor_parametro_delta_negativo = vetor_delta(self.parametros.estimativa,j,-delta2) # Vetor que irá conter o incremento no parâmetro j

                    # Cálculo da função objetivo
                    FO_delta_positivo=self.__FO(vetor_parametro_delta_positivo,self.__args_model)
                    FO_delta_positivo.start()
                                     
                    FO_delta_negativo=self.__FO(vetor_parametro_delta_negativo,self.__args_model)
                    FO_delta_negativo.start()
                     
                    FO_delta_positivo.join()
                    FO_delta_negativo.join()                    
                    
                    # Fórmula de diferença finita para i=j (REF????)
                    matriz_hessiana[i][j]=(FO_delta_positivo.result-2*FO_otimo+FO_delta_negativo.result)/(delta1*delta2)
                     
                else:
                    
                    vetor_parametro_delta_ipositivo_jpositivo = vetor_delta(self.parametros.estimativa,[i,j],[delta1,delta2])
                    
                    FO_ipositivo_jpositivo=self.__FO(vetor_parametro_delta_ipositivo_jpositivo,self.__args_model)
                    FO_ipositivo_jpositivo.start()
                    
                    vetor_parametro_delta_inegativo_jpositivo=vetor_delta(self.parametros.estimativa,[i,j],[-delta1,delta2])
 
                    FO_inegativo_jpositivo=self.__FO(vetor_parametro_delta_inegativo_jpositivo,self.__args_model)
                    FO_inegativo_jpositivo.start()

                    vetor_parametro_delta_ipositivo_jnegativo=vetor_delta(self.parametros.estimativa,[i,j],[delta1,-delta2])
   
                    FO_ipositivo_jnegativo=self.__FO(vetor_parametro_delta_ipositivo_jnegativo,self.__args_model)
                    FO_ipositivo_jnegativo.start()

                    vetor_parametro_delta_inegativo_jnegativo=vetor_delta(self.parametros.estimativa,[i,j],[-delta1,-delta2])
                    
                    FO_inegativo_jnegativo=self.__FO(vetor_parametro_delta_inegativo_jnegativo,self.__args_model)
                    FO_inegativo_jnegativo.start()
                    
                    FO_ipositivo_jpositivo.join()
                    FO_inegativo_jpositivo.join()
                    FO_ipositivo_jnegativo.join()
                    FO_inegativo_jnegativo.join()
                    
                    # Referẽncia???
                    matriz_hessiana[i][j]=((FO_ipositivo_jpositivo.result-FO_inegativo_jpositivo.result)/(2*delta1)\
                    -(FO_ipositivo_jnegativo.result-FO_inegativo_jnegativo.result)/(2*delta1))/(2*delta2)
 
        return array(matriz_hessiana)
        
    def __Matriz_Gy(self,delta=1e-5):
        u'''
        Método para calcular a matriz Gy(derivada segunda da Fobj em relação aos parâmetros e y).
        
        Método de derivada central.
        
        ========
        Entradas
        ========
        * delta: valor do incremento relativo para o cálculo da derivada. Incremento relativo à ordem de grandeza do parâmetro.

        =====
        Saída
        ===== 
        Retorna a matriz Gy.
        '''
        
        matriz_Gy = [[1. for col in xrange(self.y.NV*self.y.experimental.NE)] for row in xrange(self.parametros.NV)]

        for i in xrange(self.parametros.NV): 
            for j in xrange(self.y.NV*self.y.experimental.NE):
                                
                delta1 = (10**(floor(log10(abs(self.parametros.estimativa[i])))))*delta #varia os elementos do vetor de thetas.
                delta2 = (10**(floor(log10(abs(self.y.experimental.vetor_estimativa[j])))))*delta #varia os elementos do vetor dos y experimentais.
                
                vetor_parametro_delta_ipositivo = vetor_delta(self.parametros.estimativa,i,delta1) #Vetor alterado dos parâmetros para entrada na função objetivo
                vetor_y_delta_jpositivo         = vetor_delta(self.y.experimental.vetor_estimativa,j,delta2)
                
                args                            = copy(self.__args_model).tolist()
                args[0]                         = vetor_y_delta_jpositivo
                
                FO_ipositivo_jpositivo          = self.__FO(vetor_parametro_delta_ipositivo,args) # Valor da _FO para vetor de parâmetros e Yexperimentais alterados.
                FO_ipositivo_jpositivo.start()
                
                
                vetor_parametro_delta_inegativo = vetor_delta(self.parametros.estimativa,i,-delta1)
 
                FO_inegativo_jpositivo          = self.__FO(vetor_parametro_delta_inegativo,args)
                FO_inegativo_jpositivo.start()
                
                vetor_y_delta_jnegativo         = vetor_delta(self.y.experimental.vetor_estimativa,j,-delta2) 
                args                            = copy(self.__args_model).tolist()
                args[0]                         = vetor_y_delta_jnegativo
   
                FO_ipositivo_jnegativo          = self.__FO(vetor_parametro_delta_ipositivo,args)
                FO_ipositivo_jnegativo.start()
                
                    
                FO_inegativo_jnegativo          = self.__FO(vetor_parametro_delta_inegativo,args)
                FO_inegativo_jnegativo.start()
                    
                FO_ipositivo_jpositivo.join()
                FO_inegativo_jpositivo.join()
                FO_ipositivo_jnegativo.join()
                FO_inegativo_jnegativo.join()
                    
                # Referẽncia???
                matriz_Gy[i][j]=((FO_ipositivo_jpositivo.result-FO_inegativo_jpositivo.result)/(2*delta1)\
                -(FO_ipositivo_jnegativo.result-FO_inegativo_jnegativo.result)/(2*delta1))/(2*delta2)
         
        
        return array(matriz_Gy)

    def regiaoAbrangencia(self,PA=0.95):
        u'''
        Método para avaliação da região de abrangência
        '''
        Fisher = f.ppf(PA,self.parametros.NV,(self.y.experimental.NE*self.y.NV-self.parametros.NV))            
        Comparacao = self.Otimizacao.best_fitness*(1+float(self.parametros.NV)/(self.y.experimental.NE*self.y.NV-float(self.parametros.NV))*Fisher)
        
        Regiao = []; Hist_Posicoes = []; Hist_Fitness = []
        for it in xrange(self.Otimizacao.itmax):
            for ID_particula in xrange(self.Otimizacao.Num_particulas):
                if self.Otimizacao.historico_fitness[it][ID_particula] <= Comparacao:
                    Regiao.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Posicoes.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Fitness.append(self.Otimizacao.historico_fitness[it][ID_particula])
                
        self.__etapas.append(self.__etapasdisponiveis[4]) # Inclusão desta etapa da lista de etapas

        return (Regiao, Hist_Posicoes, Hist_Fitness)
        
    def analiseResiduos(self,PA):
        u'''
        Método para realização da análise de resíduos.
        
        ========
        Entradas
        ========
        PA= Probabilidade de Abrângência
        
        ======
        Saídas
        ======
        
        * Saídas na forma de gráficos
        * As grandezas resíduos possuem o atributo "estatisticas".
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         
        self.__validacaoArgumentosEntrada('analiseResiduos',None,None)       

        # ---------------------------------------------------------------------
        # CÁLCULO DOS RESÍDUOS
        # ---------------------------------------------------------------------          
        # Calculos dos residuos (ou desvios) - estão baseados nos dados de validação
        residuo_y = self.y.validacao.matriz_estimativa - self.y.calculado.matriz_estimativa
        residuo_x = self.x.validacao.matriz_estimativa - self.x.calculado.matriz_estimativa
        
        # ---------------------------------------------------------------------
        # ATRIBUIÇÃO A GRANDEZAS
        # ---------------------------------------------------------------------       
        # Attribuição dos valores nos objetos
        self.x._SETresiduos(residuo_x,None,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETresiduos(residuo_y,None,{'estimativa':'matriz','incerteza':'incerteza'})

        # ---------------------------------------------------------------------
        # CÁLCULO DE R2 e R2 ajustado
        # ---------------------------------------------------------------------   
        self.R2         = {}
        self.R2ajustado = {}
        # Para y:
        for i,symb in enumerate(self.y.simbolos):
            SSE = sum(self.y.residuos.matriz_estimativa[:,i]**2)
            SST = sum((self.y.experimental.matriz_estimativa[:,i]-\
                  mean(self.y.experimental.matriz_estimativa[:,i]))**2)
            self.R2[symb]         = 1 - SSE/SST
            self.R2ajustado[symb] = 1 - (SSE/(self.y.experimental.NE-self.parametros.NV))\
                                       /(SST/(self.y.experimental.NE - 1))
        # Para x:                                           
#        for i,symb in enumerate(self.x.simbolos):
#            SSEx = sum(self.x.residuos.matriz_estimativa[:,i]**2)
#            SSTx = sum((self.x.experimental.matriz_estimativa[:,i]-\
#                  mean(self.x.experimental.matriz_estimativa[:,i]))**2)
#            self.R2[symb]         = 1 - SSEx/SSTx
#            self.R2ajustado[symb] = 1 - (SSEx/(self.x.experimental.NE-self.parametros.NV))\
#                                       /(SSTx/(self.x.experimental.NE - 1))


        # ---------------------------------------------------------------------
        # VALIDAÇÃO DO VALOR DA FUNÇÃO OBJETIVO COMO UMA CHI 2
        # ---------------------------------------------------------------------
        '''Há duas situações possíveis ao se analisar a FO com a chi2, se a FO pertence ao intervalo chi2min < FO < chi2max, 
           o modelo representa bem os dados, ou se FO não pertence a este intervalo, neste caso temos:
           FO < chi2min, O modelo representa os dados esperimentais muito melhor que o esperado,
           o que pode indicar que há super parametrização do modelo ou que os erros esperimentais estão superestimados:
           FO > chi2max: o modelo não é capaz de explicar os erros experimentais 
           ou pode haver subestimação dos erros esperimentais'''
        
        gL = self.y.experimental.NE*self.y.NV - self.parametros.NV
        
        chi2max = chi2.ppf(PA+(1-PA)/2,gL)
        chi2min = chi2.ppf((1-PA)/2,gL)
        
        if chi2min < self.Otimizacao.best_fitness < chi2max:
            result= u'O modelo representa bem o conjunto de dados'
        else:
            result= u'O modelo não representa bem o conjunto de dados'

        self.estatisticaFO = {'chi2max':chi2max,'chi2min':chi2min,'FO': (self.Otimizacao.best_fitness, result) }
        

        # ---------------------------------------------------------------------
        # EXECUÇÃO DE GRÁFICOS E TESTES ESTATÍSTICOS
        # ---------------------------------------------------------------------             
        # DESCOMENTAR QUANDO RECONCILIAÇÃO !!
        #self.x.Graficos(self.__base_path + sep + 'Graficos'  + sep,ID=['residuo'])
        #self.x._testesEstatisticos()
        self.y.Graficos(self.__base_path + sep + 'Graficos'  + sep,ID=['residuo'])
        self.y._testesEstatisticos(self.x.experimental.matriz_estimativa)
        
        # Gráficos que dependem de informações da estimação (y)
        # TO DO: RELOCAR PARA A SESSÃO DE GRÁFICOS

        base_path  = self.__base_path + sep + 'Graficos'  + sep
  
        for i,simb in enumerate(self.y.simbolos):         
            base_dir =  sep + 'Grandezas' + sep + self.y.simbolos[i] + sep
            # Gráficos da otimização
            Validacao_Diretorio(base_path,base_dir)  
        
            ymodelo = self.y.experimental.matriz_estimativa[:,i]
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            plot(ymodelo,self.y.residuos.matriz_estimativa[:,i], 'o')
            xlabel(u'Valores Ajustados '+self.y.labelGraficos()[i])
            ylabel(u'Resíduos '+self.y.labelGraficos()[i])
            ax.yaxis.grid(color='gray', linestyle='dashed')                        
            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.axhline(0, color='black', lw=2)
            fig.savefig(base_path+base_dir+'residuos_versus_ycalculado.png')
            close()

        self.__etapas.append(self.__etapasdisponiveis[5]) # Inclusão desta etapa na lista de etapas
        
#        for i,simb in enumerate(self.y.simbolos):
#            base_dir =  sep + 'Grandezas' + sep + self.y.simbolos[i] + sep
#            # Gráficos da otimização
#            Validacao_Diretorio(base_path,base_dir)  
#        
#            ymodelo = self.y.experimental.matriz_estimativa[:,i]
#            fig = figure()
#            ax = fig.add_subplot(1,1,1)
#            ax.set_ylabel("Modelo")
#            ax.set_xlabel("Residuos")
#            ax.set_title("")
#            influence_plot(self.y.residuos.matriz_estimativa[:,i],True, 0.05, 'cooks', 48, 0.75, ax)
#            fig.savefig(base_path+base_dir+'outlier.png')
#            close()
#
#        self.__etapas.append(self.__etapasdisponiveis[5]) # Inclusão desta etapa na lista de etapas
#        return self.estatisticaFO

    def graficos(self,lista_de_etapas,PA):
        u'''
        Métodos para gerar e salvar os gráficos
        =======================
        Entradas (obrigatórias)
        =======================
        * ``Etapa``   : Status que determinam se o método deve gerar os gráficos de entrada ou de otimização
        * ``PA``      : Probabilidade de Abragência
        ==========
        Atributos
        ==========
                
        '''
        self.__etapas.append(self.__etapasdisponiveis[5]) # Inclusão desta etapa da lista de etapas
           
        base_path  = self.__base_path + sep +'Graficos'+ sep
        
        #Sub-rotina que geram os gráficos de entrada e saída
        def graficos_entrada_saida(x,y,ux,uy,ix,iy,base_dir,info):            
            #Gráfico apenas com os pontos experimentais
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            plot(x,y,'o')
            # obtençao do tick do grafico
            # eixo x
            label_tick_x   = ax.get_xticks().tolist()                 
            tamanho_tick_x = (label_tick_x[1] - label_tick_x[0])/2
            # eixo y
            label_tick_y = ax.get_yticks().tolist()
            tamanho_tick_y = (label_tick_y[1] - label_tick_y[0])/2
            # Modificação do limite dos gráficos
            xmin   = min(x) - tamanho_tick_x
            xmax   = max(x) + tamanho_tick_x
            ymin   = min(y) - tamanho_tick_y
            ymax   = max(y) + tamanho_tick_y
            xlim(xmin,xmax)
            ylim(ymin,ymax)
            # Labels
            xlabel(self.x.labelGraficos(info)[ix],fontsize=20)
            ylabel(self.y.labelGraficos(info)[iy],fontsize=20)
            #Grades
            grid(b = 'on', which = 'major', axis = 'both')
            fig.savefig(base_path+base_dir+info+'_'+self.y.simbolos[iy]+'_funcao_de_'+self.x.simbolos[ix]+'_sem_incerteza')
            close()                    
                    
            #Grafico com os pontos experimentais e as incertezas
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            xerr = 2*ux
            yerr = 2*uy
            errorbar(x,y,xerr=xerr,yerr=yerr,marker='o')
            # obtençao do tick do grafico
            # eixo x
            label_tick_x   = ax.get_xticks().tolist()                 
            tamanho_tick_x = (label_tick_x[1] - label_tick_x[0])/2
            # eixo y
            label_tick_y = ax.get_yticks().tolist()
            tamanho_tick_y = (label_tick_y[1] - label_tick_y[0])/2
            # Modificação dos limites dos gráficos                    
            xmin  = min(x - xerr) - tamanho_tick_x
            ymin  = min(y - yerr) - tamanho_tick_y
            xmax  = max(x + xerr) + tamanho_tick_x                    
            ymax  = max(y + yerr) + tamanho_tick_y
            xlim(xmin,xmax)
            ylim(ymin,ymax)
            # Labels
            xlabel(self.x.labelGraficos(info)[ix],fontsize=20)
            ylabel(self.y.labelGraficos(info)[iy],fontsize=20)
            #Grades
            grid(b = 'on', which = 'major', axis = 'both')
            fig.savefig(base_path+base_dir+info+'_'+self.y.simbolos[iy]+'_funcao_de_'+self.x.simbolos[ix]+'_com_incerteza')
            close()
            
        #If para gerar os gráficos das grandezas de entrada (x e y)        
        if('entrada' in lista_de_etapas) and('gerarEntradas'in self.__etapas):
            base_dir = sep + 'Grandezas' + sep
            Validacao_Diretorio(base_path,base_dir)

            self.x.Graficos(base_path, ID = ['experimental','validacao'])
            self.y.Graficos(base_path, ID = ['experimental','validacao'])

            for iy in xrange(self.y.NV):
                for ix in xrange(self.x.NV):
                    x = self.x.experimental.matriz_estimativa[:,ix]
                    y = self.y.experimental.matriz_estimativa[:,iy]
                    ux = self.x.experimental.matriz_incerteza[:,ix]
                    uy = self.y.experimental.matriz_incerteza[:,iy]                    
                    graficos_entrada_saida(x,y,ux,uy,ix,iy,base_dir,'experimental')


        if('otimizacao' in lista_de_etapas) and('otimizacao' in self.__etapas):
            # Gráficos da otimização
            base_dir = sep+'PSO'+ sep
            Validacao_Diretorio(base_path,base_dir)
    
            self.Otimizacao.Graficos(base_path+base_dir,Nome_param=self.parametros.simbolos,Unid_param=self.parametros.unidades,FO2a2=True)


        if('estimacao' in lista_de_etapas) and('regiaoAbrangencia' in self.__etapas) and('analiseResiduos' in self.__etapas):
            # Gráficos da estimação
            base_dir = sep + 'Estimacao' + sep
            Validacao_Diretorio(base_path,base_dir)


            if self.parametros.NV == 1:
                
                # Região de abrangência (método da verossimilhança)
                Regiao, Hist_Posicoes, Hist_Fitness = self.regiaoAbrangencia(PA)
                    
                aux = []
                for it in xrange(size(self.parametros.regiao_abrangencia)/self.parametros.NV):     
                    aux.append(self.parametros.regiao_abrangencia[it][0])
                       
                X = [Hist_Posicoes[it][0] for it in xrange(len(Hist_Posicoes))]
                Y = Hist_Fitness
                X_sort   = sort(X)
                Y_sort   = [Y[i] for i in argsort(X)]    
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                plot(X_sort,Y_sort,'bo',markersize=4)
                plot(self.parametros.estimativa[0],self.Otimizacao.best_fitness,'ro',markersize=8)
                plot([min(aux),min(aux)],[min(Y_sort),max(Y_sort)/4],'r-')
                plot([max(aux),max(aux)],[min(Y_sort),max(Y_sort)/4],'r-')
                ax.text(min(aux),max(Y_sort)/4,u'%.2e'%(min(aux),), fontsize=8, horizontalalignment='center')
                ax.text(max(aux),max(Y_sort)/4,u'%.2e'%(max(aux),), fontsize=8, horizontalalignment='center')
                ax.yaxis.grid(color='gray', linestyle='dashed')
                ax.xaxis.grid(color='gray', linestyle='dashed')
                xlim((self.parametros.estimativa[0]-2.5*self.parametros.matriz_covariancia[0,0],self.parametros.estimativa[0]+2.5*self.parametros.matriz_covariancia[0,0]))
                ylabel(r"$\quad \Phi $",fontsize = 20)
                xlabel(self.parametros.labelGraficos()[0],fontsize=20)
                fig.savefig(base_path+base_dir+'regiao_verossimilhanca_'+str(self.parametros.simbolos[0])+'_'+str(self.parametros.simbolos[0])+'.png')
                close()
            
            else:
                
                Combinacoes = int(factorial(self.parametros.NV)/(factorial(self.parametros.NV-2)*factorial(2)))
                p1 = 0; p2 = 1; cont = 0; passo = 1
                
                for pos in xrange(Combinacoes):
                    if pos == (self.parametros.NV-1)+cont:
                        p1 +=1; p2 = p1+1; passo +=1
                        cont += self.parametros.NV-passo
                    
                    fig = figure()
                    ax = fig.add_subplot(1,1,1)
                    
                    for it in xrange(size(self.parametros.regiao_abrangencia)/self.parametros.NV):     
                        PSO, = plot(self.parametros.regiao_abrangencia[it][p1],self.parametros.regiao_abrangencia[it][p2],'bo',linewidth=2.0,zorder=1)
                    
                    Fisher = f.ppf(PA,self.parametros.NV,(self.y.experimental.NE*self.y.NV-self.parametros.NV))            
                    Comparacao = self.Otimizacao.best_fitness*(float(self.parametros.NV)/(self.y.experimental.NE*self.y.NV-float(self.parametros.NV))*Fisher)
                    cov = array([[self.parametros.matriz_covariancia[p1,p1],self.parametros.matriz_covariancia[p1,p2]],[self.parametros.matriz_covariancia[p2,p1],self.parametros.matriz_covariancia[p2,p2]]])
                    ellipse, width, height, theta = plot_cov_ellipse(cov, [self.parametros.estimativa[p1],self.parametros.estimativa[p2]], Comparacao, fill = False, color = 'r', linewidth=2.0,zorder=2)
                    plot(self.parametros.estimativa[p1],self.parametros.estimativa[p2],'r*',markersize=10.0,zorder=2)
                    ax.yaxis.grid(color='gray', linestyle='dashed')                        
                    ax.xaxis.grid(color='gray', linestyle='dashed')
                    xlabel(self.parametros.labelGraficos()[p1],fontsize=20)
                    ylabel(self.parametros.labelGraficos()[p2],fontsize=20)
                    if abs(theta)>=179.9:
                        hx = width / 2.
                        hy = height /2.
                    elif abs(theta) >= 89.9 and abs(theta) <= 90.1:
                        hx = height / 2.
                        hy=  width  / 2.
                    else:
                        hx = abs(width*cos(radians(theta))/2.)
                        hy = max(abs(width*sin(radians(theta))/2.),abs(height*sin(radians(theta))/2.,))

                    xauto = [ax.get_xticks()[0],ax.get_xticks()[-1]]
                    yauto = [ax.get_yticks()[0],ax.get_yticks()[-1]]
                    xlim((min([self.parametros.estimativa[p1] - 1.15*hx,xauto[0]]), \
                          max([self.parametros.estimativa[p1] + 1.15*hx,xauto[-1]])))
                    ylim((min([self.parametros.estimativa[p2] - 1.15*hy,yauto[0]]),\
                          max([self.parametros.estimativa[p2] + 1.15*hy,yauto[-1]])))
                    legend([ellipse,PSO],[u"Ellipse",u"Verossimilhança"])
                    fig.savefig(base_path+base_dir+'Regiao_verossimilhanca_'+str(self.parametros.simbolos[p1])+'_'+str(self.parametros.simbolos[p2])+'.png')
                    close()
                    p2+=1            
       
            base_dir = sep + 'Grandezas' + sep
            Validacao_Diretorio(base_path,base_dir)
            self.x.Graficos(base_path, ID = ['calculado'])
            self.y.Graficos(base_path, ID = ['calculado'])
            
            for iy in xrange(self.y.NV):
                for ix in xrange(self.x.NV):
                    x = self.x.calculado.matriz_estimativa[:,ix]
                    y = self.y.calculado.matriz_estimativa[:,iy]
                    ux = self.x.calculado.matriz_incerteza[:,ix]
                    #Falta a matriz incerteza do modelo, então está sendo usado a incerteza do pontos
                    #experimentais apenas para compilar
        #           uy = self.y.calculado.matriz_incerteza[:,i]
                    uy = self.y.experimental.matriz_incerteza[:,iy]                    
                    graficos_entrada_saida(x,y,ux,uy,ix,iy,base_dir,'calculado')
            
            base_dir = sep + 'Estimacao' + sep
            Validacao_Diretorio(base_path,base_dir)
            for iy in xrange(self.y.NV):
               # Gráfico comparativo entre valores experimentais e calculados pelo modelo, sem variância, uma reta é um indicativo de qualidade da estimação         
                    y  = self.y.experimental.matriz_estimativa[:,iy]
                    ym = self.y.calculado.matriz_estimativa[:,iy]
                    diagonal = linspace(min(y),max(y))  
                    fig = figure()
                    ax = fig.add_subplot(1,1,1)
                    plot(y,ym,'bo',linewidth=2.0)
                    plot(diagonal,diagonal,'k-',linewidth=2.0)
                    ax.yaxis.grid(color='gray', linestyle='dashed')                        
                    ax.xaxis.grid(color='gray', linestyle='dashed')
                    label_tick_x   = ax.get_xticks().tolist()                 
                    tamanho_tick_x = (label_tick_x[1] - label_tick_x[0])/2
                    ymin   = min(ym) - tamanho_tick_x
                    ymax   = max(ym) + tamanho_tick_x
                    xlim((ymin,ymax))
                    ylim((ymin,ymax))                
                    xlabel(self.y.labelGraficos('experimental')[iy])
                    ylabel(self.y.labelGraficos('calculado')[iy])
                    fig.savefig(base_path+base_dir+'grafico_'+str(self.y.simbolos[iy])+'_ye_ym_sem_var.png')
                    close()
                    
if __name__ == "__main__":
    from numpy import ones
    from Funcao_Objetivo import WLS
    from Modelo import Modelo
    # Exemplo validação: Exemplo resolvido 5.11, 5.12, 5.13 (capítulo 5) (Análise de Dados experimentais I)
#    Tempo
    x1 = transpose(array([120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,\
    45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,\
    60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,\
    60.1,60.0,30.0,30.0,60.0],ndmin=2))
    
    #Temperatura
    x2 = transpose(array([600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,\
    620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,\
    620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,\
    639.0,639.0,639.0,639.0,639.0,639.0,639.0],ndmin=2))
    
    x = concatenate((x1,x2),axis=1)

    ux = ones((41,2))
    
    y = transpose(array([0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,\
    0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,\
    0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,\
    0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449],ndmin=2))
        
    uy = ones((41,1))    
    
    Estime = Estimacao(WLS,Modelo,nomes_x = ['variavel teste x1','variavel teste 2'],simbolos_x=[r't','T'],label_latex_x=[r'$t$','$T$'],nomes_y=['y1'],simbolos_y=[r'y1'],nomes_param=['theyta'+str(i) for i in xrange(2)],simbolos_param=[r'theta%d'%i for i in xrange(2)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(2)])
    sup=[1,30000]
    inf=[0,20000]

#    # Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
#    x1 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
#    y1 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
#    x2 = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
#    y2 = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
#    
#    ux1 = ones((10,1))
#    ux2 = ones((10,1))
#    uy1 = ones((10,1))    
#    uy2 = ones((10,1))
#    
#    x  = concatenate((x1,x2),axis=1)
#    y  = concatenate((y1,y2),axis=1)    
#    ux = concatenate((ux1,ux2),axis=1)
#    uy = concatenate((uy1,uy2),axis=1)
#
#    Estime = Estimacao(WLS,Modelo,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=[r'theta%d'%i for i in xrange(4)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(4)])
#    sup = [6.  ,.3  ,8.  ,0.7]
#    inf = [1.  , 0  ,1.  ,0.]
#    
    # Continuacao
    Estime.gerarEntradas(x,y,ux,uy)    
    #print Estime.x.labelGraficos()
 #   Estime.otimiza(sup=[2,2,2,2],inf=[-2,-2,-2,-2],algoritmo='PSO',itmax=5)
 #   Estime.graficos(0.95)
#    saida = concatenate((Estime.x.experimental.matriz_estimativa[:,0:1],Estime.x.experimental.matriz_incerteza[:,0:1]),axis=1)
#    for i in xrange(1,Estime.NX):
#        aux = concatenate((Estime.x.experimental.matriz_estimativa[:,i:i+1],Estime.x.experimental.matriz_incerteza[:,i:i+1]),axis=1)
#        saida =  concatenate((saida,aux),axis=1)
#    grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário
    
    # Otimização
    Estime.otimiza(sup=sup,inf=inf,algoritmo='PSO',itmax=100,Num_particulas=30,metodo={'busca':'Otimo','algoritmo':'PSO','inercia':'TVIW-Adaptative-VI'})
    Estime.incertezaParametros(.95,1e-5,metodo='geral')
    grandeza = Estime._armazenarDicionario()
    Estime.analiseResiduos(0.95)
    lista_de_etapas = ['entrada','otimizacao','estimacao'] 
    Estime.graficos(lista_de_etapas,0.95)       
    grandeza2 = Estime._armazenarDicionario()