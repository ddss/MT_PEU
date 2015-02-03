# -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author(es): Daniel, Francisco, Anderson, Leomar, Victor, Leonardo
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# Importação de pacotes de terceiros
from numpy import array, transpose, concatenate,size, diag, linspace, min, max, \
sort, argsort, mean,  std, amin, amax, copy

from scipy.stats import f

from scipy.misc import factorial
from numpy.linalg import inv
from math import floor, log10

from matplotlib import use
use('Agg')

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot

from os import getcwd, sep

# Subrotinas próprias e adaptações (desenvolvidas pelo GI-UFBA)
from Grandeza import Grandeza
from subrotinas import Validacao_Diretorio, plot_cov_ellipse
from PSO import PSO
from Funcao_Objetivo import WLS
from Modelo import Modelo


class Estimacao:
    
    def __init__(self,FO,Modelo,simbolos_y,simbolos_x,simbolos_param,projeto='Projeto',**kwargs):
        '''
        Classe para Executar a estimação de parâmetros        
        
        =======================
        Entradas (obrigatórias)
        =======================
        * ``FO``             : função objetivo (Thread)
        * ``Modelo``         : modelo (Thread). O modelo deve retornar um array com número de colunas igual ao número de y.
        * ``simbolos_y``     : lista com os simbolos das variáveis y (Não podem haver caracteres especiais)
        * ``simbolos_x``     : lista com os simbolos das variáveis x (Não podem haver caracteres especiais)
        * ``simbolos_param`` : lista com o simbolos dos parâmetros (Não podem haver caracteres especiais)
        * ``projeto``        : nome do projeto
            
        Exemplo de função objetivo: ::
                
            from threading import Thread
            
            class FO(Thread):
                result = 0
                def __init__(self,param,args):
                    Thread.__init__(self)
                    self.x = param
    
                def run(self):
                    
                    self.result =  self.x**2            
        
        =========
        Keywords:
        =========
        * ``simbolos_x``     (list): lista com os símbolos para x 
        * ``unidades_x``     (list): lista com as unidades para x (inclusive em formato LATEX)
        * ``label_latex_x``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``simbolos_y``     (list): lista com os símbolos para y
        * ``unidades_y``     (list): lista com as unidades para y (inclusive em formato LATEX)
        * ``label_latex_y``  (list): lista com os símbolos das variáveis em formato LATEX
        
        * ``simbolos_param`` (list): lista com os símbolos para os parâmetros (inclusive em formato LATEX)
        * ``unidades_param`` (list): lista com as unidades para os parâmetros (inclusive em formato LATEX)
        * ``label_latex_param`` (list): lista com os símbolos d['residuo_y%d'%(i,) for i in xrange(self.y.NV)],['ry_%d'%(i,) for i in xrange(self.y.NV)],\
                         self.y.unidades,[r'$res_y_%d$'%(i,) for i in xrange(self.y.NV)as variáveis em formato LATEX
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÕES GERAIS DE KEYWORDS
        # ---------------------------------------------------------------------
        self.__validacaoArgumentosEntrada(kwargs,'init')
        
        # ---------------------------------------------------------------------
        # VALIDAÇÃO ESPECÍFICA:
        # ---------------------------------------------------------------------
        # SÍMBOLOS
        # Verificação se os símbolos possuem caracteres especiais
        for simb in [simbolos_y,simbolos_x,simbolos_param]:
            for elemento in simb:
                if set('[~!@#$%^&*()_+{}":;\']+$').intersection(elemento):
                    raise NameError('Os nomes das grandezas não podem ter caracteres especiais. Simbolo incorreto: '+elemento)       
        
        # PROJETO
        # Verificação se o nome do projeto possui caracteres especiais
        if set('[~!@#$%^&*()_+{}":;\']+$').intersection(projeto):
            raise NameError('O nome do projeto não pode conter caracteres especiais')  
        
        # Inicialização das variáveis
        # Variável      = Grandeza(simbolos  , nomes                               , unidades                            , label_latex)
        self.x          = Grandeza(simbolos_x,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        self.y          = Grandeza(simbolos_y,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        self.parametros = Grandeza(simbolos_param,kwargs.get(self.__keywordsEntrada[6]),kwargs.get(self.__keywordsEntrada[7]),kwargs.get(self.__keywordsEntrada[8]))
            
        # Criação das variáveis internas
        self.__FO        = FO
        self.__modelo    = Modelo
        self.__base_path = getcwd()+ sep +str(projeto)+sep
        
        # Controle interno das etapas do algoritmo (métodos executados)
        self.__etapasdisponiveis = ['__init__','gerarEntradas','otimizacao','incertezaParametros','regiaoAbrangencia','analiseResiduos'] # Lista de etapas que o algoritmo irá executar
        self.__etapas            = [self.__etapasdisponiveis[0]] # Variável de armazenamento das etapas realizadas pelo algoritmo
            
             
    def __validacaoArgumentosEntrada(self,keywargs,etapa,args=None):
        '''
        Validação para verificar se todos os argumentos das rotinas entrada foram definidos corretamente.
        
        * Se houve keyword erradas        
        '''
        # Keywords disponíveis        
        self.__keywordsEntrada = ['nomes_x','unidades_x','label_latex_x','nomes_y','unidades_y','label_latex_y','nomes_param','unidades_param','label_latex_param'] # Keywords disponíveis para a entrada
        self.__keywordsOtimizacao = {'PSO':['itmax','Num_particulas','sup','inf','posinit_sup', 'posinit_inf','w' ,'C1' ,'C2', 'Vmax','Vreinit' , 'otimo' , 'deltaw', 'k', 'gama'],'Nelder_Mead':[]}        
        self.__keywordsOtimizacaoObrigatorias = {'PSO':['sup','inf'],'Nelder_Mead':[]}  

        if etapa =='init':
            # Validação se houve keywords digitadas incorretamente:
            keyincorreta  = [key for key in keywargs.keys() if not key in self.__keywordsEntrada]
        
            if len(keyincorreta) != 0:
                raise NameError('keyword(s) incorretas: '+', '.join(keyincorreta)+'.'+' Keywords disponíveis: '+', '.join(self.__keywordsEntrada)+'.')
    
        if etapa == 'otimizacao':
                    
            if (not args in self.__keywordsOtimizacao.keys()) and  args != None:
                raise NameError((u'A opção de algoritmo não está correta. Escolha entre as opções: '+(len(self.__keywordsOtimizacao.keys())-1)*' %s,'+' %s')%tuple(self.__keywordsOtimizacao.keys()))
            
            if  args == 'PSO':
                keyincorreta  = [key for key in keywargs.keys() if not key in self.__keywordsOtimizacao['PSO']]
    
            if len(keyincorreta) != 0:
                raise NameError(('keyword(s) incorretas: '+(len(keyincorreta)-1)*'%s, '+u'%s. Verifique documentação para keywords disponíveis.')%tuple(keyincorreta))
    
            # Validação das keywords obrigatórias por método
            keyobrigatoria = [key for key in self.__keywordsOtimizacaoObrigatorias[args] if not key in keywargs.keys()]
                
            if len(keyobrigatoria) != 0:
                aux = [args]
                aux.extend(keyobrigatoria)
                raise NameError((u'Para o método de %s a(s) keyword(s) obrigatória(s) não foram (foi) definida(s): '+(len(keyobrigatoria)-1)*'%s, '+u'%s.')%tuple(aux))
    
    
    def __validacaoDadosEntrada(self,dados,udados,Ndados,NE):
        '''
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
 
                
    def __defaults(self,kwargs,algoritmo):
        '''
        Definição dos valores dos paramêmtros para os métodos, inclusive os valores default
        '''
        
        if algoritmo == 'PSO':
            
            itmax          = 500 if kwargs.get('itmax')          == None else kwargs.get('itmax')
            Num_particulas = 30  if kwargs.get('Num_particulas') == None else kwargs.get('Num_particulas')
        
            return (itmax, Num_particulas)
   
    def gerarEntradas(self,xe,ye,ux,uy,xval=None,yval=None,uxval=None,uyval=None,uxy=None):
        '''
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
        # DEFAULT
        # Caso não definidos dados de validação, será assumido os valores experimentais
        if xval == None:
            xval = xe
        
        if yval == None:
            yval = ye
        
        if uxval == None:
            uxval = ux
            
        if uyval == None:
            uyval = uy        
        
        # QUANTIDADE DE PONTOS EXPERIMENTAIS
        self.NE  = size(xe,0) # Número de observações

        # VALIDAÇÃO dos dados de entrada x, y, xval e yval
        self.__validacaoDadosEntrada(xe,ux,self.x.NV,self.NE) 
        self.__validacaoDadosEntrada(ye,uy,self.y.NV,self.NE)
        self.__validacaoDadosEntrada(xval,uxval,self.x.NV,self.NE) 
        self.__validacaoDadosEntrada(yval,uyval,self.y.NV,self.NE)

        
        # Salvando os dados experimentais nas variáveis.
        self.x._SETexperimental(xe,ux,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETexperimental(ye,uy,{'estimativa':'matriz','incerteza':'incerteza'}) 
        
        # Salvando os dados de validação.
        self.x._SETvalidacao(xval,uxval,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETvalidacao(yval,uyval,{'estimativa':'matriz','incerteza':'incerteza'}) 
     
        self.__etapas.append(self.__etapasdisponiveis[1]) # Inclusão desta etapa da lista de etapas
        
    def _armazenarDicionario(self):
        '''
        Método opcional para armazenar as Grandezas (x,y e parãmetros) na
        forma de um dicionário, cujas chaves são os símbolos.
        
        ======
        Saídas
        ======
        
        * grandeza: dicionário cujas chaves são os símbolos das grandezas e respectivos
        conteúdos objetos da classe Grandezas.
        '''
        grandeza = {}        
        for j,simbolo in enumerate(self.y.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.y.nomes[j]],[self.y.unidades[j]],[self.y.label_latex[j]])
            if self.__etapasdisponiveis[1] in self.__etapas:
                grandeza[simbolo]._SETexperimental(self.y.experimental.matriz_estimativa[:,j:j+1],self.y.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:
                grandeza[simbolo]._SETcalculado(self.y.calculado.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)

        for j,simbolo in enumerate(self.x.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.x.nomes[j]],[self.x.unidades[j]],[self.x.label_latex[j]])
            if self.__etapasdisponiveis[1] in self.__etapas:            
                grandeza[simbolo]._SETexperimental(self.x.experimental.matriz_estimativa[:,j:j+1],self.x.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:            
                grandeza[simbolo]._SETcalculado(self.x.calculado.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)

        for j,simbolo in enumerate(self.parametros.simbolos):
            grandeza[simbolo] = Grandeza([simbolo],[self.parametros.nomes[j]],[self.parametros.unidades[j]],[self.parametros.label_latex[j]])
            if self.__etapasdisponiveis[2] in self.__etapas:
                if self.parametros.matriz_covariancia == None:
                    grandeza[simbolo]._SETparametro(self.parametros.estimativa[j],None,None)
                else:
                    grandeza[simbolo]._SETparametro(self.parametros.estimativa[j],self.parametros.matriz_covariancia[j,j],None)

        return grandeza
    

    def otimiza(self,algoritmo='PSO',args=None,**kwargs):
        '''
        Método para realização da otimização        
        
        =======================
        Entradas (Obrigatórias)
        =======================

        * algoritmo : string informando o algoritmo de otimização a ser utilizado. Cada algoritmo tem suas próprias keywords
        * args      : argumentos extras a serem passados para o modelo
        
        =======================================
        Keywords (alguns dependem do algoritmo)
        =======================================
        
        algoritmo = PSO
        
        * sup           : limite superior de busca
        * inf           : limite inferior de busca
        * Num_particulas: número de particulas
        * itmax         : número máximo de iterações
        '''
        
        # Validação dos keywords do método de otimização
        self.__validacaoArgumentosEntrada(kwargs,'otimizacao',algoritmo)       

    
        self.__args_model = [self.y.experimental.vetor_estimativa,self.x.experimental.matriz_estimativa,\
        self.y.experimental.matriz_covariancia,self.x.experimental.matriz_covariancia,\
        args,self.x.simbolos,self.y.simbolos,self.parametros.simbolos] # Argumentos extras a serem passados para a função objetivo
        
        if algoritmo == 'PSO': # Obtençao das valores e definiçaão dos valores default

            itmax, Num_particulas = self.__defaults(kwargs,algoritmo)

            # Executar a otimização
            self.Otimizacao = PSO(kwargs.get('sup'),kwargs.get('inf'),Num_particulas=Num_particulas,itmax=itmax,args_model=self.__args_model)
            self.Otimizacao.Busca(self.__FO,printit=False)
            self.parametros._SETparametro(self.Otimizacao.gbest,None,None) # Atribuindo o valor ótimo dos parâemetros


        # O modelo é calculado para os dados de validação
        aux = self.__modelo(self.parametros.estimativa,self.x.validacao.matriz_estimativa,[args,self.x.simbolos,self.y.simbolos,self.parametros.simbolos])
        aux.start()
        aux.join()
                
        # Salvando os resultados
        self.y._SETcalculado(aux.result,None,{'estimativa':'matriz','incerteza':'variancia'},None)
        self.x._SETcalculado(self.x.experimental.matriz_estimativa,self.x.experimental.matriz_incerteza,{'estimativa':'matriz','incerteza':'incerteza'},None)

        self.__etapas.append(self.__etapasdisponiveis[2]) # Inclusão desta etapa da lista de etapas
 
        self.incertezaParametros(.95,1e-6)
        
    def incertezaParametros(self,PA=0.95,delta=1e-5):       
        '''
        Método para avaliação da matriz covariãncia dos parâmetros.
        
        =======================
        Entradas (Obrigatórias)
        =======================
        * PA         : probabilidade de abrangência para gerar a região de abrangência
        * delta      : incremento para o cálculo das derivadas (derivada numérica)

        ======
        Saídas
        ======
        * a matriz de covariância é salva na Grandeza parâmetros
        '''
        
        # Except temporário
        try:
            matriz_covariancia = 2*inv(self.__Hessiana_FO_Param(delta))
        except:
            matriz_covariancia = None
            
        self.parametros._SETparametro(self.parametros.estimativa,matriz_covariancia,self.parametros.regiao_abrangencia)
        
        self.regiaoAbrangencia(PA) # método para avaliação da região de abrangência
                
        self.__etapas.append(self.__etapasdisponiveis[3]) # Inclusão desta etapa da lista de etapas
        
    def __Hessiana_FO_Param(self,delta=1e-5):
        '''
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
 
        def Vetor_delta(posicao,delta):
            '''
            Subrotina para somar "delta" a uma posicao do vetor.
            '''
            vetor = copy(self.parametros.estimativa)

            if isinstance(posicao,list):
                vetor[posicao[0]] = vetor[posicao[0]]+delta[0]
                vetor[posicao[1]] = vetor[posicao[1]]+delta[1]
            else:
                vetor[posicao] = vetor[posicao]+delta
                
            return vetor
            
        matriz_hessiana=[[1. for col in range(self.parametros.NV)] for row in range(self.parametros.NV)]
        
        FO_otimo = self.Otimizacao.best_fitness # Valor da função objetivo no ponto ótimo

        for i in range(self.parametros.NV): 
            for j in range(self.parametros.NV):
                
                delta1 = (10**(floor(log10(abs(self.parametros.estimativa[i])))))*delta
                delta2 = (10**(floor(log10(abs(self.parametros.estimativa[j])))))*delta
                
                if i==j:
                    # Incrementos
                    vetor_parametro_delta_positivo = Vetor_delta(i,delta1) # Vetor que irá conter o incremento no parâmetro i
                    vetor_parametro_delta_negativo = Vetor_delta(j,-delta2)  # Vetor que irá conter o incremento no parâmetro j

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
                    
                    vetor_parametro_delta_ipositivo_jpositivo = Vetor_delta([i,j],[delta1,delta2])
                    
                    FO_ipositivo_jpositivo=self.__FO(vetor_parametro_delta_ipositivo_jpositivo,self.__args_model)
                    FO_ipositivo_jpositivo.start()
                    
                    vetor_parametro_delta_inegativo_jpositivo=Vetor_delta([i,j],[-delta1,delta2])
 
                    FO_inegativo_jpositivo=self.__FO(vetor_parametro_delta_inegativo_jpositivo,self.__args_model)
                    FO_inegativo_jpositivo.start()

                    vetor_parametro_delta_ipositivo_jnegativo=Vetor_delta([i,j],[delta1,-delta2])
   
                    FO_ipositivo_jnegativo=self.__FO(vetor_parametro_delta_ipositivo_jnegativo,self.__args_model)
                    FO_ipositivo_jnegativo.start()

                    vetor_parametro_delta_inegativo_jnegativo=Vetor_delta([i,j],[-delta1,-delta2])

                    
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
        

    def regiaoAbrangencia(self,PA=0.95):
        '''
        Método para avaliação da região de abrangência
        '''
        
        Fisher = f.ppf(PA,self.parametros.NV,(self.NE*self.y.NV-self.parametros.NV))            
        Comparacao = self.Otimizacao.best_fitness*(1+float(self.parametros.NV)/(self.NE*self.y.NV-float(self.parametros.NV))*Fisher)
        
        Regiao = []; Hist_Posicoes = []; Hist_Fitness = []
        for it in xrange(self.Otimizacao.itmax):
            for ID_particula in xrange(self.Otimizacao.Num_particulas):
                if self.Otimizacao.historico_fitness[it][ID_particula] <= Comparacao:
                    Regiao.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Posicoes.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Fitness.append(self.Otimizacao.historico_fitness[it][ID_particula])
            
        self.parametros._SETparametro(self.parametros.estimativa,self.parametros.matriz_covariancia,Regiao)
        
        self.__etapas.append(self.__etapasdisponiveis[4]) # Inclusão desta etapa da lista de etapas

        return (Hist_Posicoes, Hist_Fitness)
        
    def analiseResiduos(self):
        '''
        Método para realização da análise de resíduos.
        
        ======
        Saídas
        ======
        
        * Saídas na forma de gráficos
        * As grandezas resíduos possuem o atributo "estatisticas".
        '''
        # Criação dos resíduos como Grandezas
        
        # Calculos dos residuos (ou desvios) - estão baseados nos dados de validação
        residuo_y = self.y.validacao.matriz_estimativa - self.y.calculado.matriz_estimativa
        residuo_x = self.x.validacao.matriz_estimativa - self.x.calculado.matriz_estimativa
        
        # Attribuição dos valores nos objetos
        self.x._SETresiduos(residuo_x,None,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._SETresiduos(residuo_y,None,{'estimativa':'matriz','incerteza':'incerteza'})
        
        # DESCOMENTAR QUANDO RECONCILIAÇÃO !!
        #self.x.Graficos(self.__base_path + sep + 'Graficos'  + sep,ID=['residuo'])
        #self.x._testesEstatisticos()
        self.y.Graficos(self.__base_path + sep + 'Graficos'  + sep,ID=['residuo'])
        self.y._testesEstatisticos()
        
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

    def graficos(self,lista_de_etapas,PA):
        '''
        Métodos para gerar e salvar os gráficos
        =======================
        Entradas (obrigatórias)
        =======================
        * ``Etapa``   : Status que determinam se o método deve gerar os gráficos de entrada ou de otimização
        * ``PA``      : 
        ==========
        Atributos
        ==========
                
        '''
        self.__etapas.append(self.__etapasdisponiveis[5]) # Inclusão desta etapa da lista de etapas
           
#        base_path = os.sep + ' Graficos '+ os.sep
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
            
            for i in xrange(self.y.NV):
                # Região de abrangência (método da verossimilhança)
                Hist_Posicoes , Hist_Fitness = self.regiaoAbrangencia(PA)
               
                if self.parametros.NV == 1:
                        
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
                    ylabel(r"$\quad \Phi $",fontsize = 20)
                    xlabel(self.parametros.labelGraficos()[0],fontsize=20)
                    fig.savefig(base_path+base_dir+'regiao_verossimilhanca_'+str(self.parametros.nomes[0])+'_'+str(self.parametros.nomes[0])+'.png')
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
                        
                        Fisher = f.ppf(PA,self.parametros.NV,(self.NE*self.y.NV-self.parametros.NV))            
                        Comparacao = self.Otimizacao.best_fitness*(float(self.parametros.NV)/(self.NE*self.y.NV-float(self.parametros.NV))*Fisher)
                        cov = array([[self.parametros.matriz_covariancia[p1,p1],self.parametros.matriz_covariancia[p1,p2]],[self.parametros.matriz_covariancia[p2,p1],self.parametros.matriz_covariancia[p2,p2]]])
                        ellipse = plot_cov_ellipse(cov, [self.parametros.estimativa[p1],self.parametros.estimativa[p2]], Comparacao, fill = False, color = 'r', linewidth=2.0,zorder=2)
                        plot(self.parametros.estimativa[p1],self.parametros.estimativa[p2],'r*',markersize=10.0,zorder=2)
                        ax.yaxis.grid(color='gray', linestyle='dashed')                        
                        ax.xaxis.grid(color='gray', linestyle='dashed')
                        xlabel(self.parametros.labelGraficos()[p1],fontsize=20)
                        ylabel(self.parametros.labelGraficos()[p2],fontsize=20)
                        legend([ellipse,PSO],[u"Ellipse",u"Verossimilhança"])
                        fig.savefig(base_path+base_dir+'Regiao_verossimilhanca_'+str(self.parametros.nomes[p1])+'_'+str(self.parametros.nomes[p2])+'.png')
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
                for ix in xrange(self.x.NV):
               # Gráfico comparativo entre valores experimentais e calculados pelo modelo, sem variância         
                    y  = self.y.experimental.matriz_estimativa[:,i]
                    ym = self.y.calculado.matriz_estimativa[:,i]
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
                    xlabel(self.y.nomes[i]+' experimental')
                    ylabel(self.y.nomes[i]+' calculado')
                    fig.savefig(base_path+base_dir+'grafico_'+str(self.y.nomes[i])+'_ye_ym_sem_var.png')
                    close()
                    
if __name__ == "__main__":
    from numpy import ones

    # Exemplo validação: Exemplo resolvido 5.11, 5.12, 5.13 (capítulo 5) (Análise de Dados experimentais I)
    #Tempo
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

    # Exemplo de validacao Exemplo resolvido 5.2 (capitulo 6) (Análise de dados experimentais 1)
#    x = transpose(array([1.,2.,3.,5.,10,15.,20.,30.,40.,50.],ndmin=2))
#    y = transpose(array([1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51],ndmin=2))
#    ux = ones((10,1))
#    uy = ones((10,1))    
    
#    Estime = Estimacao(WLS,Modelo,Nomes_x = ['variavel teste x1'],simbolos_x=[r'x'],label_latex_x=[r'$x$'],Nomes_y=['y1'],simbolos_y=[r'y1'],Nomes_param=['theyta'+str(i) for i in xrange(2)],simbolos_param=[r'theta%d'%i for i in xrange(2)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(2)])
#    sup = [10,10]
#    inf = [-10,-10]
    
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
    Estime.otimiza(sup=sup,inf=inf,algoritmo='PSO',itmax=300,Num_particulas=30)
    grandeza = Estime._armazenarDicionario()
    Estime.analiseResiduos()
    lista_de_etapas = ['entrada','otimizacao','estimacao']
    Estime.graficos(lista_de_etapas,0.95)       
