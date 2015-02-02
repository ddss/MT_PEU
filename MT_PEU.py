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

from scipy.stats import f, normaltest, anderson, shapiro, ttest_1samp, kstest,\
 norm, probplot, ttest_ind

from scipy.misc import factorial
from numpy.linalg import inv
from math import floor, log10

from matplotlib import use
use('Agg')

from matplotlib.pyplot import figure, axes, plot, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, show, text, hist, boxplot

from os import getcwd, sep

# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import matriz2vetor, vetor2matriz, Validacao_Diretorio 
from PSO import PSO
from Funcao_Objetivo import WLS
from Modelo import Modelo


class Organizador:
    
    def __init__(self,estimativa,incerteza,tipos={'estimativa':'matriz','incerteza':'incerteza'},NE=None):
        '''
        Classe para organizar os dados das estimativas e suas respectivas incertezas, disponibilizando-os na forma de matriz, vetores e listas.
        
        ========    
        Entradas
        ========
        
        * ``estimativa`` (array) : estimativas para as observações das variáveis (na forma de um vetor ou matriz). \
        Caso seja uma matriz, cada coluna contém as estimativas para uma variável. Se for um vetor, as estimativas estão \
        numa única coluna, sendo necessário fornecer a entrada NE.
        * ``incerteza``  (array) : incerteza (ou variância) para os valores das estimativas. Esta informação sempre será uma matriz. No entanto, \
        caso sejam as incertezas, cada coluna da contém a incerteza para os pontos de uma variável. Caso seja as variâncias, será a matriz de covariância.
        * ``tipos``      (dicionário): dicionário que define como as entradas estimativa e incerteza foram definidas:\
        As chaves e conteúdos são: estimativa (conteúdo: ``matriz`` ou ``vetor``) e incerteza (conteúdo: ``incerteza`` ou `variancia``).
        * ``NE`` (int): quantidade de pontos experimentais. Necessário apenas quanto a estimativa é um vetor.
        
        **AVISO:**
        
        * se estimativa for uma matriz, espera-se que ``ìncerteza`` seja uma matriz em que cada coluna seja as *INCERTEZAS* para cada observação de uma certa variável (ela será o atributo ``.matriz_incerteza`` ) (``tipo`` = {`estimativa`:`matriz`, `incerteza`:`incerteza`})
        * se estimativa for um vetor, espera-se que ``ìncerteza`` seja a matriz de *COVARIÂNCIA* (dimensão de estimativa x dimensçao de estimativa) (ela será o atributo ``.matriz_covariancia`` ) (``tipo`` = {`estimativa`:`vetor`, `incerteza`:`variancia`})
        * se a incerteza for definida na forma de uma matriz de incertezas, a matriz de covariância assumirá que os elementos fora da diagonal principal são ZEROS.
        
        =========
        Atributos
        =========
        
        * ``.matriz_estimativa`` (array): cada variável está alocada em uma coluna que contém suas observações.
        * ``.vetor_estimativa``  (array): todas as observações de todas as variáveis estão em um único vetor.
        * ``lista_estimativa``   (list):  lista de listas, em quw cada lista interna contém as estimativas para uma variável.   
        * ``.matriz_incerteza``  (array): matriz em que cada coluna contém a incerteza de cada ponto de uma certeza variável.
        * ``.matriz_covariancia`` (array): matriz de covariância.
        * ``lista_incerteza``(list): lista de listas, em que cada lista interna contém as incertezas para uma variável. 
        * ``lista_variancia``(list): ista de listas, onde cada lista interna contém as variâncias para uma variável. Aqui \
        é considerado somente a diagonal principal da matriz de covariancia.

        ** AVISO: **

        * Caso a incerteza seja definida como None, os atributos relacionados a ela NÃO serão criados.
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO da entrada tipos:
        # ---------------------------------------------------------------------
        # verificação se as chaves estão corretas:
        if ('estimativa' not in tipos.keys()) or ('incerteza' not in tipos.keys()):
            raise NameError(u'É necessário definir as chaves estimativa e incerteza da entrada tipos. Elas não foram\
            incluídas ou a grafia está errada')
        
        # conteúdos disponíveis
        tiposEstimativa = ('matriz','vetor') 
        tiposIncerteza  = ('incerteza','variancia')            
        
        # verificação se os conteúdos estão corretos
        if tipos['estimativa'] not in tiposEstimativa:
            raise NameError(('Os conteúdos disponíveis para a chave estimativa do dicionário tipos são: '\
            +'%s, '*(len(tiposEstimativa)-1)+'%s.')%tiposEstimativa)
        
        if tipos['incerteza'] not in tiposIncerteza:
            raise NameError(('Os conteúdos disponíveis para a chave incerteza do dicionário tipos são: '\
            +'%s, '*(len(tiposIncerteza)-1)+'%s.')%tiposIncerteza)
        
        # ---------------------------------------------------------------------
        # CRIAÇÃO dos atributos na forma de ARRAYS
        # ---------------------------------------------------------------------
        if tipos['estimativa'] == 'matriz':
            
            self.matriz_estimativa  = estimativa
            self.vetor_estimativa   = matriz2vetor(self.matriz_estimativa) # conversão de matriz para vetor
        
        if tipos['estimativa'] == 'vetor':
            
            self.vetor_estimativa   = estimativa
            self.matriz_estimativa  = vetor2matriz(self.vetor_estimativa,NE) # Conversão de vetor para uma matriz  
        
        if tipos['incerteza'] == 'incerteza':

            if incerteza != None:            
                self.matriz_incerteza   = incerteza            
                self.matriz_covariancia = diag(transpose(matriz2vetor(self.matriz_incerteza**2)).tolist()[0])

        if tipos['incerteza'] == 'variancia':

            if incerteza != None:        
                self.matriz_covariancia = incerteza      
                self.matriz_incerteza   = vetor2matriz(transpose(array([diag(self.matriz_covariancia**0.5)])),NE)
        # ---------------------------------------------------------------------
        # Criação dos atributos na forma de LISTAS
        # ---------------------------------------------------------------------
        self.lista_estimativa = self.matriz_estimativa.transpose().tolist()
        
        if incerteza != None:        
            self.lista_incerteza  = self.matriz_incerteza.transpose().tolist()
            self.lista_variancia  = diag(self.matriz_covariancia).tolist()

class Grandeza:
    
    def __init__(self,nomes,simbolos,unidades,label_latex):
        '''
        Classe para organizar as características das Grandezas:
        
        * experimentais
        * do modelo
        * parâmetros
        * resíduos
        
        =======
        Entrada
        =======
        
        * nomes (list)      : deve ser uma lisra contendo o nome das variáveis
        * simbolos (list)   : deve ser uma lista contendo os símbolos, na ordem de entrada de cada variável
        * label_latex(list) : de ser uma lista contendo os símbolos em formato LATEX
        
        =======
        Métodos        
        =======
        
        * _experimental: irá criar o atributo experimental. Deve ser usado se se tratar de dados experimentais
        * _modelo      : irá criar o atributo experimental. Deve ser usado se se tratar de dados do modelo
        * _parametro   : irá criar os atributos estimativa, matriz_covariancia, regiao_abrangencia. Deve ser usado para os parâmetros
        * _residuo_x   : irá criar os atributos x. Deve ser usado para os resíduos de x
        * _residuo_y   : irá criar os atributos y. Deve ser usado para os resíduos de y
    
        =========
        Atributos
        =========
        
        * ``.nomes`` (list): lista com os nomes das variáveis
        * ``.simbolos`` (list): lista com os símbolos das variáveis (inclusive em código Latex)
        * ``.experimental`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _experimental**
        * ``.modelo`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _modelo**
        * ``.estimativa`` (list): lista com estimativas. **só exitirá após execução do método _parametro**
        * ``.matriz_covariancia`` (array): array representando a matriz covariância. **só exitirá após execução do método _parametro**
        * ``.regiao_abrangencia`` (list): lista representando os pontos pertencentes à região de abrangência. **só exitirá após execução do método _parametro**
        * ``.x`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _residuo_x**
        * ``.y`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _residuo_y**
        '''

        self.nomes       = nomes
        if nomes == None:
            self.nomes = [None]*len(nomes)

        self.simbolos    = simbolos        
        if simbolos == None:
            self.simbolos = [None]*len(nomes)

        self.unidades    = unidades
        if unidades == None:
            self.unidades = [None]*len(nomes)
        
        self.label_latex = label_latex
        if label_latex == None:
            self.label_latex = [None]*len(nomes)

    
    def _experimental(self,estimativa,variancia,tipo):
        
        self.__ID         = 'experimental'        
        self.experimental = Organizador(estimativa,variancia,tipo)        
        
    def _validacao(self,estimativa,variancia,tipo):
        
        self.__ID      = 'validacao'
        self.validacao = Organizador(estimativa,variancia,tipo)


    def _modelo(self,estimativa,variancia,tipo,NE):
        
        self.__ID   = 'modelo'
        self.modelo = Organizador(estimativa,variancia,tipo,NE)     
    
 
    def _parametro(self,estimativa,variancia,regiao):
        
        self.__ID               = 'parametro'                
        self.estimativa         = estimativa
        self.matriz_covariancia = variancia
        self.regiao_abrangencia = regiao

    def _residuo(self,estimativa,variancia,tipo):
        
        self.__ID         = 'residuo'
        self.estimativa = Organizador(estimativa,variancia,tipo)           

    def _testesEstatisticos(self):
        '''
        Subrotina para realizar testes estatísticos nos resíduos
        
        =================
        Testes realizados
        =================
        NORMALIDADE:
        
        * normaltest: Retorna o pvalor do teste de normalidade. Hipótese nula: a amostra vem de distribuição normal
        * shapiro   : Retorna o valor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal
        * anderson  : Retorna o valor do Teste para os dados provenientes de uma distribuição em particular. Hipotese nula: a amostra vem de uma normal
        * probplot  : Gera um gráfico de probabilidade de dados de exemplo contra os quantis de uma distribuição teórica especificado (a distribuição normal por padrão).
                      Calcula uma linha de melhor ajuste para os dados se "encaixar" é verdadeiro e traça os resultados usando Matplotlib.
                      tornar um conjunto de dados positivos transformados por uma transformação Box-Cox power.
        MÉDIA:
        * ttest_1sam: Retorna o pvalor para média determinada. Hipótese nula: a amostra tem a média determinada
      
        SAÍDA (sobe a forma de ATRIBUTO)
        * estatisticas (float):  Valor das hipóteses testadas. Para a hipótese nula tida como verdadeira, um valor abaixo de 0.05 nos diz que para 95% de confiança pode-se rejeitar essa hipótese
        '''
    

        if self.__ID == 'residuo':
            
            pvalor = {}
            for nome in self.simbolos:
                pvalor[nome] = {}
                

            for i,nome in enumerate(self.simbolos):
                dados = self.estimativa.matriz_estimativa[:,i]
                
                # Testes para normalidade
                # Lista que contém as chamadas das funções de teste:
                if size(dados) < 20: # Se for menor do que 20 não será realizado no normaltest, pois ele só é válido a partir dste número de dados
                    pnormal=[None, shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['Normalidade'] = {'normaltest':None, 'shapiro':pnormal[1][1], 'anderson':[[pnormal[2][0]], pnormal[2][1][1]],'kstest':pnormal[3][1]}
                else:
                    pnormal=[normaltest(dados), shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['Normalidade'] = {'normaltest':pnormal[0][1], 'shapiro':pnormal[1][1], 'anderson':[[pnormal[2][0]], pnormal[2][1][1]],'kstest':pnormal[3][1]}

                # Dicionário para salvar os resultados                
                # Testes para a média:
                pmedia = [ttest_1samp(dados,0.), ttest_ind(dados,norm.rvs(loc=0.,scale=std(dados,ddof=1),size=size(dados)))]
                pvalor[nome]['Media'] = {'ttest':pmedia[0][1],'ttest_ind':pmedia[1][1]}
                
        else:
            raise NameError(u'Os testes estatísticos são válidos apenas para o resíduos')

        self.estatisticas = pvalor
            
    def Graficos(self,base_path=None):
        
        if base_path == None:
            base_path = getcwd()

        base_dir  = sep + 'Grandezas' + sep
        Validacao_Diretorio(base_path,base_dir)

        if self.__ID == 'residuo':

            # BOXPLOT
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            boxplot(self.estimativa.matriz_estimativa)
            ax.set_xticklabels(self.simbolos)
            fig.savefig(base_path+'Boxplot_Residuos')

            for i,nome in enumerate(self.simbolos):
                # Gráficos da estimação
                base_dir = base_dir + sep + self.simbolos[i] + sep
                Validacao_Diretorio(base_path,base_dir)

                dados = self.estimativa.matriz_estimativa[:,i]
        
                # TENDENCIA
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                plot(linspace(1,size(dados),num=size(dados)),dados, 'o')
                xlabel('Ordem de Coleta')
                ylabel(self.simbolos[i])
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
                xlim((0,size(dados)))
                ax.axhline(0, color='black', lw=2)
                fig.savefig(base_path+base_dir+'Ordem_'+self.simbolos[i])
                close()        
        
                # AUTO CORRELAÇÃO
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                ax.acorr(dados,usevlines=True, normed=True,maxlags=None)
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
                ax.axhline(0, color='black', lw=2)
                xlim((0,size(dados)))
                fig.savefig(base_path+base_dir+'autocorrelacao_'+self.simbolos[i])
                close()

                # HISTOGRAMA                
                fig = figure()
                hist(dados, normed=True)
                xlabel(self.simbolos[i])
                ylabel(u'Frequência')
                fig.savefig(base_path+base_dir+'histograma_'+self.simbolos[i])
                close()

                # NORMALIDADE               
                res = probplot(dados, dist='norm', sparams=(mean(dados),std(dados,ddof=1)))
                fig = figure()
                plot(res[0][0], res[0][1], 'o', res[0][0], res[1][0]*res[0][0] + res[1][1])
                xlabel('Quantis')
                ylabel('Valores ordenados')
                xmin = amin(res[0][0])
                xmax = amax(res[0][0])
                ymin = amin(dados)
                ymax = amax(dados)
                posx = xmin + 0.70 * (xmax - xmin)
                posy = ymin + 0.01 * (ymax - ymin)
                text(posx, posy, "$R^2$=%1.4f" % res[1][2])
                fig.savefig(base_path+base_dir+'probplot_'+self.simbolos[i])
                

class Estimacao:
    
    def __init__(self,FO,Modelo,Nomes_y,Nomes_x,Nomes_param,projeto=None,**kwargs):
        '''
        Classe para Executar a estimação de parâmetros        
        
        =======================
        Entradas (obrigatórias)
        =======================
        * ``FO``          : função objetivo (Thread)
        * ``Modelo``      : modelo (Thread). O modelo deve retornar um array com número de colunas igual ao número de y.
        * ``Nomes_y``     : lista coms os nomes das variáveis y
        * ``Nomes_x``     : lista com os nomes das variáveis x
        * ``Nomes_param`` : lista com o nomes dos parâmetros
        * ``projeto``     : nome do projeto
            
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
        * ``label_latex_param`` (list): lista com os símbolos d['residuo_y%d'%(i,) for i in xrange(self.NY)],['ry_%d'%(i,) for i in xrange(self.NY)],\
                         self.y.unidades,[r'$res_y_%d$'%(i,) for i in xrange(self.NY)as variáveis em formato LATEX
        '''
        self.__validacaoArgumentosEntrada(kwargs,'init')
        self.__validacaoSimbologiaUnidade(kwargs)
        
        # Inicialização das variáveis
        self.x          = Grandeza(Nomes_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        self.y          = Grandeza(Nomes_y    ,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        #self.xval       = Grandeza(Nomes_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]),kwargs.get(self.__keywordsEntrada[2]))
        #self.yval       = Grandeza(Nomes_y    ,kwargs.get(self.__keywordsEntrada[3]),kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        self.parametros = Grandeza(Nomes_param,kwargs.get(self.__keywordsEntrada[6]),kwargs.get(self.__keywordsEntrada[7]),kwargs.get(self.__keywordsEntrada[8]))
        
        # Número de variáveis
        self.NX  = size(self.x.nomes) # Número de variáveis independentes
        self.NY  = size(self.y.nomes) # Número de variáveis dependentes
        self.NP  = size(self.parametros.nomes) # Número de parâmetros
    
        # Criaçaão das variáveis internas
        self.__FO        = FO
        self.__modelo    = Modelo
        self.__base_path = getcwd()+'/'+str(projeto)+'/'
        
        # Controle interno das etapas do algoritmo (métodos executados)
        self.__etapasdisponiveis = ['__init__','gerarEntradas','otimizacao','incertezaParametros','regiaoAbrangencia','analiseResiduos'] # Lista de etapas que o algoritmo irá executar
        self.__etapas     = [self.__etapasdisponiveis[0]] # Variável de armazenamento das etapas realizadas pelo algoritmo
        
    def __validacaoSimbologiaUnidade(self,keywargs):
        '''
        Método para validação da simbologia e das unidades, se entradas pelo usuário
        
        * Verificação se a simbologia de todas as varíaveis x, y, parametros foi definida, caso incluídas
        * Verificação se a unidade de todas as varíaveis x, y, parametros foi definida, caso incluídas
        '''
        
        if size(keywargs.get(self.__keywordsEntrada[0])) != size(keywargs.get(self.__keywordsEntrada[1])) and keywargs.get(self.__keywordsEntrada[1]) != None:
            raise ValueError(u'Os nomes para as variáveis x e sua simbologia devem ter o mesmo tamanho')            
            
    
    def __validacaoArgumentosEntrada(self,keywargs,etapa,args=None):
        '''
        Validação para verificar se todos os argumentos das rotinas entrada foram definidos corretamente.
        
        * Se houve keyword erradas        
        '''
        # Keywords disponíveis        
        self.__keywordsEntrada = ['simbolos_x','unidades_x','label_latex_x','simbolos_y','unidades_y','label_latex_y','simbolos_param','unidades_param','label_latex_param'] # Keywords disponíveis para a entrada
        self.__keywordsOtimizacao = {'PSO':['itmax','Num_particulas','sup','inf','posinit_sup', 'posinit_inf','w' ,'C1' ,'C2', 'Vmax','Vreinit' , 'otimo' , 'deltaw', 'k', 'gama'],'Nelder_Mead':[]}        
        self.__keywordsOtimizacaoObrigatorias = {'PSO':['sup','inf'],'Nelder_Mead':[]}  

        if etapa =='init':
            # Validação se houve keywords digitadas incorretamente:
            keyincorreta  = [key for key in keywargs.keys() if not key in self.__keywordsEntrada]
        
            if len(keyincorreta) != 0:
                raise NameError(('keyword(s) incorretas: '+(len(keyincorreta)-1)*'%s, '+u'%s. Verifique documentação para keywords disponíveis.')%tuple(keyincorreta))
    
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
        self.__validacaoDadosEntrada(xe,ux,self.NX,self.NE) 
        self.__validacaoDadosEntrada(ye,uy,self.NY,self.NE)
        self.__validacaoDadosEntrada(xval,uxval,self.NX,self.NE) 
        self.__validacaoDadosEntrada(yval,uyval,self.NY,self.NE)

        
        # Salvando os dados experimentais nas variáveis.
        self.x._experimental(xe,ux,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._experimental(ye,uy,{'estimativa':'matriz','incerteza':'incerteza'}) 
        
        # Salvando os dados de validação.
        self.x._validacao(xval,uxval,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._validacao(yval,uyval,{'estimativa':'matriz','incerteza':'incerteza'}) 
    
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
            grandeza[simbolo] = Grandeza(self.y.nomes[j],simbolo,self.y.unidades[j],self.y.label_latex[j])
            if self.__etapasdisponiveis[1] in self.__etapas:
                grandeza[simbolo]._experimental(self.y.experimental.matriz_estimativa[:,j:j+1],self.y.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:
                grandeza[simbolo]._modelo(self.y.modelo.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)

        for j,simbolo in enumerate(self.x.simbolos):
            grandeza[simbolo] = Grandeza(self.x.nomes[j],simbolo,self.x.unidades[j],self.x.label_latex[j])
            if self.__etapasdisponiveis[1] in self.__etapas:            
                grandeza[simbolo]._experimental(self.x.experimental.matriz_estimativa[:,j:j+1],self.x.experimental.matriz_incerteza[:,j:j+1],{'estimativa':'matriz','incerteza':'incerteza'})
            if self.__etapasdisponiveis[2] in self.__etapas:            
                grandeza[simbolo]._modelo(self.x.modelo.matriz_estimativa[:,j:j+1],None,{'estimativa':'matriz','incerteza':'variancia'},None)

        for j,simbolo in enumerate(self.parametros.simbolos):
            grandeza[simbolo] = Grandeza(self.parametros.nomes[j],simbolo,self.parametros.unidades[j],self.parametros.label_latex[j])
            if self.__etapasdisponiveis[2] in self.__etapas:
                if self.parametros.matriz_covariancia == None:
                    grandeza[simbolo]._parametro(self.parametros.estimativa[j],None,None)
                else:
                    grandeza[simbolo]._parametro(self.parametros.estimativa[j],self.parametros.matriz_covariancia[j,j],None)

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
            self.parametros._parametro(self.Otimizacao.gbest,None,None) # Atribuindo o valor ótimo dos parâemetros


        # O modelo é calculado para os dados de validação
        aux = self.__modelo(self.parametros.estimativa,self.x.validacao.matriz_estimativa,args)
        aux.start()
        aux.join()
                
        # Salvando os resultados
        self.y._modelo(aux.result,None,{'estimativa':'matriz','incerteza':'variancia'},None)
        self.x._modelo(self.x.experimental.matriz_estimativa,self.x.experimental.matriz_incerteza,{'estimativa':'matriz','incerteza':'incerteza'},None)

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
            
        self.parametros._parametro(self.parametros.estimativa,matriz_covariancia,self.parametros.regiao_abrangencia)
        
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
            
        matriz_hessiana=[[1. for col in range(self.NP)] for row in range(self.NP)]
        
        FO_otimo = self.Otimizacao.best_fitness # Valor da função objetivo no ponto ótimo

        for i in range(self.NP): 
            for j in range(self.NP):
                
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
        
        Fisher = f.ppf(PA,self.NP,(self.NE*self.NY-self.NP))            
        Comparacao = self.Otimizacao.best_fitness*(1+float(self.NP)/(self.NE*self.NY-float(self.NP))*Fisher)
        
        Regiao = []; Hist_Posicoes = []; Hist_Fitness = []
        for it in xrange(self.Otimizacao.itmax):
            for ID_particula in xrange(self.Otimizacao.Num_particulas):
                if self.Otimizacao.historico_fitness[it][ID_particula] <= Comparacao:
                    Regiao.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Posicoes.append(self.Otimizacao.historico_posicoes[it][ID_particula])
                Hist_Fitness.append(self.Otimizacao.historico_fitness[it][ID_particula])
            
        self.parametros._parametro(self.parametros.estimativa,self.parametros.matriz_covariancia,Regiao)
        
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
        self.residuos_x = Grandeza(nomes=['residuo_'+self.x.nomes[i] for i in xrange(self.NX)],simbolos=['res_'+self.x.simbolos[i] for i in xrange(self.NX)],\
                         unidades = self.x.unidades,label_latex = [r'$res_x_%d$'%(i,) for i in xrange(self.NX)])
        self.residuos_y = Grandeza(nomes=['residuo_'+self.y.nomes[i] for i in xrange(self.NY)],simbolos=['res_'+self.y.simbolos[i] for i in xrange(self.NY)],\
                         unidades = self.y.unidades,label_latex = [r'$res_y_%d$'%(i,) for i in xrange(self.NY)])
        
        # Calculos dos residuos (ou desvios) - estão baseados nos dados de validação
        residuo_y = self.y.validacao.matriz_estimativa - self.y.modelo.matriz_estimativa
        residuo_x = self.x.validacao.matriz_estimativa - self.x.modelo.matriz_estimativa
        
        # Attribuição dos valores nos objetos
        self.residuos_x._residuo(residuo_x,None,{'estimativa':'matriz','incerteza':'incerteza'})
        self.residuos_y._residuo(residuo_y,None,{'estimativa':'matriz','incerteza':'incerteza'})
        
        # DESCOMENTAR QUANDO RECONCILIAÇÃO !!
        # self.residuos_x._testesEstatisticos()
        # self.residuos_x.Graficos(self.__base_path)
        self.residuos_y.Graficos(self.__base_path + sep + 'Graficos'  + sep)
        self.residuos_y._testesEstatisticos()
        
        # Gráficos que dependem de informações da estimação (y)
        # TO DO: RELOCAR PARA A SESSÃO DE GRÁFICOS

        base_path  = self.__base_path + sep + 'Graficos'  + sep
  
        for i,simb in enumerate(self.y.simbolos):         
            base_dir =  sep + 'Grandezas' + sep + self.residuos_y.simbolos[i] + sep
            # Gráficos da otimização
            Validacao_Diretorio(base_path,base_dir)  
        
            ymodelo = self.y.experimental.matriz_estimativa[:,i]
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            plot(ymodelo,self.residuos_y.estimativa.matriz_estimativa[:,i], 'o')
            xlabel(u'Valores Ajustados '+self.y.simbolos[i])
            ylabel(u'Resíduos '+self.y.simbolos[i])
            ax.yaxis.grid(color='gray', linestyle='dashed')                        
            ax.xaxis.grid(color='gray', linestyle='dashed')
            ax.axhline(0, color='black', lw=2)
            fig.savefig(base_path+base_dir+'grafico_residuo_'+str(self.y.nomes[i])+'_versus_ycalculado.png')
            close()

        self.__etapas.append(self.__etapasdisponiveis[5]) # Inclusão desta etapa na lista de etapas
        
    def graficos(self,PA):
        
        base_path  = self.__base_path + '/Graficos/'
        # Gráficos da otimização
        base_dir = '/PSO/'
        Validacao_Diretorio(base_path,base_dir)

        self.Otimizacao.Graficos(base_path+base_dir,Nome_param=self.parametros.simbolos,Unid_param=self.parametros.unidades,FO2a2=True)
        
        # Gráficos da estimação
        base_dir = '/Estimacao/'
        Validacao_Diretorio(base_path,base_dir)
        
        
        # Gráfico comparativo entre valores experimentais e calculados pelo modelo, sem variância
        for i in xrange(self.NY):
            y  = self.y.experimental.matriz_estimativa[:,i]
            ym = self.y.modelo.matriz_estimativa[:,i]
            
            ymin = min(y)            
            ymax = max(y)            
            
            diag = linspace(min(y),max(y))  
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            plot(y,ym,'bo',linewidth=2.0)
            plot(diag,diag,'k-',linewidth=2.0)
            ax.yaxis.grid(color='gray', linestyle='dashed')                        
            ax.xaxis.grid(color='gray', linestyle='dashed')
            xlim((ymin,ymax))
            ylim((ymin,ymax))
            
            xlabel(self.y.nomes[i]+' experimental')
            ylabel(self.y.nomes[i]+' calculado')
            fig.savefig(base_path+base_dir+'grafico_'+str(self.y.nomes[i])+'_ye_ym_sem_var.png')
            close()

            
        # Região de abrangência (verossimilhança)
        
        Hist_Posicoes , Hist_Fitness = self.regiaoAbrangencia(PA)
       
        if self.NP == 1:
                
            aux = []
            for it in xrange(size(self.parametros.regiao_abrangencia)/self.NP):     
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
            xlabel(self.parametros.nomes[0],fontsize=20)
            fig.savefig(base_path+base_dir+'Regiao_verossimilhanca_'+str(self.parametros.nomes[0])+'_'+str(self.parametros.nomes[0])+'.png')
            close()
        
        else:
            
            Combinacoes = int(factorial(self.NP)/(factorial(self.NP-2)*factorial(2)))
            p1 = 0; p2 = 1; cont = 0; passo = 1
            
            for pos in xrange(Combinacoes):
                if pos == (self.NP-1)+cont:
                    p1 +=1; p2 = p1+1; passo +=1
                    cont += self.NP-passo
                
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                
                for it in xrange(size(self.parametros.regiao_abrangencia)/self.NP):     
                    PSO, = plot(self.parametros.regiao_abrangencia[it][p1],self.parametros.regiao_abrangencia[it][p2],'bo',linewidth=2.0,zorder=1)
            
                plot(self.parametros.estimativa[p1],self.parametros.estimativa[p2],'r*',markersize=10.0,zorder=2)
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
             
                xlabel(self.parametros.nomes[p1],fontsize=20)
                ylabel(self.parametros.nomes[p2],fontsize=20)

                fig.savefig(base_path+base_dir+'Regiao_verossimilhanca_'+str(self.parametros.nomes[p1])+'_'+str(self.parametros.nomes[p2])+'.png')
                close()
                p2+=1


        
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
    
    Estime = Estimacao(WLS,Modelo,Nomes_x = ['variavel teste x1','variavel teste 2'],simbolos_x=[r't','T'],label_latex_x=[r'$t$','$T$'],Nomes_y=['y1'],simbolos_y=[r'y1'],Nomes_param=['theyta'+str(i) for i in xrange(2)],simbolos_param=[r'theta%d'%i for i in xrange(2)],label_latex_param=[r'$\theta_{%d}$'%i for i in xrange(2)])
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
    grandeza = Estime._armazenarDicionario() # ETAPA PARA CRIAÇÃO DOS DICIONÁRIOS - Grandeza é uma variável que retorna as grandezas na forma de dicionário
    
    # Otimização
    Estime.otimiza(sup=sup,inf=inf,algoritmo='PSO',itmax=300,Num_particulas=30)
    grandeza = Estime._armazenarDicionario()
    #Estime.incertezaParametros(delta=1e-6) 
    Estime.analiseResiduos()
    Estime.graficos(.95)

        
