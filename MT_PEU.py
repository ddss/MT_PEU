# -*- coding: utf-8 -*-
"""
Principais classes do motor de cálculo do PEU

@author: Daniel
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

# Importação de pacotes de terceiros
from numpy import array, transpose, concatenate,size, diag, linspace, min, max, \
sort, argsort
from scipy.stats import f
from scipy.misc import factorial

from matplotlib.pyplot import figure, axes, plot, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close

from os import getcwd

# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import matriz2vetor, vetor2matriz, Validacao_Diretorio 
from PSO_rev26 import PSO
from Funcao_Objetivo import WLS
from Modelo import Modelo


class Organizador:
    
    def __init__(self,estimativa,incerteza,tipos={'estimativa':'matriz','incerteza':'incerteza'},NE=None):
        '''
        Classe para organizar as estimativas e suas incertezas, convertendo  de matrizes para vetores quando necessário
        
        ========    
        Entradas
        ========
        
        * ``estimativas`` (array) : estimativas para as observações das variáveis (na forma de um vetor ou matriz)
        * ``incerteza``   (array) : incerteza para os valores das variáveis ou a matriz de covariância
        * ``tipo``       (dicionário) : definindo os métodos, cujas chaves são estimativa (conteúdo - matriz ou vetor) e incerteza (conteúdo  - incerteza ou variancia)
        
        **AVISO:**
        
        * se estimativa for uma matriz (``tipo`` = matriz), espera-se que ``ìncerteza`` seja uma matriz em que cada coluna seja as *INCERTEZAS* para cada observação de uma certa variável (ela será o atributo ``.matriz_incerteza`` )
        * se estimativa for um vetor (``tipo`` = vetor), espera-se que ``ìncerteza`` seja a matriz de *COVARIÂNCIA* (dimensão de estimativa x dimensçao de estimativa) (ela será o atributo ``.matriz_covariancia`` )
        
        =========
        Atributos
        =========
        
        * ``.matriz_estimativa`` (array): cada variável está alocada em uma coluna que contém suas observações
        * ``.vetor_estimativa``  (array): todas as observações de todas as variáveis estão em um único vetor
        * ``.matriz_incerteza``  (array): uma matriz em que cada coluna contém a incerteza das observações de uma certeza variável
        * ``.matriz_covariancia`` (array): matriz de covariância
        '''
        
        if tipos['estimativa'] == 'matriz':
            
            self.matriz_estimativa  = estimativa
            self.vetor_estimativa   = matriz2vetor(self.matriz_estimativa)
        
        if tipos['estimativa'] == 'vetor':
            
            self.vetor_estimativa   = estimativa
            self.matriz_estimativa  = vetor2matriz(self.vetor_estimativa,NE)       
        
        if tipos['incerteza'] == 'incerteza':

            if incerteza != None:            
                self.matriz_incerteza   = incerteza            
                self.matriz_covariancia = diag(transpose(matriz2vetor(self.matriz_incerteza**2)).tolist()[0])

        if tipos['incerteza'] == 'variancia':

            if incerteza != None:        
                self.matriz_covariancia = incerteza      
                self.matriz_incerteza   = vetor2matriz(transpose(array([diag(self.matriz_covariancia**0.5)])),NE)

class Grandeza:
    
    def __init__(self,nomes,simbolos,unidades):
        '''
        Classe para organizar as características das Grandezas:
        
        * experimentais
        * do modelo
        * parâmetros
        * resíduos
        
        =======
        Entrada
        =======
        
        * simbolos (list): deve ser uma lista contendo os símbolos, na ordem de entrada de cada variável
        
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
        self.nomes    = nomes
        self.simbolos = simbolos
        self.unidades = unidades
    
    def _experimental(self,estimativa,variancia,tipo):
        
        self.experimental = Organizador(estimativa,variancia,tipo)        
        
    def _modelo(self,estimativa,variancia,tipo,NE):

        self.modelo = Organizador(estimativa,variancia,tipo,NE)     

    def _parametro(self,estimativa,variancia,regiao):
        
        self.estimativa         = estimativa
        self.matriz_covariancia = variancia
        self.regiao_abrangencia = regiao
        
    def _residuo_x(self,estimativa,variancia,tipo,NE):
        
        self.x = Organizador(estimativa,variancia,tipo,NE)   
    
    def _residuo_y(self,estimativa,variancia,tipo,NE):
        
        self.y = Organizador(estimativa,variancia,tipo,NE)   

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
        * ``simbolos_x``     (list): lista com os símbolos para x (inclusive em formato LATEX)    
        * ``unidades_x``     (list): lista com as unidades para x (inclusive em formato LATEX)
        * ``simbolos_y``     (list): lista com os símbolos para y (inclusive em formato LATEX)
        * ``unidades_y``     (list): lista com as unidades para y (inclusive em formato LATEX)
        * ``simbolos_param`` (list): lista com os símbolos para os parâmetros (inclusive em formato LATEX)
        * ``unidades_param`` (list): lista com as unidades para os parâmetros (inclusive em formato LATEX)
       
        '''
        self.__validacaoArgumentosEntrada(kwargs,'init')
        self.__validacaoSimbologiaUnidade(kwargs)
        
        # Inicialização das variáveis
        self.x          = Grandeza(Nomes_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]))
        self.y          = Grandeza(Nomes_y    ,kwargs.get(self.__keywordsEntrada[2]),kwargs.get(self.__keywordsEntrada[3]))
        self.xval       = Grandeza(Nomes_x    ,kwargs.get(self.__keywordsEntrada[0]),kwargs.get(self.__keywordsEntrada[1]))
        self.yval       = Grandeza(Nomes_y    ,kwargs.get(self.__keywordsEntrada[2]),kwargs.get(self.__keywordsEntrada[3]))
        self.parametros = Grandeza(Nomes_param,kwargs.get(self.__keywordsEntrada[4]),kwargs.get(self.__keywordsEntrada[5]))
        
        # Número de variáveis
        self.NX  = size(self.x.nomes) # Número de variáveis independentes
        self.NY  = size(self.y.nomes) # Número de variáveis dependentes
        self.NP  = size(self.parametros.nomes) # Número de parâmetros
        
        # Criaçaão das variáveis internas
        self.__FO        = FO
        self.__modelo    = Modelo
        self.__base_path = getcwd()+'/'+str(projeto)+'/'
        
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
        self.__keywordsEntrada = ['simbolos_x','unidades_x','simbolos_y','unidades_y','simbolos_param','unidades_param'] # Keywords disponíveis para a entrada
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
                raise NameError((u'Para o método de %s a(s) keyword(s) obrigatória(s) não foram (foi) definida(s): '+(len(keyincorreta)-1)*'%s, '+u'%s.')%tuple(aux))
    
    def __validacaoDadosEntrada(self,dados,udados,Ndados):
        '''
        Validação dos dados de entrada 
        
        * verificar se as colunas dos arrays de entrada são iguais aos nomes das variáveis definidas (y, x)
        '''
        
        if size(dados,1) != Ndados: 
            raise ValueError(u'O número de variáveis definidas foi %s, mas foram inseridos dados apenas para %s variáveis.'%(Ndados,size(dados,1)))
       
        if Ndados != size(udados,1):
           raise ValueError(u'O número de variáveis definidas foi %s, mas foram inseridos dados apenas para %s incertezas.'%(Ndados,size(udados,1)))
          
    def __defaults(self,kwargs,algoritmo):
        '''
        Definição dos valores dos paramêmtros para os métodos, inclusive os valores default
        '''
        
        if algoritmo == 'PSO':
            
            itmax          = 500 if kwargs.get('itmax')          == None else kwargs.get('itmax')
            Num_particulas = 30  if kwargs.get('Num_particulas') == None else kwargs.get('Num_particulas')
        
            return (itmax, Num_particulas)
        
         
    def otimiza(self,xe,ye,ux,uy,uxy=None,args=None,algoritmo='PSO',**kwargs):
        '''
        Método para realização da otimização        
        
        =======================
        Entradas (Obrigatórias)
        =======================
        
        * xe        : array com as variáveis independentes na forma de colunas
        * ux        : array com as incertezas das variáveis independentes na forma de colunas
        * ye        : array com as variáveis dependentes na forma de colunas
        * uy        : array com as incertezas das variáveis dependentes na forma de colunas
        * uxy       : covariância entre x e y
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

        # Validação dos dados de entrada x e y
        self.__validacaoDadosEntrada(xe,ux,self.NX) 
        self.__validacaoDadosEntrada(ye,uy,self.NY)
        
        # Salvando os dados nas variáveis 
        self.x._experimental(xe,ux,{'estimativa':'matriz','incerteza':'incerteza'})
        self.y._experimental(ye,uy,{'estimativa':'matriz','incerteza':'incerteza'}) 
 
        self.NE  = size(self.x.experimental.matriz_estimativa,0) # Número de observações
    
        args_model = [self.y.experimental.vetor_estimativa,self.x.experimental.matriz_estimativa,\
        self.y.experimental.matriz_covariancia,self.x.experimental.matriz_covariancia,\
        self.x.simbolos,self.y.simbolos,args] # Argumentos extras a serem passados para a função objetivo
        
        if algoritmo == 'PSO': # Obtençao das valores e definiçaão dos valores default

            itmax, Num_particulas = self.__defaults(kwargs,algoritmo)

            # Executar a otimização
            self.Otimizacao = PSO(kwargs.get('sup'),kwargs.get('inf'),Num_particulas=Num_particulas,itmax=itmax,args_model=args_model)
            self.Otimizacao.Busca(self.__FO)
            self.parametros._parametro(self.Otimizacao.gbest,None,None) # Atribuindo o valor ótimo dos parâemetros

        aux = self.__modelo(self.parametros.estimativa,self.x.experimental.matriz_estimativa,args)
        aux.start()
        aux.join()
        
        # Salvando os resultados
        self.y._modelo(aux.result,None,{'estimativa':'matriz','incerteza':'variancia'},None)
        self.x._modelo(self.x.experimental.matriz_estimativa,self.x.experimental.matriz_incerteza,{'estimativa':'matriz','incerteza':'incerteza'},None)

        self.incertezaParametros(self.__FO,self.__modelo)
        
    def incertezaParametros(self,FO,Modelo,PA=0.95):
        '''
        Método para avaliação da matriz covariãncia dos parâmetros.
        
        '''
        matriz_covariancia = diag(len(self.parametros.estimativa)*[1])
        
        self.parametros._parametro(self.parametros.estimativa,matriz_covariancia,self.parametros.regiao_abrangencia)
        
        self.regiaoAbrangencia(PA) # método para avaliação da região de abrangência
        
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
        
        return (Hist_Posicoes, Hist_Fitness)
        
    def analiseResiduos(self):
        '''
        Método para realização da análise de resíduos
        '''
        self.residuos = Grandeza()
        
        residuo_y = self.y.experimental.matriz_estimativa - self.y.modelo.matriz_estimativa
        residuo_x = self.x.experimental.matriz_estimativa - self.x.modelo.matriz_estimativa
        
        self.residuos._residuo_x(residuo_x,None,{'estimativa':'matriz','incerteza':'incerteza'},self.NE)
        self.residuos._residuo_y(residuo_y,None,{'estimativa':'matriz','incerteza':'incerteza'},self.NE)


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
            
            if self.y.simbolos == None and self.y.unidades == None:
                xlabel(r'$y_{%d}$'%(i+1)+' experimental')
                ylabel(r'$y_{%d}$'%(i+1)+' calculado')
            elif self.y.simbolos != None and self.y.unidades == None:
                xlabel(self.y.simbolos[i]+' experimental')
                ylabel(self.y.simbolos[i]+' calculado')   
            elif self.y.simbolos != None and self.y.unidades != None:
                xlabel(self.y.simbolos[i]+'/'+self.y.unidades[i]+' experimental')
                ylabel(self.y.simbolos[i]+'/'+self.y.unidades[i]+' calculado')   
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
            if self.parametros.simbolos == None and self.parametros.unidades == None:
                    xlabel(r'$\theta_{%d}$'%(i+1),fontsize=20)
            elif self.parametros.simbolos != None and self.parametros.unidades == None:
                    xlabel(self.parametros.simbolos[0],fontsize=20)   
            elif self.parametros.simbolos != None and self.parametros.unidades != None:
                    xlabel(self.parametros.simbolos[0]+'/'+self.parametros.unidades[0],fontsize=20)  
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
             
                if self.parametros.simbolos == None and self.parametros.unidades == None:
                        xlabel(r'$\theta_{%d}$'%(p1+1),fontsize=20)
                        ylabel(r'$\theta_{%d}$'%(p2+1),fontsize=20)
                elif self.parametros.simbolos != None and self.parametros.unidades == None:
                        xlabel(self.parametros.simbolos[p1],fontsize=20)   
                        ylabel(self.parametros.simbolos[p2],fontsize=20)  
                elif self.parametros.simbolos != None and self.parametros.unidades != None:
                        xlabel(self.parametros.simbolos[p1]+'/'+self.parametros.unidades[p1],fontsize=20)   
                        ylabel(self.parametros.simbolos[p2]+'/'+self.parametros.unidades[p2],fontsize=20)                 
                fig.savefig(base_path+base_dir+'Regiao_verossimilhanca_'+str(self.parametros.nomes[p1])+'_'+str(self.parametros.nomes[p2])+'.png')
                close()
                p2+=1


        
if __name__ == "__main__":
    
    x1 = transpose(array([1,2,3,4,5,6,7,8,9,10],ndmin=2))
    x2 = transpose(array([1,2,3,4,5,6,7,8,9,10],ndmin=2))
    
    ux1 = transpose(array([1,1,1,1,1,1,1,1,1,1],ndmin=2))
    ux2 = transpose(array([1,1,1,1,1,1,1,1,1,1],ndmin=2))
    
    y1 = transpose(array([2,3,4,5,6,7,8,9,10,11],ndmin=2))
    y2 = transpose(array([2,4,6,8,10,12,14,16,18,20],ndmin=2))

    uy1 = transpose(array([1,1,1,1,1,1,1,1,1,1],ndmin=2))
    uy2 = transpose(array([1,1,1,1,1,1,1,1,1,1],ndmin=2))
    
    x  = concatenate((x1,x2),axis=1)
    y  = concatenate((y1,y2),axis=1)
    ux = concatenate((ux1,ux2),axis=1)
    uy = concatenate((uy1,uy2),axis=1)


    Estime = Estimacao(WLS,Modelo,Nomes_x = ['x1','x2'],simbolos_x=[r'$x_1$',r'$x_2$'],Nomes_y=['y1','y2'],simbolos_y=[r'$y_1$',r'$y_2$'],Nomes_param=['theyta'+str(i) for i in xrange(4)],simbolos_param=[r'$\theta_{%d}$'%i for i in xrange(4)])
    Estime.otimiza(x,y,ux,uy,sup=[2,2,2,2],inf=[-2,-2,-2,-2],algoritmo='PSO',itmax=5)
    #Estime.graficos(0.95)
    saida = concatenate((Estime.x.experimental.matriz_estimativa[:,0:1],Estime.x.experimental.matriz_incerteza[:,0:1]),axis=1)
    for i in xrange(1,Estime.NX):
        aux = concatenate((Estime.x.experimental.matriz_estimativa[:,i:i+1],Estime.x.experimental.matriz_incerteza[:,i:i+1]),axis=1)
        saida =  concatenate((saida,aux),axis=1)
        
    print saida
    #Estime.analiseResiduos()