# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:05:02 2015

@author: danielsantana
"""
# Importação de pacotes de terceiros
from numpy import array, transpose ,size, diag, linspace, min, max, \
 mean,  std, amin, amax

from scipy.stats import normaltest, anderson, shapiro, ttest_1samp, kstest,\
 norm, probplot, ttest_ind

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot

from os import getcwd, sep


# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import matriz2vetor, vetor2matriz, Validacao_Diretorio 

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
        * ``NE`` (float): número de observações (para cada grandeza)
        * ``NV`` (float): número de variáveis

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

        # ---------------------------------------------------------------------
        # Número de pontos experimentais e de variáveis
        # ---------------------------------------------------------------------
        self.NE = size(self.matriz_estimativa,0)
        
class Grandeza:
    
    def __init__(self,simbolos,nomes=None,unidades=None,label_latex=None):
        '''
        Classe para organizar as características das Grandezas:
        
        * experimentais
        * do modelo
        * validação
        * parâmetros
        * resíduos
        
        =======
        Entrada
        =======
        OBRIGATÓRIO:
        * simbolos (list)   : deve ser uma lista contendo os símbolos, na ordem de entrada de cada variável   
        
        OPCIONAL:
        * nomes (list)      : deve ser uma lisra contendo o nome das variáveis
        * unidades (list)   : deve ser uma lista contendo as unidades das variáveis
        * label_latex(list) : de ser uma lista contendo os símbolos em formato LATEX
        
        =======
        Métodos        
        =======
        
        * _SETexperimental: irá criar o atributo experimental. Deve ser usado se se tratar de dados experimentais
        * _SETmodelo      : irá criar o atributo modelo. Deve ser usado se se tratar de dados do modelo
        * _SETvalidacao   : irá criar o atributo validacao. Deve ser usado se se tratar de dados de validação
        * _SETparametro   : irá criar os atributos estimativa, matriz_covariancia, regiao_abrangencia. Deve ser usado para os parâmetros
        * _SETresiduos    : irá criar o atributo resíduos. Deve ser usado para os resíduos de x

        =========
        Atributos
        =========
        
        * ``.nomes`` (list): lista com os nomes das variáveis
        * ``.simbolos`` (list): lista com os símbolos das variáveis (inclusive em código Latex)
        * ``.experimental`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _experimental**
        * ``.calculado`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _calculado**
        * ``.estimativa`` (list): lista com estimativas. **só exitirá após execução do método _parametro**
        * ``.matriz_covariancia`` (array): array representando a matriz covariância. **só exitirá após execução do método _parametro**
        * ``.regiao_abrangencia`` (list): lista representando os pontos pertencentes à região de abrangência. **só exitirá após execução do método _parametro**
        * ``.x`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _residuo_x**
        * ``.y`` (objeto): objeto Organizador (vide documentação do mesmo). **só exitirá após execução do método _residuo_y**
        '''
        if simbolos == None:
            raise NameError('Os símbolos das grandezas são obrigatórios')

        self.simbolos    = simbolos        

        self.nomes       = nomes
        if nomes == None:
            self.nomes = [None]*len(simbolos)

        self.unidades    = unidades
        if unidades == None:
            self.unidades = [None]*len(simbolos)
        
        self.label_latex = label_latex
        if label_latex == None:
            self.label_latex = [None]*len(simbolos)
        
        self.__ID = []
        self.__ID_disponivel = ['experimental','validacao','calculado','parametro','residuo']

        # ------------------------------------------------------------------------------------
        # VALIDAÇÂO referente ao tamanho das listas de simbolos, unidades, nomes e label_latex
        # -------------------------------------------------------------------------------------
        self.__validacaoEntrada()
    
        # ---------------------------------------------------------------------
        # Número de pontos experimentais e de variáveis
        # ---------------------------------------------------------------------   
        self.NV = len(simbolos)
        
    def __validacaoEntrada(self):
        '''
        Verificação:
        - se os atributos de simbologia, nome, unidades e label_latex são Listas.
        - se os tamanhos dos atributos de simbologia, nome, unidades e label_latex são os mesmos.
        '''
        # Verificação se nomes, unidade e label_latex são listas
        for elemento in [self.nomes,self.unidades,self.label_latex]:
            if not isinstance(elemento,list):
                raise TypeError(u'A simbologia, nomes, unidades e label_latex de uma grandeza devem ser LISTAS.')        
       
       # Verificação se nomes, unidade e label_latex possuem mesmo tamanho
        for elemento in [self.nomes,self.unidades,self.label_latex]:
            if len(elemento) != len(self.simbolos):
                raise ValueError(u'A simbologia, nomes, unidades e label_latex de uma grandeza devem ser listas de MESMO tamanho.')        
        
        # Verificação se os símbolos possuem caracteres especiais
        for simb in self.simbolos:
            if set('[~!@#$%^&*()_+{}":;\']+$').intersection(simb):
                raise NameError('Os nomes das grandezas não podem ter caracteres especiais. Simbolo incorreto: '+simb)       
    
    def _SETexperimental(self,estimativa,variancia,tipo):
        
        self.__ID.append('experimental')
        self.experimental = Organizador(estimativa,variancia,tipo)        
        
    def _SETvalidacao(self,estimativa,variancia,tipo):
        
        self.__ID.append('validacao')
        self.validacao = Organizador(estimativa,variancia,tipo)

    def _SETcalculado(self,estimativa,variancia,tipo,NE):
        
        self.__ID.append('calculado')
        self.calculado = Organizador(estimativa,variancia,tipo,NE)     
 
    def _SETresiduos(self,estimativa,variancia,tipo):
        
        self.__ID.append('residuo')
        self.residuos = Organizador(estimativa,variancia,tipo)  

    def _SETparametro(self,estimativa,variancia,regiao):
        
        self.__ID.append('parametro')           
        self.estimativa         = estimativa
        self.matriz_covariancia = variancia
        self.regiao_abrangencia = regiao

        
    def labelGraficos(self,add=None):
        '''
        Método para definição do label dos gráficos relacionado às grandezas.
        
        =======
        Entrada
        =======
        * add: texto que se deseja escrever antes da unidade. Deve ser um string
        '''
        # VALIDAÇÃO Da variável add
        if (add != None) and (not isinstance(add,str)):
            raise TypeError(u'A variável add deve ser um string')
            
        # Definição dos labels: latex ou nomes ou simbolos (nesta ordem)
        label = [None]*len(self.nomes)
        
        for z in xrange(len(self.nomes)):

            if self.label_latex[z] != None:
                label[z] = self.label_latex[z]
            elif self.nomes[z] != None:
                label[z] = self.nomes[z]
            elif self.simbolos[z] != None:
                label[z] = self.simbolos[z]

            if add != None:
                label[z] = label[z] +' '+ add
            
            # Caso seja definido uma unidade, esta será incluída no label
            if self.unidades[z] != None:
                label[z] = label[z]+' '+"/"+self.unidades[z]
            
        return label

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
    
        if 'residuo' in self.__ID: # Testes para os resíduos
            
            pvalor = {}
            for nome in self.simbolos:
                pvalor[nome] = {}
                

            for i,nome in enumerate(self.simbolos):
                dados = self.residuos.matriz_estimativa[:,i]
                
                # Testes para normalidade
                # Lista que contém as chamadas das funções de teste:
                if size(dados) < 20: # Se for menor do que 20 não será realizado no normaltest, pois ele só é válido a partir dste número de dados
                    pnormal=[None, shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':None, 'shapiro':pnormal[1][1], 'anderson':[[pnormal[2][0]], pnormal[2][1][1]],'kstest':pnormal[3][1]}
                else:
                    pnormal=[normaltest(dados), shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':pnormal[0][1], 'shapiro':pnormal[1][1], 'anderson':[[pnormal[2][0]], pnormal[2][1][1]],'kstest':pnormal[3][1]}

                # Dicionário para salvar os resultados                
                # Testes para a média:
                pmedia = [ttest_1samp(dados,0.), ttest_ind(dados,norm.rvs(loc=0.,scale=std(dados,ddof=1),size=size(dados)))]
                pvalor[nome]['residuo-Media'] = {'ttest':pmedia[0][1],'ttest_ind':pmedia[1][1]}
                
        else:
            raise NameError(u'Os testes estatísticos são válidos apenas para o resíduos')

        self.estatisticas = pvalor
            
    def Graficos(self,base_path=None,ID=None):
        '''
        Método para gerar os gráficos das grandezas, cujas informações só dependam dela.
        
        =======
        Entrada
        =======
        
        * base_path = caminho onde os gráficos deverão ser salvos
        * ID        = Identificação da grandeza. Este ID é útil apenas para as grandezas \
        dependentes e independentes, ele identifica para qual atributo os gráficos devem ser avaliados. \
        Caso seja None, será feito os gráficos para TODOS os atributos disponíveis.
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO DOS IDs:
        # ---------------------------------------------------------------------
        if ID == None:
            ID = self.__ID
        
        if False in [ele in self.__ID_disponivel for ele in ID]:
            raise NameError(u'Foi inserido uma ID indiponível. IDs disponíveis: '+','.join(self.__ID_disponivel))

        if base_path == None:
            base_path = getcwd()

        base_dir  = sep + 'Grandezas' + sep
        Validacao_Diretorio(base_path,base_dir)

        if 'residuo' in ID:
            # BOXPLOT
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            boxplot(self.residuos.matriz_estimativa)
            ax.set_xticklabels(self.simbolos)
            fig.savefig(base_path+base_dir+'residuos_boxplot'+'_'.join(self.simbolos))
            close()
            
            base_path = base_path + base_dir
            for i,nome in enumerate(self.simbolos):
                # Gráficos da estimação
                base_dir = sep + self.simbolos[i] + sep
                Validacao_Diretorio(base_path,base_dir)

                dados = self.residuos.matriz_estimativa[:,i]
        
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
                fig.savefig(base_path+base_dir+'residuos_ordem')
                close()        
        
                # AUTO CORRELAÇÃO
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                ax.acorr(dados,usevlines=True, normed=True,maxlags=None)
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
                ax.axhline(0, color='black', lw=2)
                xlim((0,size(dados)))
                fig.savefig(base_path+base_dir+'residuos_autocorrelacao')
                close()

                # HISTOGRAMA                
                fig = figure()
                hist(dados, normed=True)
                xlabel(self.labelGraficos()[i])
                ylabel(u'Frequência')
                fig.savefig(base_path+base_dir+'residuos_histograma')
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
                fig.savefig(base_path+base_dir+'residuos_probplot')
                close()
                
        if ('experimental' in ID or 'validacao' in ID or 'calculado' in ID):
                
                if 'residuo' in ID: # remover de ID o resíduo, pois foi tratado separadamente
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
                        
                        fig = figure()
                        ax  = fig.add_subplot(1,1,1)
                        plot(x,dados,'o')
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
                        ymin   = min(dados) - tamanho_tick_y
                        ymax   = max(dados) + tamanho_tick_y
                        xlim(xmin,xmax)
                        ylim(ymin,ymax)
                        # Labels
                        xlabel(u'Número de pontos',fontsize=20)
                        ylabel(self.labelGraficos(atributo)[i],fontsize=20)
                        #Grades
                        grid(b = 'on', which = 'major', axis = 'both')
                        savefig(base_path+base_dir+atributo+'_observacoes.png')
                        close()   


