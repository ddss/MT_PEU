# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 11:05:02 2015

@author: danielsantana
"""
# Importação de pacotes de terceiros
from numpy import array, transpose ,size, diag, linspace, min, max, \
    mean,  std, amin, amax, ndarray, ones, sqrt, insert, nan, shape


from scipy.stats import normaltest, anderson, shapiro, ttest_1samp, kstest,\
 norm, probplot, ttest_ind

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot

from os import getcwd, sep
from statsmodels.stats.diagnostic import acorr_breush_godfrey, acorr_ljungbox, het_breushpagan, het_white
from statsmodels.stats.stattools import durbin_watson

# Subrotinas próprias (desenvolvidas pelo GI-UFBA)
from subrotinas import matriz2vetor, vetor2matriz, Validacao_Diretorio, matrizcorrelacao

class Organizador:
    
    def __init__(self,estimativa,incerteza,gL=[],tipos={'estimativa':'matriz','incerteza':'incerteza'},NE=None):
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
        * ``gL''(lista)           : graus de liberdade
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
        
        if not isinstance(estimativa,ndarray):
                raise TypeError(u'Os dados de entrada precisam ser arrays.')
        
        if incerteza is not None:
            if not isinstance(incerteza,ndarray):
                    raise TypeError(u'Os dados de entrada precisam ser arrays.')            
        
        if not isinstance(gL,list):
			raise TypeError(u'os graus de liberdade precisam ser listas')
		
		# TODO: Corrigir este teste
        #if (len(gL) != size(estimativa)) and (len(gL) != 0) :
	    #		raise ValueError(u'Os graus de liberdade devem ter o mesmo tamanho das estimativas')    
		
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

            if incerteza is not None:
                self.matriz_incerteza   = incerteza            
                self.matriz_covariancia = diag(transpose(matriz2vetor(self.matriz_incerteza**2)).tolist()[0])
                self.matriz_correlacao  = matrizcorrelacao(self.matriz_covariancia)

        if tipos['incerteza'] == 'variancia':

            if incerteza is not None:
                self.matriz_covariancia = incerteza     
                self.matriz_incerteza   = vetor2matriz(array(diag(self.matriz_covariancia)**0.5,ndmin=2).transpose(),NE)
                self.matriz_correlacao  = matrizcorrelacao(self.matriz_covariancia)

        # ---------------------------------------------------------------------
        # Criação dos atributos na forma de LISTAS
        # ---------------------------------------------------------------------
        self.lista_estimativa = self.matriz_estimativa.transpose().tolist()
        
        if incerteza is not None:
            self.lista_incerteza  = self.matriz_incerteza.transpose().tolist()
            self.lista_variancia  = diag(self.matriz_covariancia).tolist()
        else:
            self.lista_incerteza  = None
            self.lista_variancia  = None

        # ---------------------------------------------------------------------
        # Número de pontos experimentais e de variáveis
        # ---------------------------------------------------------------------

        self.NE = size(self.matriz_estimativa,0)

        # ---------------------------------------------------------------------
        # Graus de liberdade
        # ---------------------------------------------------------------------   
        self.gL = gL if len(gL) !=0 else [[100]*self.NE]*size(self.matriz_estimativa,1)

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

    def _SETexperimental(self,estimativa,variancia,gL,tipo):

        self.__ID.append('experimental')
        self.experimental = Organizador(estimativa,variancia,gL,tipo)        
        
    def _SETvalidacao(self,estimativa,variancia,gL,tipo):
        
        self.__ID.append('validacao')
        self.validacao = Organizador(estimativa,variancia,gL,tipo)

    def _SETcalculado(self,estimativa,variancia,gL,tipo,NE):
        
        self.__ID.append('calculado')
        self.calculado = Organizador(estimativa,variancia,gL,tipo,NE)     
 
    def _SETresiduos(self,estimativa,variancia,gL,tipo):
        
        self.__ID.append('residuo')
        self.residuos = Organizador(estimativa,variancia,gL,tipo)  

    def _SETparametro(self, estimativa, variancia, regiao):

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
            try:
                shape(variancia)[1]
            except:
                raise TypeError(u'A variância precisa ser um array com duas dimensões.')

            if shape(variancia)[0] != shape(variancia)[1]:
                raise TypeError(u'A variância precisa ser quadrada.')

            if shape(variancia)[0] != self.NV:
                    raise ValueError(u'A dimensão da matriz de covariãncia deve ser coerente com os simbolos dos parâmetros')

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
            self.matriz_incerteza   = vetor2matriz(array(diag(self.matriz_covariancia)**0.5,ndmin=2).transpose(),self.NV)
        self.regiao_abrangencia = regiao

 
    def labelGraficos(self,add=None):
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

    def _testesEstatisticos(self,z):
        u'''
        Subrotina para realizar testes estatísticos nos resíduos

        =======
        Entrada
        =======
        * z: variáveis para explicadores. Objetivo: avaliar homocedasticidade

        =================
        Testes realizados
        =================
        
        **NORMALIDADE**:
        
        * normaltest: Retorna o pvalor do teste de normalidade. Hipótese nula: a amostra vem de distribuição normal
        * shapiro   : Retorna o pvalor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal
        * anderson  : Retorna a estatistica do Teste para os dados provenientes de uma distribuição Normal e um parâmetro comparativo para PA de confiança, onde se a estatistica for maior que os valores críticos (1-PA) tem-se que: a hipótese nula de que os dados vem da distribuição normal pode ser rejeitada.
                      A lista de valores críticos (nível se significância) é para 15%, 10%, 5%, 2.5%, 1%, respectivamente.
                      Hipotese nula: a amostra vem de uma normal
        * kstest    : Retorna o pvalor de normalidade. Hipótese nula: a amostra vem de uma distribuição normal

        **MÉDIA**:
        
        * ttest_1sam: Retorna o pvalor para média determinada. Hipótese nula: a amostra tem a média determinada
       
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
         
        =====
        SAÍDA
        =====
        Sob a forma de atributo:
        * estatisticas (dict):  p-valor das hipóteses testadas. Para a hipótese nula tida como verdadeira,
        um valor abaixo de (1-PA) nos diz que para PA de confiança pode-se rejeitar essa hipótese.

        OBS: para o teste de durbin_watson e anderson Darling, são retornadas estatísticas e não p-valores.

        =================
        Referências
        =================
        [1] White, H. (1980). "A Heteroskedasticity-Consistente Covariance Matrix Estimador e um teste direto para Heteroskedasticity". Econometrica 48 (4):. 817-838 JSTOR 1.912.934 . MR 575027 .

        '''
        # TODO: Verificar teste de acorr_ljungbox
        # TODO: detalhar melhor a documentação
        # TODO: revise as documentações para avaliar se os valores de resposta estão coerentes
    
        if 'residuo' in self.__ID: # Testes para os resíduos
            # Variável para salvar os nomes dos testes estatísticos - consulta
            self.__nomesTestes = {'residuo-Normalidade':['normaltest','shapiro', 'anderson','kstest'],
                                  'residuo-Media':['ttest'],
                                  'residuo-Autocorrelacao':['Durbin Watson','Ljung-Box'],
                                  'residuo-Homocedasticidade':['white test','Bresh Pagan']}
            pvalor = {}
            for nome in self.simbolos:
                pvalor[nome] = {}

            for i,nome in enumerate(self.simbolos):
                dados = self.residuos.matriz_estimativa[:,i]
        
                # Lista que contém as chamadas das funções de teste:
                if size(dados) < 3: # Se for menor do que 3, não se pode executar o teste de shapiro
                    pnormal=[None, None, anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':None, 'shapiro':None, 'anderson':{'estatistica A_Darling':pnormal[2][0],'valores criticos':pnormal[2][1]},'kstest':pnormal[3][1]}

                elif size(dados) < 20: # Se for menor do que 20 não será realizado no normaltest, pois ele só é válido a partir dste número de dados
                    pnormal=[None, shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':None, 'shapiro':pnormal[1][1], 'anderson':{'estatistica A_Darling':pnormal[2][0],'valores criticos':pnormal[2][1]},'kstest':pnormal[3][1]}
                else:
                    pnormal=[normaltest(dados), shapiro(dados), anderson(dados, dist='norm'),kstest(dados,'norm',args=(mean(dados),std(dados,ddof=1)))]                
                    pvalor[nome]['residuo-Normalidade'] = {'normaltest':pnormal[0][1], 'shapiro':pnormal[1][1], 'anderson':{'estatistica A_Darling':pnormal[2][0],'valores criticos':pnormal[2][1]},'kstest':pnormal[3][1]}

                # Testes para a média:
                pvalor[nome]['residuo-Media'] = {'ttest':float(ttest_1samp(dados,0.)[1])}
             
                # Testes para a autocorrelação:
                #ljungbox = acorr_ljungbox(dados, lags=None, boxpierce=True)
                pvalor[nome]['residuo-Autocorrelacao'] = {'Durbin Watson':{'estatistica':durbin_watson(dados)}}#, 'Ljung-Box':{'p-valor chi2':ljungbox[1],'p-valor Box-Pierce test':ljungbox[3]}}
                
                # Testes para a Homocedásticidade:
                # TODO: Avaliar se z deve ser passado para cada coluna da entrada.
                pheter= [het_white(dados,insert(z, 0, 1, axis=1)),het_breushpagan(dados,z)]
                pvalor[nome]['residuo-Homocedasticidade'] = {'white test':{'p-valor multiplicador de Lagrange':pheter[0][1], 'p-valor Teste F':pheter[0][3]},'Bresh Pagan':{'p-valor multiplicador de lagrange':pheter[1][1],'p-valor Teste F':pheter[1][3]}}
        else:
            raise NameError(u'Os testes estatísticos são válidos apenas para o resíduos')

        self.estatisticas = pvalor
            
    def Graficos(self,base_path=None,ID=None,fluxo=None):
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
        base_dir  = sep + 'Grandezas' + sep
        Validacao_Diretorio(base_path,base_dir)

        if 'residuo' in ID:
            # BOXPLOT
            #checa a variabilidade dos dados, assim como a existência de possíveis outliers
            fig = figure()
            ax = fig.add_subplot(1,1,1)
            boxplot(self.residuos.matriz_estimativa)
            ax.set_xticklabels(self.simbolos)
            fig.savefig(base_path+base_dir+'residuos_fl'+str(fluxo)+'_boxplot_'+'_'.join(self.simbolos))
            close()
            
            base_path = base_path + base_dir
            for i,nome in enumerate(self.simbolos):
                # Gráficos da estimação
                base_dir = sep + self.simbolos[i] + sep
                Validacao_Diretorio(base_path,base_dir)

                dados = self.residuos.matriz_estimativa[:,i]
        
                # TENDÊNCIA
                #Testa a aleatoriedade dos dados, plotando os valores do residuo versus a ordem em que foram obtidos
                #dessa forma verifica-se há alguma tendência
                # TODO: colocar legenda
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                plot(linspace(1,size(dados),num=size(dados)),dados, 'o')
                plot(linspace(1,size(dados),num=size(dados)),[mean(dados)]*size(dados),'-.r')
                xlabel('Ordem de Coleta')
                ylabel(self.simbolos[i])
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
                xlim((0,size(dados)))
                ax.axhline(0, color='black', lw=2)
                fig.savefig(base_path+base_dir+'residuos_fl'+str(fluxo)+'_ordem')
                close()        
        
                # AUTO CORRELAÇÃO
                #Gera um gráfico de barras que verifica a autocorrelação
                fig = figure()
                ax = fig.add_subplot(1,1,1)
                ax.acorr(dados,usevlines=True, normed=True,maxlags=None)
                ax.yaxis.grid(color='gray', linestyle='dashed')                        
                ax.xaxis.grid(color='gray', linestyle='dashed')
                ax.axhline(0, color='black', lw=2)
                xlim((0,size(dados)))
                fig.savefig(base_path+base_dir+'residuos_fl'+str(fluxo)+'_autocorrelacao')
                close()

                # HISTOGRAMA                
                #Gera um gráfico de histograma, importante na verificação da pdf
                fig = figure()
                hist(dados, normed=True)
                xlabel(self.labelGraficos()[i])
                ylabel(u'Frequência')
                fig.savefig(base_path+base_dir+'residuos_fl'+str(fluxo)+'_histograma')
                close()

                # NORMALIDADE 
                #Verifica se os dados são oriundos de uma pdf normal, o indicativo disto é a obtenção de uma reta              
                res = probplot(dados, dist='norm', sparams=(mean(dados),std(dados,ddof=1)))
                if not (nan in res[0][0].tolist() or nan in res[0][1].tolist() or nan in res[1]):

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
                    fig.savefig(base_path+base_dir+'residuos_fl'+str(fluxo)+'_probplot')
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
                        savefig(base_path+base_dir+atributo+'_fl'+str(fluxo)+'_observacoes.png')
                        close()