# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:36:09 2014

@author: Daniel
"""

from numpy import concatenate, size, arctan2, degrees, sqrt, copy, ones, array
from numpy.linalg import eigh
from os import path, makedirs

from matplotlib.pyplot import gca
from matplotlib.patches import Ellipse

#THREAD
from Queue import Queue, Empty



def matriz2vetor(matriz):
    u'''
    Subrotina para converter uma matriz (array com várias colunas) em um vetor (array com uma coluna)
    =======
    Entrada
    =======
    * ``matriz`` (array): matriz que se deseja converter para vetor
    
    =====
    Saída
    =====
    * ``vetor`` (array): array com uma coluna
    
    =======
    Exemplo
    =======
    Exemplo: ::
        
        >>> from numpy import array, transpose
        >>> x1 = transpose(array([[1,2,3,4,5]]))
        >>> x2 = transpose(array([[6,7,8,9,10]]))
        >>> matriz  = concatenate((x1,x2),axis=1)
        >>> vetor   = matriz2vetor(matriz)
    '''
    # Obtenção da primeira coluna da matriz    
    vetor = matriz[:,0:1]
    for i in xrange(1,size(matriz,1)):
        # Concatenar as colunas abaixo da anterior
        vetor = concatenate((vetor,matriz[:,i:i+1]))
    
    return vetor


def vetor2matriz(vetor,NE):
    u'''
    Subrotina para converter um vetor (array com uma coluna) em um vetor (array com uma coluna)

    =======
    Entrada
    =======
    * ``vetor`` (array): vetor que se deseja converter para matriz
    * ``NE`` (float): quantidade de dados de cada coluna da matriz
    
    =====
    Saída
    =====
    
    * ``matriz`` (array): matriz convertida
    
    =======
    Exemplo
    =======
    Exemplo: ::
        
        >>> from numpy import array, transpose
        >>> x1 = transpose(array([[1,2,3,4,5]]))
        >>> x2 = transpose(array([[6,7,8,9,10]]))
        >>> vetor  = concatenate((x1,x2),axis=0)
        >>> matriz   = vetor2matriz(vetor,5)
    '''

    pos_inicial = 0  # primeira linha da primeira coluna
    pos_final   = NE # última linha da primeira coluna 
    matriz = vetor[pos_inicial:pos_final] # gera a primeira coluna

    for i in xrange(1,size(vetor)/NE):
        pos_inicial += NE
        pos_final   += NE
        # concatenando as colunas uma ao lado da outra
        matriz = concatenate((matriz,vetor[pos_inicial:pos_final]),1) 
        
    return matriz


def Validacao_Diretorio(base_path,diretorio=None):
        # Baseado em código de terceiros
        # Validação da existência de diretório
        if diretorio != None:
            directory = path.split(base_path+diretorio+"Teste.txt")[0]
        else:
            directory = path.split(base_path)[0]
        
        if directory == '':
            directory = '.'
 
        # Se o diretório não existir, crie
        if not path.exists(directory):
            makedirs(directory)

def Coef_R2(residuo,yexp):
        residuo = residuo
        yexp    = yexp.tolist()[0]
        
        SEline  = residuo**2
        SEy     = sum([(ye - sum(yexp)/size(residuo))**2 for ye in yexp])
        
        return 1 -  SEline/SEy

def CovarianciaXY(matriz_cov_x,matriz_cov_y):
    matriz_cov_xy = concatenate((matriz_cov_x,matriz_cov_y))
    
    
    
def plot_cov_ellipse(cov, pos, c2=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the 
    ellipse patch artist.

    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the 
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        A matplotlib ellipse artist
        # Código é adaptado e obtigo de terceiros: https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    """
    def eigsorted(cov):
        vals, vecs = eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = gca()

    vals, vecs = eigsorted(cov)
    theta = degrees(arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * sqrt(c2*vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
    
    ax.add_artist(ellip)
    return (ellip, width, height, theta)
    
def vetor_delta(entrada_vetor,posicao,delta):
    
    u'''
    Subrotina para alterar o(s) elementos de um vetor, acrescentando ou retirando um determinado ''delta''.
    =======
    Entrada
    =======
    
    *``entrada_vetor´´(array, ndmi=1): Vetor ao qual a posição i e j ou apenas i será alterada.
    *``posicao´´(list ou float)= posição a ser repassada pela estrutura 'for'.
    *``delta´´(float): valor do incremento, proporcional a grandeza que será acrescida ou decrescida.
    
    
    =====
    Saída
    =====
    * ``vetor`` (array): array com uma linha.
    
    =======
    Exemplo
    =======
    Exemplo: ::
        
        >>>from numpy import array
        >>>from subrotinas import vetor_delta
        
        >>> x1 =(array([1,2,3,4,5]))
        
        print vetor_delta(x1,3,5)
        
        >>> array([1, 2, 3, 9, 5])
    '''
            
    vetor = copy(entrada_vetor)

    if isinstance(posicao,list):
        vetor[posicao[0]] = vetor[posicao[0]]+delta[0]
        vetor[posicao[1]] = vetor[posicao[1]]+delta[1]
    else:
        vetor[posicao] = vetor[posicao]+delta
                
    return vetor
    
def matrizcorrelacao(matriz_covariancia):
    u'''
    Calcula a matriz de correlação de determinada matriz covariância
    '''
    if size(matriz_covariancia,0) != size(matriz_covariancia,1):
        raise ValueError(u'A matriz precisa ser quadrada para calcular a matriz dos coeficientes de correlação.')
    
    matriz_correlacao  = ones((size(matriz_covariancia,0),size(matriz_covariancia,0)))    
    for i in xrange(size(matriz_covariancia,0)):
        for j in xrange(size(matriz_covariancia,0)):
            matriz_correlacao[i,j]  = matriz_covariancia[i,j]/sqrt(matriz_covariancia[i,i]*matriz_covariancia[j,j])

    return matriz_correlacao

def lista2matriz(lista):
    res = array(lista[0],ndmin=2).transpose()
    for i in lista[1:]:
        aux = array(i,ndmin=2).transpose()
        res = concatenate((res,aux),1)
    
    return res
    
def ThreadExceptionHandling(Thread,argumento1,argumento2,argumento3):
    u'''
    Subrotina para lidar com exceptions em Thread.
    
    =======
    Entrada
    =======
    
    * Thread: deve ser uma Thread com a seguinte estrutura [1]::
        
    >>> import threading
    >>> import Queue
    >>>
    >>> class ExcThread(threading.Thread):
    >>>
    >>>     def __init__(self, bucket):
    >>>         threading.Thread.__init__(self)
    >>>         self.bucket = bucket
    >>>
    >>>     def run(self):
    >>>         try:
    >>>             raise Exception('An error occured here.')
    >>>         except Exception:
    >>>              self.bucket.put(sys.exc_info())        
    
    Referência:
    
    [1] http://stackoverflow.com/questions/2829329/catch-a-threads-exception-in-the-caller-thread-in-python
    
    '''
    bucket = Queue()
    thread_obj = Thread(argumento1,argumento2,argumento3,bucket=bucket)
    thread_obj.start()

    while True:
        try:
            exc = bucket.get(block=False)
        except Empty:
            pass
        else:
            # Informações sobre o erro ocorrido:
            exc_type, exc_obj, exc_trace = exc

            raise SyntaxError('Erro no modelo. Erro identificado "%s" no modelo.'%exc_obj)
            
        thread_obj.join(0.1)
        if thread_obj.isAlive():
            continue
        else:
            break

class flag:
    
    def __init__(self):
        u'''Classe para padronizar o uso de flags no motor de cálculo.
        
        =========
        Atributos
        =========

        * **info**: dicionário que informa a situação atual das flags. As chaves são 'status' (TRUE ou FALSE) e 'descrição' (o que está ocorrendo)
        
        * **resumo**: apresenta um resumo completo de todas as possibilidades de status das flags e seus respectivos significados
                    
        =======
        Métodos
        =======
        
        * **ToggleActive(caracteristica)** : muda o status da caracteristica da flag para TRUE
        * **ToggleInactive(caracteristica)**: muda o status da caracteristica da flag para FALSE
        
        É necessário informar característica que está sendo modificada. Características disponíveis: 'dadosvalidacao' e 'reconciliacao'.
        '''               
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        self._caracteristicas_disponiveis = ['dadosexperimentais','dadosvalidacao','reconciliacao','calc_termo_independente']

        self.info = {}
        for elemento in self._caracteristicas_disponiveis:
            self.info[elemento] = False
        
    def __validacao(self,caracteristica):
        u'''Validação das entradas
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if isinstance(caracteristica,str):
            
            if caracteristica not in self._caracteristicas_disponiveis:
                raise NameError(u'A caracteristica "'+str(caracteristica)+'" não está disponível. Características disponíveis: '+', '.join(self._caracteristicas_disponiveis)+'.')
        
        elif isinstance(caracteristica,list):
        
            teste = [isinstance(elemento,str) for elemento in caracteristica]
            
            if False in teste:
                raise TypeError('As características devem ser strings.')
            
            diferenca = set(caracteristica).difference(set(self._caracteristicas_disponiveis))
            if len(diferenca) != 0:
                raise NameError(u'Característica(s) indisponível(is): '+', '.join(diferenca) +'. Características disponíveis: '+', '.join(self._caracteristicas_disponiveis)+'.')
                
        else:            
            raise TypeError(u'A caracteristica deve ser uma lista ou um string.')
                    
        # ---------------------------------------------------------------------
        # AÇÃO
        # ---------------------------------------------------------------------      
        if not isinstance(caracteristica,list):
            caracteristica = [caracteristica]
         
        self.__caracteristica = caracteristica
        
    def __Toggle(self):
        u''' Método interno para realizar ação de mudança de status
        
        '''
        for elemento in self.__caracteristica:
            self.info[elemento]    = self.__togglestatus 
    
    def ToggleActive(self,caracteristica):
        '''
        Irá marcar a flag como TRUE
        =======
        Entrada
        =======
        
        * característica (lista de strings ou string): o que a flag está indicando
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         
        self.__validacao(caracteristica)

        self.__togglestatus = True
        self.__Toggle()

    def ToggleInactive(self,caracteristica):
        '''
        Irá marcar a flag como FALSE
        =======
        Entrada
        =======
        
        * característica (lista de strings ou string): o que a flag está indicando
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------         
        self.__validacao(caracteristica)

        self.__togglestatus = False
        self.__Toggle()
