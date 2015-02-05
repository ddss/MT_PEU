# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 10:36:09 2014

@author: Daniel
"""

from numpy import concatenate, size, arctan2, degrees, sqrt, copy
from numpy.linalg import eigh
from os import path, makedirs

from matplotlib.pyplot import gca
from matplotlib.patches import Ellipse


def matriz2vetor(matriz):
    '''
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
    '''
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
    return (ellip, width, height)
    
def vetor_delta(entrada_vetor,posicao,delta):
            
    vetor = copy(entrada_vetor)

    if isinstance(posicao,list):
        vetor[posicao[0]] = vetor[posicao[0]]+delta[0]
        vetor[posicao[1]] = vetor[posicao[1]]+delta[1]
    else:
        vetor[posicao] = vetor[posicao]+delta
                
    return vetor