# -*- coding: utf-8 -*-
"""
Arquivo que contém subrotinas genéricas para uso pelo MT_PEU.

@author: Daniel
"""

from numpy import concatenate, size, arctan2, degrees, sqrt, \
    copy, ones, array, cos, sin, pi, roots, linspace, iscomplex, transpose, dot
from numpy.linalg import eigh, inv
from os import path, makedirs

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot, gca

from matplotlib.patches import Ellipse

def WLS (parametros,*argumentos):
    u"""
    Subrotina para ......
    """
    argumentos = argumentos[0]

    y = argumentos[0] #dados experimentais da grandeza dependente (x)
    x = argumentos[1] #dados experimentais da grandeza independente (y)
    Vy = argumentos[2] #incerteza (Uy)
    Vx = argumentos[3] #incerteza (Ux)
    args = argumentos[4] #argumentos externos passados pelo usuário

    # Modelo
    modelo = argumentos[5]

    # Simbologia (especificidade do PEU)
    simb_x = argumentos[6]
    simb_y = argumentos[7]
    simb_parametros = argumentos[8]

    ym = matriz2vetor(modelo(parametros, x, [args, simb_x, simb_y, simb_parametros]))

    d = y - ym
    result = float(dot(dot(transpose(d), inv(Vy)), d))

    return result
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

def eval_cov_ellipse(cov, pos, c2=2, **kwargs):
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

        Additional keyword arguments are pass on to the ellipse patch.

    Returns
    -------
        Information used in ellipse construction
        # Código é adaptado e obtigo de terceiros: https://github.com/joferkington/oost_paper_code/blob/master/error_ellipse.py
    """
    def eigsorted(cov):
        vals, vecs = eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]


    vals, vecs = eigsorted(cov)
    theta = degrees(arctan2(*vecs[:,0][::-1]))
    # Width and height are "full" widths, not radius
    width, height = 2 * sqrt(c2*vals)

    # CÁLCULO DOS PONTOS PERTENCENTES AOS EIXOS DA ELIPSE:
    invcov = inv(cov)
    alpha  = [vecs[1,0]/vecs[0,0],vecs[1,1]/vecs[0,1]]
    lamb   = [sqrt(c2/(invcov[0,0]+2*alpha_i*invcov[0,1] + alpha_i**2*invcov[1,1])) for alpha_i in alpha]

    coordenadas_x = [pos[0]+lamb[0],pos[0]-lamb[0],pos[0]+lamb[1],pos[0]-lamb[1]]
    coordenadas_y = [pos[1]+alpha[0]*lamb[0],pos[1]-alpha[0]*lamb[0],pos[1]+alpha[1]*lamb[1],pos[1]-alpha[1]*lamb[1]]


    # CÁLCULO DOS PONTOS EXTREMOS
    k = invcov[0,0]/(invcov[0,1] + 1e-100) # 1e-100 evita NaN quando invcov[0,1] é igual a zero.
    delta = sqrt(c2/(k**2*invcov[1,1]-2*k*invcov[0,1]+invcov[0,0]))
    coordenadas_x.extend([pos[0]+delta,pos[0]-delta])
    coordenadas_y.extend([pos[1]-delta*k,pos[1]+delta*k])

    k = invcov[1,1]/(invcov[0,1] + 1e-100) # 1e-100 evita NaN quando invcov[0,1] é igual a zero.

    delta = sqrt(c2/(k**2*invcov[0,0]-2*k*invcov[0,1]+invcov[1,1]))
    coordenadas_y.extend([pos[1]+delta,pos[1]-delta])
    coordenadas_x.extend([pos[0]-delta*k,pos[0]+delta*k])

    return coordenadas_x, coordenadas_y, width, height, theta

def matrizcorrelacao(matriz_covariancia):
    u"""
    Calcula a matriz de correlação de determinada matriz covariância
    """
    if size(matriz_covariancia,0) != size(matriz_covariancia,1):
        raise ValueError(u'A matriz precisa ser quadrada para calcular a matriz dos coeficientes de correlação.')

    matriz_correlacao  = ones((size(matriz_covariancia,0),size(matriz_covariancia,0)))
    for i in range(size(matriz_covariancia,0)):
        for j in range(size(matriz_covariancia,0)):
            matriz_correlacao[i,j]  = matriz_covariancia[i,j]/sqrt(matriz_covariancia[i,i]*matriz_covariancia[j,j])

    return matriz_correlacao

def lista2matriz(lista):
    res = array(lista[0],ndmin=2).transpose()
    for i in lista[1:]:
        aux = array(i,ndmin=2).transpose()
        res = concatenate((res,aux),1)

    return res

