# -*- coding: utf-8 -*-
"""
Arquivo que contém subrotinas genéricas para uso pelo MT_PEU.

@author: Daniel
"""

from numpy import concatenate, size, arctan2, degrees, sqrt, \
    copy, ones, array, cos, sin, pi, roots, linspace, iscomplex
from numpy.linalg import eigh, inv
from os import path, makedirs

from matplotlib.pyplot import figure, axes, axis, plot, errorbar, subplot, xlabel, ylabel,\
    title, legend, savefig, xlim, ylim, close, grid, text, hist, boxplot, gca

from matplotlib.patches import Ellipse

def matriz2vetor(matriz):
    u"""
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
    """
    # Obtenção da primeira coluna da matriz    
    vetor = matriz[:,0:1]
    for i in xrange(1,size(matriz,1)):
        # Concatenar as colunas abaixo da anterior
        vetor = concatenate((vetor,matriz[:,i:i+1]))
    
    return vetor


def vetor2matriz(vetor,NE):
    u"""
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
    """

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

    # Comprimento dos eixos da elipse
    b = height/2. # MENOR EIXO
    a = width/2.  # MAIOR EIXO

    alpha = theta*pi/180.

    if theta < 0:
        alpha += pi

    if alpha >= pi:
        alpha -= pi

    if 0 <= alpha <= pi/2.:
        h_maior_eixo = (cos(alpha)*a, sin(alpha)*a)
        h_menor_eixo = (cos(pi-alpha-pi/2.)*b,sin(pi-alpha-pi/2.)*b)
        # Cálculo dos pontos extremos para cada eixo
        pontos_maior_eixo = ((pos[0] + h_maior_eixo[0], pos[1] + h_maior_eixo[1]),
                             (pos[0] - h_maior_eixo[0], pos[1] - h_maior_eixo[1]))
        pontos_menor_eixo = ((pos[0] - h_menor_eixo[0], pos[1] + h_menor_eixo[1]),
                             (pos[0] + h_menor_eixo[0], pos[1] - h_menor_eixo[1]))

    else:
        h_maior_eixo = (cos(pi-alpha)*a,sin(pi-alpha)*a)
        h_menor_eixo = (cos(alpha-pi/2.)*b,sin(alpha-pi/2.)*b)
        # Cálculo dos pontos extremos para cada eixo
        pontos_maior_eixo = ((pos[0] + h_maior_eixo[0], pos[1] - h_maior_eixo[1]),
                             (pos[0] - h_maior_eixo[0], pos[1] + h_maior_eixo[1]))
        pontos_menor_eixo = ((pos[0] + h_menor_eixo[0], pos[1] + h_menor_eixo[1]),
                             (pos[0] - h_menor_eixo[0], pos[1] - h_menor_eixo[1]))

    invcov = inv(cov)

    # Equação da Elipse
    coordenadas_x = [pontos_maior_eixo[0][0],pontos_maior_eixo[1][0],pontos_menor_eixo[0][0],pontos_menor_eixo[1][0]]
    limite_max = max(coordenadas_x)
    limite_min = min(coordenadas_x)
    teste = True
    h = (limite_max - limite_min)/1000
    theta1 = limite_max
    while teste:
        theta1 = theta1 + h
        deltatheta1 = theta1 - pos[0]
        a = invcov[1,1]
        b = 2*invcov[0,1]*deltatheta1
        c = deltatheta1**2*invcov[0,0]-c2
        deltatheta2 = roots([a,b,c])
        if iscomplex(deltatheta2[0]):
            teste = False
    limite_max = theta1-h
    theta1 = limite_min
    teste = True
    while teste:
        theta1 = theta1 - h
        deltatheta1 = theta1 - pos[0]
        a = invcov[1,1]
        b = 2*invcov[0,1]*deltatheta1
        c = deltatheta1**2*invcov[0,0]-c2
        deltatheta2 = roots([a,b,c])
        if iscomplex(deltatheta2[0]):
            teste = False
    limite_min = theta1+h

    coordenadas_y =  [pontos_maior_eixo[0][1],pontos_maior_eixo[1][1],pontos_menor_eixo[0][1],pontos_menor_eixo[1][1]]
    limite_max_y = max(coordenadas_y)
    limite_min_y = min(coordenadas_y)

    teste = True
    h = (limite_max_y - limite_min_y)/1000
    theta2 = limite_max_y
    while teste:
        theta2 = theta2 + h
        deltatheta2 = theta2 - pos[1]
        a = invcov[0,0]
        b = 2*invcov[1,0]*deltatheta2
        c = deltatheta2**2*invcov[1,1]-c2
        deltatheta1 = roots([a,b,c])
        if iscomplex(deltatheta1[0]):
            teste = False
    limite_max_y = theta2-h
    theta2 = limite_min_y
    teste = True
    while teste:
        theta2 = theta2 - h
        deltatheta2 = theta2 - pos[1]
        a = invcov[0,0]
        b = 2*invcov[1,0]*deltatheta2
        c = deltatheta2**2*invcov[1,1]-c2
        deltatheta1 = roots([a,b,c])
        if iscomplex(deltatheta1[0]):
            teste = False
    limite_min_y = theta2+h

    lista_t1 = []
    lista_t2 = []
    for theta1 in linspace(limite_min,limite_max,num=1000):
        deltatheta1 = theta1 - pos[0]
        a = invcov[1,1]
        b = 2*invcov[0,1]*deltatheta1
        c = deltatheta1**2*invcov[0,0]-c2
        deltatheta2 = roots([a,b,c])
        lista_t1.extend([theta1,theta1])
        lista_t2.extend([deltatheta2[0]+pos[1],deltatheta2[1]+pos[1]])

    return ellip, pontos_maior_eixo, pontos_menor_eixo,limite_max,limite_min, limite_max_y, limite_min_y, lista_t1, lista_t2
    
def vetor_delta(entrada_vetor,posicao,delta):
    u"""
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
    """
            
    vetor = copy(entrada_vetor)

    if isinstance(posicao,list):
        vetor[posicao[0]] = vetor[posicao[0]]+delta[0]
        vetor[posicao[1]] = vetor[posicao[1]]+delta[1]
    else:
        vetor[posicao] = vetor[posicao]+delta
                
    return vetor

def matrizcorrelacao(matriz_covariancia):
    u"""
    Calcula a matriz de correlação de determinada matriz covariância
    """
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

def graficos_x_y(X, Y, ix, iy, base_path, base_dir, info, ID_fluxo):
    u"""
    Subrotina para gerar gráficos das variáveis y em função de x

    =======
    Entrada
    =======
    * X: objeto Grandeza contendo os dados a grandeza dependente
    * Y: objeto Grandeza contendo os dados a grandeza independente
    * ix: posição da variável que se deseja plotar
    * iy: posição da variável que se deseja plotar em função de x[ix]
    * info: atributo de Grandeza que se deseja plotar

    * base_path: caminho base
    * base_dir : diretório base

    * ID_fluxo: número que indica o fluxo de trabalho

    =======
    Saídas
    =======
    * Gráfico de y em função de x sem incerteza
    * Gráfico de y em função de x com incerteza
    """

    x  = eval('X.'+info+'.matriz_estimativa[:,ix]')
    y  = eval('Y.'+info+'.matriz_estimativa[:,iy]')
    ux = eval('X.'+info+'.matriz_incerteza[:,ix]')
    uy = eval('Y.'+info+'.matriz_incerteza[:,iy]')

    #Gráfico apenas com os pontos experimentais
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    plot(x,y,'o')
    # obtençao do tick do grafico
    # eixo x+
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
    xlabel(X.labelGraficos(info)[ix],fontsize=20)
    ylabel(Y.labelGraficos(info)[iy],fontsize=20)
    #Grades
    grid(b = 'on', which = 'major', axis = 'both')
    fig.savefig(base_path+base_dir+info+'_fl'+str(ID_fluxo)+'_'+Y.simbolos[iy]+'_funcao_'+X.simbolos[ix]+'_sem_incerteza')
    close()

    #Grafico com os pontos experimentais e as incertezas
    fig = figure()
    ax = fig.add_subplot(1,1,1)
    xerr = 2*ux
    yerr = 2*uy
    errorbar(x,y,xerr=xerr,yerr=yerr,fmt=None, marker='o')
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
    xlabel(X.labelGraficos(info)[ix],fontsize=20)
    ylabel(Y.labelGraficos(info)[iy],fontsize=20)
    #Grades
    grid(b = 'on', which = 'major', axis = 'both')
    fig.savefig(base_path+base_dir+info+'_fl'+str(ID_fluxo)+'_'+Y.simbolos[iy]+'_funcao_'+X.simbolos[ix]+'_com_incerteza')
    close()