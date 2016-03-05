# -*- coding: utf-8 -*-
"""
Arquivo que contém a classe gráficos

@author: Daniel
"""

# Importação de pacotes

from numpy import arctan2, degrees, sqrt, sort, argsort

from numpy.linalg import eigh, inv

from matplotlib.pyplot import figure, close, clf

from matplotlib.patches import Ellipse

class Grafico:
    def __init__(self, **kwargs):
        u"""
        Classe para a criação e gerenciamento de gráficos

        =======
        Entrada
        =======

        * kwargs: argumentos para matplotlib.pyplot.figure

        =========
        Atributos
        =========

        Esta classe usa os atributos self.axes (matplotlib.axes.Axes) e self.fig_instance (matplotlib.pyplot.figure)
        para criação dos gráficos. O atributo self.lista_graficos é usado para salvar os gráficos incluído para
        edições na legenda.
        """
        # incia a figura
        self.fig_instance = figure(**kwargs)
        # inicia o axes
        self.axes = self.fig_instance.add_subplot(1, 1, 1)
        # lista de gráficos executados -> para legenda
        self.lista_graficos = []

    def config_axes(self, offset_x=False, offset_y=False):
        u"""
        Método voltado à formatação de self.axes

        Formatações: configuração dos ticks, eixos em notação científica, offset dos eixos, e grid.
        """
        # Formato dos ticks
        self.axes.tick_params(reset=True, axis='both', right='off',top='off',direction='out', labelsize=14)

        # Definindo notação científica para os eixos
        self.axes.get_xaxis().get_major_formatter().set_powerlimits((-2, 2))
        self.axes.get_yaxis().get_major_formatter().set_powerlimits((-2, 2))

        # Definindo que offset nos eixos
        self.axes.get_xaxis().get_major_formatter().set_useOffset(offset_x)
        self.axes.get_yaxis().get_major_formatter().set_useOffset(offset_y)

        # Definindo linhas de grade no major axes
        self.axes.grid(b='on', which='major', axis='both', linestyle='dashed', color='gray')

    def get_step_tick(self):
        u"""
        Método para avaliar o tamanho do passo dos ticks dos eixos dos gráficos

        :return: passo dos ticks de x e y
        """
        # obtençao do passo dos ticks do axis
        # eixo x
        step_x_tickloc = self.axes.get_xaxis().get_majorticklocs()[1] - self.axes.get_xaxis().get_majorticklocs()[0]
        # eixo y
        step_y_tickloc = self.axes.get_yaxis().get_majorticklocs()[1] - self.axes.get_yaxis().get_majorticklocs()[0]

        return step_x_tickloc, step_y_tickloc


    def set_limites(self, x_lim, y_lim):
        u"""
        Método para atualizar os limites do axes. Avalia se os novos limites definidos não omitem informações

        x_lim: limites de x
        y_lim: limites de y
        """
        xlim_original = self.axes.get_xlim()
        ylim_original = self.axes.get_ylim()

        self.axes.set_xlim(min(xlim_original[0], x_lim[0]), max(xlim_original[1], x_lim[1]))
        self.axes.set_ylim(min(ylim_original[0], y_lim[0]), max(ylim_original[1], y_lim[1]))

    def set_label(self, label_x, label_y, **kwargs):
        u"""
        Método para adição de label aos eixos

        :param label_x (string): label para eixo x
        :param label_y (string): label para eixo y
        :param kwargs: keywords a serem passadas para matplotlib.pyplot.xlabel e ylabel
        """
        self.axes.set_xlabel(label_x, **kwargs)
        self.axes.set_ylabel(label_y, **kwargs)

    def set_legenda(self, legenda, **kwargs):
        u"""
        Método para adição de legenda

        :param legenda (list): lista com as strings para as legendas
        :param kwargs: keyword arguments a serem passados para matplotlib.pyplot.legend
        """
        # validação
        if not isinstance(legenda, list):
            raise TypeError('A legenda deve ser uma lista.')

        if len(legenda) != len(self.lista_graficos):
            raise ValueError(
                'A legenda deve ser uma lista do mesmo tamanho dos gráficos incluídos: {}'.format(len(self.lista_graficos)))

        self.axes.legend(self.lista_graficos, legenda, **kwargs)

    def grafico_dispersao_sem_incerteza(self, x, y, label_x = None, label_y = None,
                                        add_legenda = False, corrigir_limites = True, config_axes = True,
                                        **kwargs):
        u"""
        Subrotina para gerar gráfico de dispersão de variáveis y em função de x

        ========
        Entradas
        ========
        :param x (array): dados de x
        :param y (array): dados de y
        :param label_x (string): label de x (para eixo)
        :param label_y (string): label de y (para eixo)

        :param add_legenda (bool): adiciona o gráfico no atributo self.lista_graficos (para formatação de legenda)
        :param corrigir_limites (bool): corrige os limites dos gráficos (evita que pontos fiquem muito próximos ao
                                 box do gráfico).
        :param config_axes (bool): executa o método self.config_axes

        :param **kwargs: keyword argumentos a serem passados para método self.matplotlib.pyplot.plot

        """
        #if isinstance()
        # Organizando os vetores
        y = y[argsort(x)]
        x = sort(x)

        # plot
        dispersao_sem_incerteza, = self.axes.plot(x, y, **kwargs)

        # Labels
        if label_x is not None and label_y is not None:
            self.set_label(label_x, label_y, fontsize=16)

        # Modificação do limite dos gráficos
        if corrigir_limites:

            step_x_tickloc, step_y_tickloc = self.get_step_tick()
            xmin = min(x) - step_x_tickloc / 4.
            xmax = max(x) + step_x_tickloc / 4.
            ymin = min(y) - step_y_tickloc / 4.
            ymax = max(y) + step_y_tickloc / 4.
            self.set_limites((xmin, xmax), (ymin, ymax))

        # configuração do axes
        if config_axes:
            self.config_axes()

        # configuração para legenda
        if add_legenda:
            self.lista_graficos.append(dispersao_sem_incerteza)

    def grafico_dispersao_com_incerteza(self, x, y, ux, uy, label_x = None, label_y = None,
                                        fator_abrangencia_x = 2, fator_abrangencia_y = 2,
                                        add_legenda = False, corrigir_limites = True, config_axes = True, **kwargs):
        u"""
        ========
        Entradas
        ========
        :param x (array): dados de x
        :param y (array): dados de y
        :param label_x (string): label do eixo x
        :param label_y (string): label do eixo y
        :param fator_abrangencia: fator de abrangencia

        :param add_legenda (bool): adiciona o gráfico no atributo self.lista_graficos (para formatação de legenda)
        :param corrigir_limites (bool): corrige os limites dos gráficos (evita que pontos fiquem muito próximos ao
                                 box do gráfico).
        :param config_axes (bool): executa o método self.config_axes

        :param **kwargs: keyword argumentos a serem passados para método self.matplotlib.pyplot.errorbar
        """
        # organizando os vetores
        y = y[argsort(x)]
        ux = ux[argsort(x)]
        uy = uy[argsort(x)]
        x = sort(x)

        # incerteza expandida
        xerr = fator_abrangencia_x*ux
        yerr = fator_abrangencia_y*uy

        # gráfico
        dispersao_com_incerteza = self.axes.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)

        # Labels
        if label_x is not None and label_y is not None:
            self.set_label(label_x, label_y, fontsize=16)

        # Modificação do limite dos gráficos
        if corrigir_limites:
            step_x_tickloc, step_y_tickloc = self.get_step_tick()
            xmin = min(x-xerr) - step_x_tickloc / 4.
            xmax = max(x+xerr) + step_x_tickloc / 4.
            ymin = min(y-yerr) - step_y_tickloc / 4.
            ymax = max(y+yerr) + step_y_tickloc / 4.
            self.set_limites((xmin, xmax), (ymin, ymax))

        # Configuração dos axes
        if config_axes:
            self.config_axes()

        # Configuração para legenda
        if add_legenda:
                self.lista_graficos.append(dispersao_com_incerteza)

    def elipse_covariancia(self, cov, pos, c2=2, add_legenda=True):#, **kwargs):
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

        vals, vecs = eigsorted(cov)
        theta = degrees(arctan2(*vecs[:,0][::-1]))
        # Width and height are "full" widths, not radius
        width, height = 2 * sqrt(c2*vals)
        # ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)
        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=False, color='r', linewidth=2.0, zorder=2)

        self.axes.add_artist(ellip)

        # PLOT do centro
        self.axes.plot(pos[0],pos[1],'r*',markersize=10.0,zorder=2)

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

        #PLOT dos pontos extremos da elipse (correção da escala)
        self.axes.plot(coordenadas_x, coordenadas_y, '.r', markersize=0.01)

        if add_legenda:
            self.lista_graficos.append(ellip)

    def salvar_e_fechar(self, titulo, ajustar=True, config_axes=False, reiniciar=True):
        u"""
        Método para salvar o gráfico e fechar a janela

        :param titulo (string): nome do arquivo
        :param ajustar (bool): indica que o janela do gráfico deve ser cortada
        :param config_axes (bool): indica se o método self.config_axes deve ser usado
        :param reiniciar (bool): indica se o método self.reiniciar deve ser executado
        """

        # ajusta a área do gráfico
        if ajustar:
            self.fig_instance.subplots_adjust(right=0.95, top=0.95)

        # configuração ao axez
        if config_axes:
            self.config_axes()

        # salva
        self.fig_instance.savefig(titulo)
        #fecha
        close(self.fig_instance)

        # reinicia
        if reiniciar:
            self.reiniciar()

    def reiniciar(self):
        u"""
        Método para apagar o que fora plotado no axes, figura
        """
        clf()
        self.axes.clear()
        self.lista_graficos = []