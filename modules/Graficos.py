# -*- coding: utf-8 -*-
"""
Arquivo que contém a classe gráficos

@author: Daniel
"""

# Importação de pacotes

from numpy import arctan2, degrees, sqrt, sort, argsort, mean, std, nan, amin, amax

from numpy.linalg import eigh, inv

from matplotlib.pyplot import figure, close, clf
import matplotlib.ticker

from matplotlib.patches import Ellipse

from scipy.stats import probplot

from subrotinas import eval_cov_ellipse

# Definição da classe

class Grafico:
    def __init__(self, **kwargs):
        """
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

    def config_axes(self, tick=True, formato_cientifico=True, grid=True):
        u"""
        Método voltado à formatação de self.axes

        :param: tick (bool): define se o tick será alterado
        :param: formato_cientifico (bool): define se os eixos estarão em formato científico (sem offset e com potência de 10)
        :param: grid (bool): define se haverá adição de grid

        Formatações: configuração dos ticks, eixos em notação científica, offset dos eixos, e grid.
        """
        if tick:
            # Formato dos ticks
            self.axes.tick_params(reset=True, axis='both', right=False, top=False, direction='out', labelsize=14)

        if formato_cientifico:
            if isinstance(self.axes.get_xaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                self.axes.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2),
                                     useMathText=True, useOffset=False)

            if isinstance(self.axes.get_yaxis().get_major_formatter(), matplotlib.ticker.ScalarFormatter):
                self.axes.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2),
                                     useMathText=True, useOffset=False)



        if grid:
            # Definindo linhas de grade no major axes
            self.axes.grid(b=True, which='major', axis='both', linestyle='dashed', color='gray', zorder=1)

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

        x_lim (array) : limites de x
        y_lim (array): limites de y
        """
        xlim_original = self.axes.get_xlim()
        ylim_original = self.axes.get_ylim()
        self.axes.set_xlim(min(xlim_original[0], x_lim[0]), max(xlim_original[1], x_lim[1]))
        self.axes.set_ylim(min(ylim_original[0], y_lim[0]), max(ylim_original[1], y_lim[1]))

    def set_label(self, label_x=None, label_y=None,fontsize=14,**kwargs):
        u"""
        Método para adição de label aos eixos

        :param label_x (string): label para eixo x
        :param label_y (string): label para eixo y
        :param kwargs: keywords a serem passadas para matplotlib.pyplot.xlabel e ylabel
        """
        if label_x is not None:
            self.axes.set_xlabel(label_x,fontsize=fontsize,**kwargs)
        if label_y is not None:
            self.axes.set_ylabel(label_y,fontsize=fontsize,**kwargs)

    def set_legenda(self,legenda,fontsize=12,loc='best',frameon=True,fancybox=True,**kwargs):
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

        self.axes.legend(self.lista_graficos, legenda,fontsize=fontsize,loc=loc,frameon=frameon,fancybox=fancybox, **kwargs)

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
                                 box do gráfico). Usa self.set_limites
        :param config_axes (bool): executa o método self.config_axes

        :param **kwargs: keyword argumentos a serem passados para método self.matplotlib.pyplot.plot

        """
        # Organizando os vetores
        y = y[argsort(x)]
        x = sort(x)

        # plot
        dispersao_sem_incerteza, = self.axes.plot(x, y, **kwargs)

        # Labels
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
                                        fator_abrangencia_x = None, fator_abrangencia_y = None,
                                        add_legenda = False, corrigir_limites = True, config_axes = True, **kwargs):
        u"""
        ========
        Entradas
        ========
        :param x (array): dados de x
        :param y (array): dados de y
        :param ux (array ou None): incerteza de x
        :param uy (array ou None): incerteza de y
        :param label_x (string): label do eixo x
        :param label_y (string): label do eixo y
        :param fator_abrangencia: fator de abrangencia

        :param add_legenda (bool): adiciona o gráfico no atributo self.lista_graficos (para formatação de legenda)
        :param corrigir_limites (bool): corrige os limites dos gráficos (evita que pontos fiquem muito próximos ao
                                 box do gráfico). Usa self.set_limites
        :param config_axes (bool): executa o método self.config_axes

        :param **kwargs: keyword argumentos a serem passados para método self.matplotlib.pyplot.errorbar
        """
        if not isinstance(fator_abrangencia_x, list) or not isinstance(fator_abrangencia_y, list):
            raise TypeError('The coverage factors must be informed in a form of lists.')

        # organizando os vetores
        y = y[argsort(x)]
        if ux is not None:
            ux = ux[argsort(x)]
        if uy is not None:
            uy = uy[argsort(x)]
        x = sort(x)

        # incerteza expandida
        if ux is not None:
            xerr = [fator_abrangencia_x[i] * ux[i] for i in range(len(ux))]
        else:
            xerr = None
        if uy is not None:
            yerr = [fator_abrangencia_y[i]*uy[i] for i in range(len(uy))]
        else:
            yerr = None

        # gráfico
        dispersao_com_incerteza = self.axes.errorbar(x, y, xerr=xerr, yerr=yerr, **kwargs)

        # Labels
        self.set_label(label_x, label_y, fontsize=16)

        # Modificação do limite dos gráficos
        if corrigir_limites:
            step_x_tickloc, step_y_tickloc = self.get_step_tick()
            if ux is not None:
                xmin = min(x-xerr) - step_x_tickloc / 4.
                xmax = max(x+xerr) + step_x_tickloc / 4.
            else:
                xmin = min(x) - step_x_tickloc / 4.
                xmax = max(x) + step_x_tickloc / 4.
            if uy is not None:
                ymin = min(y-yerr) - step_y_tickloc / 4.
                ymax = max(y+yerr) + step_y_tickloc / 4.
            else:
                ymin = min(y) - step_y_tickloc / 4.
                ymax = max(y) + step_y_tickloc / 4.
            self.set_limites((xmin, xmax), (ymin, ymax))

        # Configuração dos axes
        if config_axes:
            self.config_axes()

        # Configuração para legenda
        if add_legenda:
                self.lista_graficos.append(dispersao_com_incerteza)


    def boxplot(self, x, label_x = None, label_y=None, config_axes=True, **kwargs):
        u"""
        Gráfico boxplot
        :param x: conjunto de dados
        :param label_x: label do eixo x (label dos ticks)
        :param label_y: label do eixo y

        kwargs: keyword arguments a serem passados para o boxplot
        """

        self.axes.boxplot(x, **kwargs)

        self.axes.set_xticklabels(label_x)

        self.set_label(None, label_y, fontsize=16)

        if config_axes:
            self.config_axes()

    def autocorr(self, x, label_x = None, label_y=None, config_axes = True, corrigir_limites=True, **kwargs):
        u"""
        Gráficos de autocorrelação.

        :param x: dados para o gráfico
        :param label_x ('string): label para o eixo x
        :param label_y'('string'): label para o eixo y
        :param config_axes (bool): executa self.config_axes
        :param corrigir_limites (bool): corrige os limites do gráfico. (x começar em zero)
        :param kwargs: keyword arguments para matplotlib.pyplot.acorr
        """

        self.axes.acorr(x, **kwargs)

        self.set_label(label_x, label_y)

        self.axes.axhline(0, color='black', lw=2)

        if corrigir_limites:
            self.axes.set_xlim(0, x.shape[0])

        if config_axes:
            self.config_axes(formato_cientifico=False)

    def histograma(self, x, label_x = None, label_y = None, config_axes=True, **kwargs):
        u"""
        Histograma

        :param x (array): dados para o gráfico
        :param label_x ('string): label para o eixo x
        :param label_y ('string):: label para o eixo y
        :param config_axes (bool): executa self.config_axes
        :param kwargs: keyword arguments para matplotlib.pyplot.hist
        """

        self.axes.hist(x, **kwargs)

        self.set_label(label_x, label_y)

        if config_axes:
            self.config_axes()

    def probplot(self, x, label_y, config_axes=True, **kwargs):
        u"""
        Gráfico de probabilidade para normal (teste)

        :param x (array): dados
        :param label_y: label para eixo y
        :param config_axes (bool): executa método self.config_axes
        :param kwargs: kwargs para plot
        """

        res = probplot(x, dist='norm', sparams=(mean(x), std(x, ddof=1)))

        if not (nan in res[0][0].tolist() or nan in res[0][1].tolist() or nan in res[1]):
            self.axes.plot(res[0][0], res[0][1], 'o', res[0][0], res[1][0] * res[0][0] + res[1][1], **kwargs)

        self.set_label('Quantis', label_y=label_y)

        xmin = amin(res[0][0])
        xmax = amax(res[0][0])
        ymin = amin(x)
        ymax = amax(x)
        posx = xmin + 0.70 * (xmax - xmin)
        posy = ymin + 0.01 * (ymax - ymin)
        self.axes.text(posx, posy, '$R^2$={:1.4f}'.format(res[1][2]))

        if config_axes:
            self.config_axes()

    def elipse_covariancia(self,cov,pos,ellipseComparacao,add_legenda=True):
        """
        Plots an `nstd` sigma error ellipse based on the specified covariance
        matrix (`cov`). Additional keyword arguments are passed on to the
        ellipse patch artist.

        Parameters
        ----------
            cov : The 2x2 covariance matrix to base the ellipse on
            pos : The location of the center of the ellipse. Expects a 2-element
                sequence of [x0, y0].
            ellipseComparacao : objective function limit value used in the comparison to select the pairs that will be part of the region
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

        coordenadas_x, coordenadas_y, width, height, theta = eval_cov_ellipse(cov,pos,ellipseComparacao)

        ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, fill=False, color='r', linewidth=2.0, zorder=2)

        self.axes.add_artist(ellip)

        # PLOT do centro
        self.axes.plot(pos[0],pos[1],'r*',markersize=10.0,zorder=2)

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
        self.axes.tick_params(reset=True)
        self.lista_graficos = []