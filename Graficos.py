# -*- coding: utf-8 -*-
"""
Arquivo que contém a classe gráficos

@author: Daniel
"""

# Importação de pacotes

from numpy import arctan2, degrees, sqrt

from numpy.linalg import eigh, inv

from matplotlib.pyplot import figure, close, clf

from matplotlib.patches import Ellipse

class Grafico:
    def __init__(self, **kwargs):
        u"""

        :param kwargs: mesmas kwargs de figure matplotlib
        """

        self.fig_instance = figure(**kwargs)
        self.axes = self.fig_instance.add_subplot(1, 1, 1)

        self.lista_graficos = []

    def config_axis(self, offset_x=False, offset_y=False):
        u"""
        Método voltada à formatação do axis de gráficos

        ax: matplotlib.axis.axes
        """
        # axis
        self.axes.get_xaxis().tick_bottom() # incluir box no fundo
        self.axes.get_yaxis().tick_left()   # incluir box na esquerda

        # Formato dos ticks
        self.axes.tick_params(axis='both', direction='out', labelsize=14, pad=-2)

        # Definindo notação científica para os eixos
        self.axes.get_xaxis().get_major_formatter().set_powerlimits((-2, 2))
        self.axes.get_yaxis().get_major_formatter().set_powerlimits((-2, 2))

        # Definindo que offset nos eixos
        self.axes.get_xaxis().get_major_formatter().set_useOffset(offset_x)
        self.axes.get_yaxis().get_major_formatter().set_useOffset(offset_y)

        # Definindo linhas de grade no major axes
        self.axes.grid(b='on', which='major', axis='both', linestyle='dashed', color='gray')

    def get_step_tick(self):
        # obtençao do passo dos ticks do axis
        # eixo x
        step_x_tickloc = abs(
            self.axes.get_xaxis().get_majorticklocs()[1] - self.axes.get_xaxis().get_majorticklocs()[0])
        # eixo y
        step_y_tickloc = abs(
            self.axes.get_yaxis().get_majorticklocs()[1] - self.axes.get_yaxis().get_majorticklocs()[0])

        return step_x_tickloc, step_y_tickloc

    def grafico_dispersao_sem_incerteza(self, x, y, label_x=None, label_y=None,
                                        add_legenda=False, corrigir_limites=True, config_axes = True,
                                        **kwargs):
        u"""
        Subrotina para gerar gráficos das variáveis y em função de x

        =======
        Entrada
        =======
        x (array): dados de x
        y (array): dados de y

        label_x: label de x (para eixo)
        label_y: label de y (para eico)

        kwargs para o plot
        """
        # plot
        dispersao_sem_incerteza, = self.axes.plot(x, y, **kwargs)

        # Labels
        if label_x is not None and label_y is not None:
            self.set_label(label_x, label_y, fontsize=16)

        if config_axes:
           self.config_axis(offset_x=False, offset_y=False)

        # Modificação do limite dos gráficos
        if corrigir_limites:

            step_x_tickloc, step_y_tickloc = self.get_step_tick()
            xmin = min(x) - step_x_tickloc / 2.
            xmax = max(x) + step_x_tickloc / 2.
            ymin = min(y) - step_y_tickloc / 2.
            ymax = max(y) + step_y_tickloc / 2.
            self.axes.set_xlim(xmin, xmax)
            self.axes.set_ylim(ymin, ymax)

        if add_legenda:
            self.lista_graficos.append(dispersao_sem_incerteza)

    def grafico_dispersao_com_incerteza(self, x, y, ux, uy, label_x = None, label_y = None,
                                        fator_abrangencia = 2,
                                        add_legenda = False, corrigir_limites = True, config_axes = True):
        """
        ========
        Entradas
        ========
        :param x (array): dados de x
        :param y (array): dados de y
        :param label_x (string): label do eixo x
        :param label_y (string):
        :param fator_abrangencia: fator de abrangencia

        # todo o trabalho é realizado no atributo self.axes
        """
        # Grafico com os pontos e as incertezas

        xerr = fator_abrangencia*ux
        yerr = fator_abrangencia*uy

        dispersao_com_incerteza = self.axes.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="o")

        # Labels
        if label_x is not None and label_y is not None:
            self.set_label(label_x, label_y, fontsize=16)

        if config_axes:
            self.config_axis(offset_x=False, offset_y=False)

        # Modificação do limite dos gráficos
        if corrigir_limites:
            step_x_tickloc, step_y_tickloc = self.get_step_tick()
            xmin = min(x-xerr) - step_x_tickloc / 4.
            xmax = max(x+xerr) + step_x_tickloc / 4.
            ymin = min(y-yerr) - step_y_tickloc / 4.
            ymax = max(y+yerr) + step_y_tickloc / 4.
            self.axes.set_xlim(xmin, xmax)
            self.axes.set_ylim(ymin, ymax)

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

    def set_label(self, label_x, label_y, **kwargs):
        self.axes.set_xlabel(label_x, **kwargs)
        self.axes.set_ylabel(label_y, **kwargs)

    def set_legenda(self, legenda, **kwargs):

        if len(legenda) != len(self.lista_graficos):

            raise ValueError('A legenda deve ser do mesmo tamanho dos gráficos incluídos {}'.format(len(self.lista_graficos)))

        self.axes.legend(self.lista_graficos, legenda, **kwargs)

    def salvar_e_fechar(self, titulo, ajustar=True, config_axes=False, reiniciar=True):

        if ajustar:
            self.fig_instance.subplots_adjust(left=0.125, right=0.95, top=0.95, bottom=0.125)

        if config_axes:
            self.config_axis()

        self.fig_instance.savefig(titulo)
        close(self.fig_instance)

        if reiniciar:
            self.reiniciar()

    def reiniciar(self):
        clf()
        self.axes.clear()
