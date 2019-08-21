from matplotlib import use
use('Agg')
from MT_PEU import EstimacaoNaoLinear
from casadi import vertcat

def Modelo(param,x):

    a1, b1, a2, b2 = param[0], param[1], param[2], param[3]
    x1, x2 = x[0], x[1]

    return vertcat(a1*x1/(1+b1*x1), a2*(x2**b2))

Estime = EstimacaoNaoLinear(Modelo,simbolos_x=['x1','x2'],simbolos_y=['y1','y2'],simbolos_param=['alpha1','alpha2', 'beta1', 'beta2'],
                          label_latex_param=[r'$\alpha_1$',r'$\alpha_2$',r'$\beta_1$',r'$\beta_2$'],unidades_y=['kg','kg'],projeto='projeto')


x1 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y1 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]
x2 = [1.,2.,3.,5.,10,15.,20.,30.,40.,50.]
y2 = [1.66,6.07,7.55,9.72,15.24,18.79,19.33,22.38,24.27,25.51]

ux1 = [1]*10
ux2 = [1]*10
uy1 = [1]*10
uy2 = [1]*10

Estime.setDados(0,(x1,ux1),(x2,ux2))
Estime.setDados(1, (y1, uy1), (y2, uy2))
Estime.setConjunto(tipo='estimacao')
Estime.otimiza_cas(Estimativa_inicial=[3,0.1,5,0.4])


