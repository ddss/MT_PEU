from MT_PEU import EstimacaoNaoLinear
import pytest
from casadi import MX, vertcat,exp
from numpy import array

def Modelo(param,x,*args):

    ko, E = param[0], param[1]
    tempo, T = x[:,0], x[:,1]

    return exp(-(ko*10**17)*tempo*exp(-E/T))

y = [0.9,0.949,0.886,0.785,0.791,0.890,0.787,0.877,0.938,
0.782,0.827,0.696,0.582,0.795,0.800,0.790,0.883,0.712,0.576,0.715,0.673,
0.802,0.802,0.804,0.794,0.804,0.799,0.764,0.688,0.717,0.802,0.695,0.808,
0.655,0.309,0.689,0.437,0.425,0.638,.659,0.449]; tempo = [120.0,60.0,60.0,120.0,120.0,60.0,60.0,30.0,15.0,60.0,
45.1,90.0,150.0,60.0,60.0,60.0,30.0,90.0,150.0,90.4,120.0,
60.0,60.0,60.0,60.0,60.0,60.0,30.0,45.1,30.0,30.0,45.0,15.0,30.0,90.0,25.0,
60.1,60.0,30.0,30.0,60.0] ;temperatura = [600.0,600.0,612.0,612.0,612.0,612.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,620.0,
620.0,620.0,620.0,620.0,620.0,620.0,631.0,631.0,631.0,631.0,631.0,639.0,639.0,
639.0,639.0,639.0,639.0,639.0,639.0,639.0]

uy = [1]*41; uxtempo = [1]*41; uxtemperatura = [1]*41

#Execução do MT_PEU
Estime = EstimacaoNaoLinear(Modelo, simbolos_x=['t','T'], simbolos_y=['y'], simbolos_param=['ko','E'], Folder='Exemplo1')
Estime.setDados(0, (tempo, uxtempo), (temperatura, uxtemperatura))
Estime.setDados(1, (y, uy))
Estime.setConjunto(tipo='estimacao')
Estime.optimize(initial_estimative=[0.5, 25000], algoritmo='ipopt')
Estime.incertezaParametros(metodoIncerteza='Geral''')
Estime.predicao()

# Valores originais
hessian = array([[5.46625676e+00, -7.50116238e-03], [-7.50116238e-03,  1.02960714e-05]])
sensibilidade = array([[-1.06335451e-01,  1.52826132e-04],
       [-5.59353653e-02,  8.03907392e-05],
       [-1.28132438e-01,  1.80542088e-04],
       [-2.26100338e-01,  3.18581523e-04],
       [-2.26100338e-01,  3.18581523e-04],
       [-1.28132438e-01,  1.80542088e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-1.16257751e-01,  1.61696650e-04],
       [-6.14815711e-02,  8.55114088e-05],
       [-2.07847997e-01,  2.89084593e-04],
       [-1.65181656e-01,  2.29742276e-04],
       [-2.78696190e-01,  3.87623533e-04],
       [-3.71165533e-01,  5.16234166e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-1.16257751e-01,  1.61696650e-04],
       [-2.78696190e-01,  3.87623533e-04],
       [-3.71165533e-01,  5.16234166e-04],
       [-2.79516557e-01,  3.88764537e-04],
       [-3.32172517e-01,  4.62000879e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.07847997e-01,  2.89084593e-04],
       [-2.21676064e-01,  3.02942522e-04],
       [-2.94742115e-01,  4.02794591e-04],
       [-2.21676064e-01,  3.02942522e-04],
       [-2.21676064e-01,  3.02942522e-04],
       [-2.94327851e-01,  4.02228457e-04],
       [-1.98222463e-01,  2.67499395e-04],
       [-3.20995753e-01,  4.33180823e-04],
       [-4.13891237e-01,  5.58542426e-04],
       [-2.86998240e-01,  3.87301491e-04],
       [-4.20992273e-01,  5.68125210e-04],
       [-4.20883720e-01,  5.67978718e-04],
       [-3.20995753e-01,  4.33180823e-04],
       [-3.20995753e-01,  4.33180823e-04],
       [-4.20883720e-01,  5.67978718e-04]])

# Dados de teste
testdata_H = [(Modelo, ['t','T'], ['y'], ['ko','E'],'Exemplo1',y,tempo,temperatura,uy,uxtempo,uxtemperatura,hessian)]
testdata_S = [(Modelo, ['t','T'], ['y'], ['ko','E'],'Exemplo1',y,tempo,temperatura,uy,uxtempo,uxtemperatura,sensibilidade)]

#Hessiana
@pytest.mark.parametrize("Modelo, simbolos_x, simbolos_y, simbolos_param, Folder, y, tempo, temperatura, uy, uxtempo, uxtemperatura, H",testdata_H)
def test_hessian(Modelo, simbolos_x, simbolos_y, simbolos_param, Folder, y, tempo, temperatura, uy, uxtempo, uxtemperatura,H):
    assert round(Estime.Hessiana.mean(),5) == round(H.mean(),5)

# Sensibilidade
@pytest.mark.parametrize("Modelo, simbolos_x, simbolos_y, simbolos_param, Folder, y, tempo, temperatura, uy, uxtempo, uxtemperatura, S",testdata_S)
def test_S(Modelo, simbolos_x, simbolos_y, simbolos_param, Folder, y, tempo, temperatura, uy, uxtempo,uxtemperatura, S):
    assert round(Estime.S.mean(), 5) == round(S.mean(), 5)



