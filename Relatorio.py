# -*- coding: utf-8 -*-
"""
Classe auxiliar para escrita de Relatórios

@author(es): Daniel, Francisco,
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""
# ---------------------------------------------------------------------
# IMPORTAÇÃO DE PACOTES DE TERCEIROS
# ---------------------------------------------------------------------
from os import getcwd, sep
from subrotinas import Validacao_Diretorio

# ---------------------------------------------------------------------
# CLASSES
# ---------------------------------------------------------------------
class Relatorio:

    def __init__(self,base_path=None,base_dir=None):
        '''
        Classe para escrita de relatórios sobre estimação de parãmetros

        ========
        Entradas
        ========

        * base_path: caminho base
        * base_dir: diretório no caminho base que os arquivos serão salvos
        '''
        # TODO: permitir fluxos
        if base_path is None:
            base_path = getcwd()

        if base_dir is None:
            base_dir = sep + 'Relatorio' + sep

        if base_path is not None:
            Validacao_Diretorio(base_path,base_dir)

        self.__base_path = base_path + base_dir

    def Parametros(self,parametros,pontoOtimo):
        '''
        Escrita sobre a etapa a estimativa dos parâmetros e sua incerteza

        =======
        Entrada
        =======
        * parametros: objeto Grandeza que contenha os atributos dos parâmetros
        * pontoOtimo: valor da função objetivo no ponto ótimo

        ==========
        Referência
        ==========
        [1] https://docs.python.org/2/tutorial/inputoutput.html
        [2] https://docs.python.org/2/library/string.html#formatstrings
        '''
        with open(self.__base_path+'relatorio-parametros.txt','wb') as f:
            # Criação do título: o tamanho dele será o máximo entre 65 e 18*NP (Apenas por estética)
            f.write(('{:#^'+str(max([70,parametros.NV*18]))+'}\n').format('PARÂMETROS'))

            # Estimativa dos parâmetros
            f.write(('Simbolos   : '+ '{:^10} '*parametros.NV).format(*parametros.simbolos)    + '\n')
            f.write(('Estimativa : '+ '{:^10.3e} '*parametros.NV).format(*parametros.estimativa) + '\n')

            if parametros.matriz_covariancia is not None:
                # Matriz de covariância, incerteza e matriz de correlação
                f.write(('Variância  : '+ '{:^10.3e} '*parametros.NV).format(*[parametros.matriz_covariancia[i,i] for i in xrange(parametros.NV)]) + '\n')
                f.write(('Incerteza  : '+ '{:^10.3e} '*parametros.NV).format(*[parametros.matriz_incerteza[i,0] for i in xrange(parametros.NV)]) + '\n')
                f.write('\n')
                f.write('Matriz de covariância:\n')
                f.write(str(parametros.matriz_covariancia)+'\n')
                f.write('\n')
                f.write('Matriz de correlação:\n')
                f.write(str(parametros.matriz_correlacao)+'\n')
            else:
                f.write('Variância : não avaliada + \n')
                f.write('Incerteza : não avaliada + \n')
                f.write('FObj ótima : '+ '{:.3e} '.format(pontoOtimo)+'- {:<} \n'.format('Valor da função objetivo no ponto ótimo'))
                f.write('\n')
                f.write('Matriz de covariância: não avaliada')
                f.write('\n')
                f.write('Matriz de correlação: não avaliada')

            f.write('\n')
            # Valor da função objetivo no ponto ótimo
            f.write( 'FObj ótima : '+ '{:.3e} '.format(pontoOtimo)+'- {:<} \n'.format('Valor da função objetivo no ponto ótimo'))
        f.close()

    def Predicao(self,x,y,R2,R2ajustado,**kwargs):
        '''
        Escrita sobre a etapa de Predição de análise de resíduos

        =======
        Entrada
        =======
        * x: objeto grandeza das variáveis independentes
        * y: objeto grandeza das variáveis dependentes
        * R2: lista com os valores de R2
        * R2ajustado: lista com os valores de R2 ajustado

        ========
        Keywargs
        ========
        export_y     (bool): exporta os dados calculados de y e sua incerteza em um txt com separação por vírgula
        export_cov_y (bool): exporta a matriz de covariância de y

        export_x     (bool): exporta os dados calculados de x e sua incerteza em um txt com separação  por vírgula
        export_cov_x (bool): exporta a matriz de covariância de x
        ==========
        Referência
        ==========
        [1] https://docs.python.org/2/tutorial/inputoutput.html
        [2] https://docs.python.org/2/library/string.html#formatstrings
        '''

        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if not isinstance(kwargs.get('export_y'),bool) and kwargs.get('export_y') is not None:
            raise TypeError('A keyword export_y deve ser booleana')

        if not isinstance(kwargs.get('export_cov_y'),bool) and kwargs.get('export_cov_y') is not None:
            raise TypeError('A keyword export_cov_y deve ser booleana')

        if kwargs.get('export_y') is None:
            export_y = False
        else:
            export_y = kwargs.get('export_y')

        if kwargs.get('export_cov_y') is None:
            export_cov_y = False
        else:
            export_cov_y = kwargs.get('export_cov_y')
        # ---------------------------------------------------------------------
        # ESCRITA DE ARQUIVO DE RELATÓRIO
        # ---------------------------------------------------------------------
        with open(self.__base_path+'relatorio-predicao.txt','wb') as f:
            f.write(('{:#^'+str(max([70,y.NV*18]))+'}\n').format('PREDIÇÃO'))
            f.write(('{:-^'+str(max([70,y.NV*18]))+'}\n').format('GRANDEZAS DEPENDENTES'))
            f.write('Adequação aos dados experimentais:\n')
            if R2 is not None and R2ajustado is not None:
                # Operador para formatar os valores de R2 e R2ajustados, porque eles são dicionários
                construtor_formatacao_simbolos = ['{'+str(symb)+':^8.3f} ' for symb in y.simbolos]
                f.write('Simbolos                             : '+ ('{:^8}'*y.NV).format(*y.simbolos) + '\n')
                f.write('Coeficiente de determinação          : '+ (''.join(construtor_formatacao_simbolos)).format(**R2) + '\n')
                f.write('Coeficiente de determinação ajustado : '+ (''.join(construtor_formatacao_simbolos)).format(**R2ajustado) + '\n')
            f.write('\n')
            # TODO: Adpatar relatório para as novas características dos testes estatísticos das grandezas
            # # Análise de resíduos - testes para normalidade
            # f.write('Análise de resíduos: testes para normalidade \n')
            # f.write('{:<10} :'.format('Simbolos')+ ('{:^14}'*y.NV).format(*y.simbolos) + '\n')
            # # construção semi-automatizada para preencher os valores dos testes estatísticos de normalidade
            # for teste in y._Grandeza__nomesTestes['residuo-Normalidade']:
            #     f.write('{:<10} : '.format(teste))
            #     for symb in y.simbolos:
            #         if isinstance(y.estatisticas[symb]['residuo-Normalidade'][teste],float):
            #             f.write('{:^14.3f}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste])+' ')
            #         elif isinstance(y.estatisticas[symb]['residuo-Normalidade'][teste],list):
            #             f.write(('{:^4.3f}  '*len(y.estatisticas[symb]['residuo-Normalidade'][teste])).format(*y.estatisticas[symb]['residuo-Normalidade'][teste])+' ')
            #         else:
            #             f.write('{:^14}'.format('N/A'))
            #     f.write('\n')
            # f.write('\n')
            #
            # # Análise de resíduos - testes para média
            # f.write('Análise de resíduos: testes para média \n')
            # f.write('{:<10} :'.format('Simbolos') + ('{:^8}'*y.NV).format(*y.simbolos) + '\n')
            # # construção semi-automatizada para preencher os valores dos testes estatísticos para média
            # for teste in y._Grandeza__nomesTestes['residuo-Media']:
            #     f.write('{:<10} : '.format(teste))
            #     for symb in y.simbolos:
            #         if isinstance(y.estatisticas[symb]['residuo-Media'][teste],float):
            #             f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Media'][teste])+' ')
            #         elif isinstance(y.estatisticas[symb]['residuo-Media'][teste],list):
            #             f.write(('{:^4.3f}  '*len(y.estatisticas[symb]['residuo-Media'][teste])).format(*y.estatisticas[symb]['residuo-Media'][teste])+' ')
            #         else:
            #             f.write('N/A')
            #     f.write('\n')
        f.close()

        # ---------------------------------------------------------------------
        # EXPORTAÇÃO DA PREDIÇÃO
        # ---------------------------------------------------------------------
        # Valores calculados e incerteza
        if export_y:
            cont = 0
            for symb in y.simbolos:
                with open(self.__base_path+symb+'-calculado-predicao.txt','wb') as f:
                    for i in xrange(y.calculado.NE):
                        f.write('{:.5f},{:.5f}\n'.format(y.calculado.matriz_estimativa[i,cont],y.calculado.matriz_incerteza[i,cont]))
                f.close()
                cont+=1

        # matriz de covariância
        if export_cov_y:
            with open(self.__base_path+'y-calculado-matriz-covariancia.txt','wb') as f:
                for i in xrange(y.NV*y.calculado.NE):
                    for j in xrange(y.NV*y.calculado.NE):
                        f.write('{:.5f} '.format(y.calculado.matriz_covariancia[i,j]))
                    f.write('\n')
            f.close()