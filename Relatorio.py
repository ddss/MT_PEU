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
            f.write(('Simbolos   : '+ '{:^10} '*parametros.NV).format(*parametros.simbolos) + '\n')
            f.write(('Estimativa : '+ '{:^10.3e} '*parametros.NV).format(*parametros.estimativa) + '\n')

            if parametros.matriz_covariancia is not None:
                # Matriz de covariância, incerteza e matriz de correlação
                f.write(('Variância  : '+ '{:^10.3e} '*parametros.NV).format(*[parametros.matriz_covariancia[i,i] for i in xrange(parametros.NV)]) + '\n')
                f.write(('Incerteza  : '+ '{:^10.3e} '*parametros.NV).format(*[parametros.matriz_incerteza[0,i] for i in xrange(parametros.NV)]) + '\n')
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
            f.write( 'FObj : '+ '{:.3f} '.format(pontoOtimo)+'- {:<} \n'.format('Valor da função objetivo no ponto ótimo'))
            f.write('\n')
            f.write(('{:-^'+str(max([70,parametros.NV*18]))+'}\n').format('RESTRIÇÕES'))
            f.write(('Simbolos        : '+ '{:^10} '*parametros.NV).format(*parametros.simbolos) + '\n')
            if parametros.limite_superior is not None:
                f.write(('Limite superior : '+ '{:^10.3e} '*parametros.NV).format(*parametros.limite_superior) + '\n')
            else:
                f.write(('Limite superior : '+ '{:^10} '*parametros.NV).format(*['N/A']*parametros.NV) + '\n')
            if parametros.limite_inferior is not None:
                f.write(('Limite inferior : '+ '{:^10.3e} '*parametros.NV).format(*parametros.limite_inferior) + '\n')
            else:
                f.write(('Limite inferior : '+ '{:^10} '*parametros.NV).format(*['N/A']*parametros.NV) + '\n')
            f.close()

    def Predicao(self,x,y,estatisticas,**kwargs):
        '''
        Escrita sobre a etapa de Predição de análise de resíduos

        =======
        Entrada
        =======
        * x: objeto grandeza das variáveis independentes
        * y: objeto grandeza das variáveis dependentes
        * estatisticas: dicionário com os valores de R2 (dicionário), R2ajustado (dicionário), FO (dicionário)

        ========
        Keywargs
        ========
        export_y     (bool): exporta os dados calculados de y, sua incerteza e graus de liberdade em um txt com separação por vírgula
        export_cov_y (bool): exporta a matriz de covariância de y

        export_x     (bool): exporta os dados calculados de x, sua incerteza e graus de liberdade em um txt com separação  por vírgula
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
        if estatisticas is not None:
            with open(self.__base_path+'relatorio-predicao.txt','wb') as f:
                # TÍTULO:
                f.write(('{:#^'+str(max([70,y.NV*18]))+'}\n').format(' PREDIÇÃO '))
                f.write(('{:=^'+str(max([70,y.NV*18]))+'}\n').format(' GRANDEZAS DEPENDENTES '))
                # R2:
                f.write('Coeficientes de correlação:\n')
                # Operador para formatar os valores de R2 e R2ajustados, porque eles são dicionários
                construtor_formatacao_simbolos = ['{'+str(symb)+':^8.3f} ' for symb in y.simbolos]
                f.write('    Simbolos                             : '+ ('{:^8}'*y.NV).format(*y.simbolos) + '\n')
                f.write('    Coeficiente de determinação          : '+ (''.join(construtor_formatacao_simbolos)).format(**estatisticas['R2']) + '\n')
                f.write('    Coeficiente de determinação ajustado : '+ (''.join(construtor_formatacao_simbolos)).format(**estatisticas['R2ajustado']) + '\n')
                f.write('\n')

                # Função objetivo
                f.write('Função objetivo:\n')
                f.write('    Info: a função objetivo deve estar entre chi2min e chi2max.\n')
                # Operador para formatar os valores de R2 e R2ajustados, porque eles são dicionários
                f.write('    chi2max: {:.3f}\n'.format(estatisticas['FuncaoObjetivo']['chi2max']))
                f.write('    FO     : {:.3f}\n'.format(estatisticas['FuncaoObjetivo']['FO']))
                f.write('    chi2min: {:.3f}\n'.format(estatisticas['FuncaoObjetivo']['chi2min']))
                f.write('\n')

                # ANÁLISE DE RESÍDUOS - testes para normalidade
                f.write('Análise de resíduos:\n')
                f.write('    Normalidade:\n')
                f.write('    {:-^45}\n'.format('Testes com p-valores'))
                f.write('    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)\n    para não rejeitar a hipótese nula (Ho).\n')
                f.write('\n')
                f.write('    {:<10} : '.format('Simbolos')+ ('{:^8}'*y.NV).format(*y.simbolos)+'\n')
                # construção semi-automatizada para preencher os valores dos testes estatísticos de normalidade
                for teste in y._Grandeza__nomesTestes['residuo-Normalidade'].keys():
                    break_line = False
                    if not isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                        f.write('    {:<10} : '.format(teste))
                    for symb in y.simbolos:
                        if isinstance(y.estatisticas[symb]['residuo-Normalidade'][teste],float):
                            f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste])+' ')
                            break_line = True
                        elif y.estatisticas[symb]['residuo-Normalidade'][teste] is None:
                            f.write('{:^8}'.format('N/A')+' ')
                            break_line = True
                    if not isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                        f.write('| Ho: {}'.format(y._Grandeza__TestesInfo['residuo-Normalidade'][teste]['H0']))
                    if break_line:
                        f.write('\n')
                f.write('\n')
                # f.write('    {:-^45}\n'.format('Testes com valores críticos'))
                # f.write('    {:<}:                '.format('Simbolos')+ ('{:^37}'*y.NV).format(*y.simbolos)+'\n')
                # for teste in y._Grandeza__nomesTestes['residuo-Normalidade'].keys():
                #     break_line = False
                #     if isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                #         f.write('        {:<}: \n'.format(teste))
                #         for key in y._Grandeza__nomesTestes['residuo-Normalidade'][teste].keys():
                #             f.write('            {:<16}:'.format(key))
                #             for symb in y.simbolos:
                #                 if isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste][key],float):
                #                     f.write('{:^37.3f}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste][key]))
                #                 else:
                #                     f.write('{:^37}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste][key]))
                #             f.write('\n')
                #         break_line = True
                #
                #     if break_line:
                #         f.write('\n')

                # ANÁLISE DE RESÍDUOS - testes para média
                f.write('    Média:\n')
                f.write('    {:-^45}\n'.format('Testes com p-valores'))
                f.write('    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)\n    para não rejeitar a hipótese nula (Ho).\n')
                f.write('\n')
                f.write('    {:<8}:'.format('Simbolos') + ('{:^8}'*y.NV).format(*y.simbolos) + '\n')
                # construção semi-automatizada para preencher os valores dos testes estatísticos para média
                for teste in y._Grandeza__nomesTestes['residuo-Media'].keys():
                     f.write('    {:<8}:'.format(teste))
                     for symb in y.simbolos:
                         if isinstance(y.estatisticas[symb]['residuo-Media'][teste],float):
                             f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Media'][teste])+' ')
                         else:
                             f.write('N/A')
                             f.write('\n')
               
                     if not isinstance(y._Grandeza__nomesTestes['residuo-Media'][teste],dict):
                        f.write('| Ho: {}'.format(y._Grandeza__TestesInfo['residuo-Media'][teste]['H0']))
                     if break_line:
                        f.write('\n')
                 
                 # ANÁLISE DE RESÍDUOS - testes para autocorrelação
                f.write('\n')
                f.write('    Autocorrelação:\n')
                f.write('    {:-^45}\n'.format('Testes com p-valores'))
                f.write('    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)\n    para não rejeitar a hipótese nula (Ho).\n')
                f.write('\n')
                f.write('    {:<}:                                 '.format('Simbolos')+ ('{:^8}'*y.NV).format(*y.simbolos)+'\n')
                for teste in y._Grandeza__nomesTestes['residuo-Autocorrelacao'].keys():
                    break_line = False
                    if teste == 'Ljung-Box':
                      if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                          f.write('    {:<}: \n'.format(teste))
                          for key in y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste].keys():
                              f.write('            {:<33}:'.format(key))
                              for symb in y.simbolos:
                                  if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste][key],float):
                                      f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Autocorrelacao'][teste][key]))
                                  else:
                                      f.write('N/A')
                              if  isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                                    f.write('| Ho: {}'.format(y._Grandeza__TestesInfo['residuo-Autocorrelacao'][teste][key]['H0']))
                              
                              f.write('\n')
                          break_line = True
                        
#                      
                    if break_line:
                        f.write('\n')
                             
                f.write('    {:-^45}\n'.format('Testes com estatística'))
                f.write('    Info: (i) Se a estatística do teste estiver próxima de 0 indica autocorrelação positiva\n          (ii) Se a estatística do teste estiver próxima de 4 indica autocorrelação negativa\n          (iii) Se a estatística do teste estiver próxima de 2 indica que não há autocorrelação.\n')
                f.write('\n')
                f.write('    {:<}:           '.format('Simbolos')+ ('{:^8}'*y.NV).format(*y.simbolos)+'\n')
                for teste in y._Grandeza__nomesTestes['residuo-Autocorrelacao'].keys():
                    break_line = False
                    if teste == 'Durbin Watson':
                      if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                          f.write('        {:<}: \n'.format(teste))
                          for key in y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste].keys():
                              f.write('            {:<11}:'.format(key))
                              for symb in y.simbolos:
                                  if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste][key],float):
                                      f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Autocorrelacao'][teste][key]))
                                  else:
                                      f.write('N/A')
                              f.write('\n')
                          break_line = True

                      if break_line:
                          f.write('\n')

                # ANÁLISE DE RESÍDUOS - testes para homocedasticidade
                f.write('    Homocedasticidade:\n')
                f.write('    {:-^45}\n'.format('Testes com p-valores'))
                f.write('    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)\n    para não rejeitar a hipótese nula (Ho).\n')
                f.write('\n')
                f.write('    {:<}:                                 '.format('Simbolos')+ ('{:^8}'*y.NV).format(*y.simbolos)+'\n')
                for teste in y._Grandeza__nomesTestes['residuo-Homocedasticidade'].keys():
                    break_line = False
                    if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste],dict):
                        f.write('    {:<}: \n'.format(teste))
                        for key in y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste].keys():
                            f.write('            {:<33}:'.format(key))
                            for symb in y.simbolos:
                                if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste][key],float):
                                    f.write('{:^8.3f}'.format(y.estatisticas[symb]['residuo-Homocedasticidade'][teste][key]))
                                else:
                                    f.write('N/A')
                            if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste],dict):
                                    f.write('| Ho: {}'.format(y._Grandeza__TestesInfo['residuo-Homocedasticidade'][teste][key]['H0']))
                            f.write('\n')
                        break_line = True

                    if break_line:
                        f.write('\n')
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
                        f.write('{:.5f},{:.5f},{:.5f}\n'.format(y.calculado.matriz_estimativa[i,cont],y.calculado.matriz_incerteza[i,cont],y.calculado.gL[cont][i]))
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