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
from numpy import inf
import xlwt
from datetime import datetime
# ---------------------------------------------------------------------
# CLASSES
# ---------------------------------------------------------------------
class Report:

    def __init__(self,fluxo,base_path=None,base_dir=None,**kwargs):
        '''
        Classe para escrita de relatórios sobre estimação de parãmetros

        ========
        Entradas
        ========

        * base_path: caminho base
        * base_dir: diretório no caminho base que os arquivos serão salvos
        '''
        self.__quebra = kwargs.get('quebra') if kwargs.get('quebra') is not None else "\n"

        # TODO: permitir fluxos
        if base_path is None:
            base_path = getcwd()

        if base_dir is None:
            base_dir = sep + 'Report' + sep

        if base_path is not None:
            Validacao_Diretorio(base_path,base_dir)

        self.__base_path = base_path + base_dir

        self.__fluxo = fluxo


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
        with open(self.__base_path+'parameters-report.html','wt') as f:
            # Criação do título: o tamanho dele será o máximo entre 65 e 18*NP (Apenas por estética)
            f.write(('<p>{:#^'+str(max([70,parametros.NV*18]))+'}</p>'+self.__quebra).format('PARÂMETROS'))
            # Estimativa dos parâmetros
            f.write( '<table border = "1">\n')
            f.write('<tr>\n')
            f.write(('<td>Simbolos</td>'+ ' <td>{:^10}</td> '*parametros.NV).format(*parametros.simbolos)+self.__quebra)
            f.write('</tr>\n')
            f.write('<tr>\n')
            f.write(('<td>Estimativa</td>'+ ' <td>{:^10.3e}</td> '*parametros.NV).format(*parametros.estimativa)+self.__quebra)
            f.write('</tr>\n')

            if parametros.matriz_covariancia is not None:
                # Matriz de covariância, incerteza e matriz de correlação
                f.write('<tr>\n')
                f.write(('<td>Variância</td>'+ '<td>{:^10.3e}</td> '*parametros.NV).format(*[parametros.matriz_covariancia[i,i] for i in range(parametros.NV)]) + self.__quebra)
                f.write('</tr>\n')
                f.write('<tr>\n')
                f.write(('<td>Incerteza</td>'+ '<td>{:^10.3e}</td> '*parametros.NV).format(*[parametros.matriz_incerteza[0,i] for i in range(parametros.NV)]) + self.__quebra)
                f.write('</tr>\n')
                f.write('</table>\n')
                f.write(self.__quebra)

                f.write('<p>Matriz de covariância:</p>'+self.__quebra)
                f.write('<p>'+str(parametros.matriz_covariancia)+'</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<p>Matriz de correlação:</p>'+self.__quebra)
                f.write('<p>'+str(parametros.matriz_correlacao)+'</p>'+self.__quebra)

            else:
                f.write('Variância : não avaliada '+self.__quebra)
                f.write('Incerteza : não avaliada '+self.__quebra)
                f.write('FObj ótima : '+ '{:.3g} '.format(pontoOtimo)+'- {:<} '.format('Valor da função objetivo no ponto ótimo')+self.__quebra)
                f.write(self.__quebra)
                f.write('Matriz de covariância: não avaliada')
                f.write(self.__quebra)
                f.write('Matriz de correlação: não avaliada')


            f.write(self.__quebra)
            # Valor da função objetivo no ponto ótimo
            f.write( '<p>FObj : '+ '{:.3g} '.format(pontoOtimo)+('- {:<} '+self.__quebra).format('Valor da função objetivo no ponto ótimo</p>'))
            f.write(self.__quebra)
            f.write(('<p>{:-^'+str(max([70,parametros.NV*18]))+'}</p>'+self.__quebra).format('RESTRIÇÕES'))

            f.write('<table border = "5">\n')
            f.write('<tr>\n')
            f.write(('<td>Simbolos</td>'+ '<td>{:^10}</td>'*parametros.NV).format(*parametros.simbolos) + self.__quebra)
            f.write('</tr>\n')
            if parametros.limite_superior != inf and parametros.limite_superior is not None:
                f.write('<tr>\n')
                f.write(('<td>Limite superior</td>'+ '<td>{:^10.3e}</td>'*parametros.NV).format(*parametros.limite_superior) + self.__quebra)
                f.write('</tr>\n')
            else:
                f.write('<tr>\n')
                f.write(('<td>Limite superior</td>'+ '<td>{:^10}</td>'*parametros.NV).format(*['N/A']*parametros.NV) + self.__quebra)
                f.write('</tr>\n')
            if parametros.limite_inferior != -inf and parametros.limite_inferior is not None:
                f.write('<tr>\n')
                f.write(('<td>Limite inferior</td>'+ '<td>{:^10.3e}</td>'*parametros.NV).format(*parametros.limite_inferior) + self.__quebra)
                f.write('</tr>\n')
            else:
                f.write('<tr>\n')
                f.write(('<td>Limite inferior</td>'+ '<td>{:^10}</td>'*parametros.NV).format(*['N/A']*parametros.NV) + self.__quebra)
                f.write('</table>\n')
            f.close()

    def Predicao(self,x,y,estatisticas,**kwargs):
        u'''
        Predicao(self,x,y,estatisticas,**kwargs)

        ============================================================================
        Write the prediction and residual analysis results in the prediction report.
        ============================================================================
            - Parameters
            ------------
            x : grandeza class instance
                instance containing the information relating to the independent variables.
            y : grandeza class instance
                instance containing the information relating to the dependent variables.
            estatisticas : dict
                dictionary with the R2, adjusted R2, and FO (objective function) values.

            - keywords
            -----------
            export_y : bool
                exports the calculated data of y, its uncertainty, and degrees of freedom in a txt with comma separation.
            export_y_xls : bool
                exports the calculated data of y, its uncertainty, and degrees of freedom in a xls.
            export_cov_y : bool
                exports the covariance matrix of y.
            export_x : bool
                exports the calculated data of x, its uncertainty, and degrees of freedom in a txt with comma separation.
            export_cov_x : bool
                exports the covariance matrix of x.
                
            - References
            -------------
            [1] https://docs.python.org/2/tutorial/inputoutput.html
            [2] https://docs.python.org/2/library/string.html#formatstrings
        '''

        self._configFolder={'graficos-subfolder-Dadosvalidacao': 'Dados Validacao',
                            'graficos-subfolder-DadosEstimacao': 'Dados Estimacao'}

        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        if not isinstance(kwargs.get('export_y'),bool) and kwargs.get('export_y') is not None:
            raise TypeError('A keyword export_y deve ser booleana')
        if not isinstance(kwargs.get('export_cov_y'),bool) and kwargs.get('export_cov_y') is not None:
            raise TypeError('A keyword export_cov_y deve ser booleana')
        if not isinstance(kwargs.get('export_y_xls'), bool) and kwargs.get('export_y_xls') is not None:
            raise TypeError('A keyword export_y_xls deve ser booleana')
        if kwargs.get('export_y_xls') is None:
            export_y_xls = False
        else:
            export_y_xls = kwargs.get('export_y_xls')
        if kwargs.get('export_y') is None:
            export_y = False
        else:
            export_y = kwargs.get('export_y')
        if kwargs.get('export_cov_y') is None:
            export_cov_y = False
        else:
            export_cov_y = kwargs.get('export_cov_y')

        PA = kwargs.get('PA')
        # ---------------------------------------------------------------------
        # REPORT FILE WRITING
        # ---------------------------------------------------------------------
        # Internal folders
        #------------------------------------------------------------
        if int(self.__fluxo) > 0:
            folder = sep + self._configFolder['graficos-subfolder-Dadosvalidacao']+' '+self.__fluxo + sep
            Validacao_Diretorio(self.__base_path, folder)
        else:
            folder = sep + self._configFolder['graficos-subfolder-DadosEstimacao'] + sep
            Validacao_Diretorio(self.__base_path, folder)
        #------------------------------------------------------------
        if estatisticas is not None:
            #with open(self.__base_path+folder+'prediction-report_fl'+self.__fluxo+'.txt','wt') as f:
            with open(self.__base_path+folder+'prediction-report'+'.html','wt') as f:
                # TITLE:
                f.write(('<p>{:#^'+str(max([70,y.NV*18]))+'}</p>'+self.__quebra).format(' PREDIÇÃO '))
                f.write(('<p>{:=^'+str(max([70,y.NV*18]))+'}</p>'+self.__quebra).format(' GRANDEZAS DEPENDENTES '))
                # R2:
                f.write('<p>Coeficientes de correlação:</p>'+self.__quebra)
                # Operator to format the values of R2 and R2 adjusted, because they are dictionaries.
                construtor_formatacao_simbolos = ['{'+str(symb)+':^8.3f}' for symb in y.simbolos]
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Simbolos:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'+ ('{:^8}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'*y.NV).format(*y.simbolos) +'</p>' + self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Coeficiente de determinação&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;: '+ ('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.join(construtor_formatacao_simbolos)).format(**estatisticas['R2'])+'</p>' + self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Coeficiente de determinação ajustado&nbsp;: '+ ('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.join(construtor_formatacao_simbolos)).format(**estatisticas['R2ajustado'])+'</p>' + self.__quebra)
                f.write(self.__quebra)

                # Objective function
                f.write('<p>Função objetivo:</p>'+self.__quebra)
                f.write('</p>&nbsp;&nbsp;&nbsp;&nbsp;Info: a função objetivo deve estar entre chi2min e chi2max.'+self.__quebra)
                # Operator to format the values of R2 and R2 adjusted, because they are dictionaries.
                f.write('<table border = "1">\n')
                f.write('<tr>\n')
                f.write('<td>chi2max</td><td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2max'])+self.__quebra)
                f.write('</tr>\n')
                f.write('<td>FO     </td><td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['FO'])+self.__quebra)
                f.write('</tr>\n')
                f.write('<td>chi2min</td><td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2min'])+self.__quebra)
                f.write('</tr>\n')
                f.write('</table>\n')
                f.write(self.__quebra)

                # RESIDUAL ANALYSIS - Normality tests
                f.write('<p>Análise de resíduos:</p>'+self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Normalidade:</p>'+self.__quebra)
                f.write('<p>{:-^45}</p>'.format('Testes com p-valores')+self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Info: p-valores devem ser maiores do que o nível de significânca (1-PA)</p>'+self.__quebra+'<p>&nbsp;&nbsp;&nbsp;&nbsp;para não rejeitar a hipótese nula (Ho).</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<table border = "1">\n')
                f.write('</tr>\n')
                f.write('<th>{:<10} &nbsp;&nbsp;&nbsp;&nbsp;</th>'.format('Simbolos')+ ('<th colspan="1">{}</th>'*y.NV).format(*y.simbolos)+self.__quebra)
                # semi-automated construction to fill in the values of the statistical normality tests.
                for teste in y._Grandeza__nomesTestes['residuo-Normalidade'].keys():
                    break_line = False
                    if not isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                        f.write('</tr>\n')
                        f.write('<td>{}:&nbsp;&nbsp;&nbsp;&nbsp;</p> '.format(teste))
                    for symb in y.simbolos:
                        if isinstance(y.estatisticas[symb]['residuo-Normalidade'][teste],float):
                            f.write('<td>{:^8.3f}</td>'.format(y.estatisticas[symb]['residuo-Normalidade'][teste])+' ')
                            break_line = True
                        elif y.estatisticas[symb]['residuo-Normalidade'][teste] is None:
                            f.write('<td>{:^8}</td>'.format('N/A')+' ')
                            break_line = True
                    if not isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                        f.write('<td> Ho: {}</td>'.format(y._Grandeza__TestesInfo['residuo-Normalidade'][teste]['H0']))
                    if break_line:
                        f.write(self.__quebra)
                f.write('</table>\n')
                f.write(self.__quebra)
                # f.write('    {:-^45}'.format('Testes com valores críticos')+self.__quebra)
                # f.write('    {:<}:                '.format('Simbolos')+ ('{:^37}'*y.NV).format(*y.simbolos)+self.__quebra)
                # for teste in y._Grandeza__nomesTestes['residuo-Normalidade'].keys():
                #     break_line = False
                #     if isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste],dict):
                #         f.write('        {:<}: '.format(teste)+self.__quebra)
                #         for key in y._Grandeza__nomesTestes['residuo-Normalidade'][teste].keys():
                #             f.write('            {:<16}:'.format(key))
                #             for symb in y.simbolos:
                #                 if isinstance(y._Grandeza__nomesTestes['residuo-Normalidade'][teste][key],float):
                #                     f.write('{:^37.3f}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste][key]))
                #                 else:
                #                     f.write('{:^37}'.format(y.estatisticas[symb]['residuo-Normalidade'][teste][key]))
                #             f.write(self.__quebra)
                #         break_line = True
                #
                #     if break_line:
                #         f.write(self.__quebra)

                # RESIDUAL ANALYSIS - mean test
                f.write('<p>    Média:</p>'+self.__quebra)
                f.write('<p>    {:-^45}</p>'.format('Testes com p-valores')+self.__quebra)
                f.write('<p>    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)</p>'+self.__quebra+'    para não rejeitar a hipótese nula (Ho).</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<p>    {:<8}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:'.format('Simbolos') + ('{:^8}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'*y.NV).format(*y.simbolos) +'</p>'+ self.__quebra)
                # construção semi-automatizada para preencher os valores dos testes estatísticos para média
                for teste in y._Grandeza__nomesTestes['residuo-Media'].keys():
                     f.write('<p>    {:<8}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:'.format(teste))
                     for symb in y.simbolos:
                         if isinstance(y.estatisticas[symb]['residuo-Media'][teste],float):
                             f.write('{:^8.3f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format(y.estatisticas[symb]['residuo-Media'][teste])+' ')
                         else:
                             f.write('N/A')
                             f.write(self.__quebra)
               
                     if not isinstance(y._Grandeza__nomesTestes['residuo-Media'][teste],dict):
                        f.write(' | Ho: {}</p>'.format(y._Grandeza__TestesInfo['residuo-Media'][teste]['H0']))
                        f.write(self.__quebra)

                     f.write('             ')
                     for symb in y.simbolos:
                        if isinstance(y.estatisticas[symb]['residuo-Media'][teste], float):
                            f.write('&nbsp;&nbsp;&nbsp;&nbsp;Aceita H0</p>' if y.estatisticas[symb]['residuo-Media'][teste] >= 1-PA else '&nbsp;&nbsp;&nbsp;&nbsp;Rejeita H0</p>')
                        else:
                            f.write('N/A</p>')
                            f.write(self.__quebra)

                     if break_line:
                        f.write(self.__quebra)

                 # RESIDUAL ANALYSIS - autocorrelation tests
                f.write(self.__quebra)
                f.write('<p>    Autocorrelação:</p>'+self.__quebra)
                f.write('<p>    {:-^45}</p>'.format('Testes com p-valores')+self.__quebra)
                f.write('<p>    Info: p-valores devem ser maiores do que o nível de significânca (1-PA)</p>\n   <p> para não rejeitar a hipótese nula (Ho).</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<p>    {:<}:                                 '.format('Simbolos')+ ('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:^8}'*y.NV).format(*y.simbolos)+'</p>'+self.__quebra)
                for teste in y._Grandeza__nomesTestes['residuo-Autocorrelacao'].keys():
                    break_line = False
                    if teste == 'Ljung-Box':
                      if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                          f.write('<p>    {:<}:</p>'.format(teste)+self.__quebra)
                          for key in y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste].keys():
                              f.write('<p>            {:<33}:'.format(key))
                              for symb in y.simbolos:
                                  if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste][key],float):
                                      f.write('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:^8.3f}'.format(y.estatisticas[symb]['residuo-Autocorrelacao'][teste][key]))
                                  else:
                                      f.write('N/A')
                              if  isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                                    f.write('| Ho: {}</p>'.format(y._Grandeza__TestesInfo['residuo-Autocorrelacao'][teste][key]['H0']))

                              f.write(self.__quebra)
                          break_line = True

#
                    if break_line:
                        f.write(self.__quebra)

                f.write('<p>    {:-^45}</p>'.format('Testes com estatística')+self.__quebra)
                f.write('<p>    Info: (i) Se a estatística do teste estiver próxima de 0 indica autocorrelação positiva</p>\n          <p>(ii) Se a estatística do teste estiver próxima de 4 indica autocorrelação negativa</p>\n          <p>(iii) Se a estatística do teste estiver próxima de 2 indica que não há autocorrelação.</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;{:<}:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format('Simbolos')+ ('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:^8}'*y.NV).format(*y.simbolos)+'</p>'+self.__quebra)
                for teste in y._Grandeza__nomesTestes['residuo-Autocorrelacao'].keys():
                    break_line = False
                    if teste == 'Durbin Watson':
                      if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste],dict):
                          f.write('<p>{:<}:</p>'.format(teste)+self.__quebra)
                          for key in y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste].keys():
                              f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:<11}:'.format(key))
                              for symb in y.simbolos:
                                  if isinstance(y._Grandeza__nomesTestes['residuo-Autocorrelacao'][teste][key],float):
                                      f.write('{:^8.3f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format(y.estatisticas[symb]['residuo-Autocorrelacao'][teste][key]))
                                  else:
                                      f.write('N/A</p>')
                              f.write(self.__quebra)
                          break_line = True

                      if break_line:
                          f.write(self.__quebra)

                # RESIDUAL ANALYSIS - homocedasticity test
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Homocedasticidade:</p>'+self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;{:-^45}</p>'.format('Testes com p-valores')+self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;Info: p-valores devem ser maiores do que o nível de significânca (1-PA)</p>\n <p>para não rejeitar a hipótese nula (Ho).</p>'+self.__quebra)
                f.write(self.__quebra)
                f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;{:<}:                                 '.format('Simbolos')+ ('&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:^8}'*y.NV).format(*y.simbolos)+'</p>'+self.__quebra)
                for teste in y._Grandeza__nomesTestes['residuo-Homocedasticidade'].keys():
                    break_line = False
                    if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste],dict):
                        f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;{:<}: </p>'.format(teste)+self.__quebra)
                        for key in y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste].keys():
                            f.write('<p>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{:<33}:'.format(key))
                            for symb in y.simbolos:
                                if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste][key],float):
                                    f.write('{:^8.3f}&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;'.format(y.estatisticas[symb]['residuo-Homocedasticidade'][teste][key]))
                                else:
                                    f.write('N/A')
                            if isinstance(y._Grandeza__nomesTestes['residuo-Homocedasticidade'][teste],dict):
                                    f.write('| Ho: {}</p>'.format(y._Grandeza__TestesInfo['residuo-Homocedasticidade'][teste][key]['H0']))
                            f.write(self.__quebra)
                        break_line = True

                    if break_line:
                        f.write(self.__quebra)
            f.close()
        # ---------------------------------------------------------------------
        # PREDICTION EXPORT
        # ---------------------------------------------------------------------
        # Calculated values and uncertainty
        if export_y: # txt format
            cont = 0
            for symb in y.simbolos:
                # with open(self.__base_path+folder+symb+'-calculado-predicao_fl'+self.__fluxo+'.txt','wt') as f:
                with open(self.__base_path+folder+symb+'-calculado-predicao'+'.txt','wt') as f:
                    for i in range(y.calculado.NE):
                        f.write('{:.5g},{:.5g},{:.5g}'.format(y.calculado.matriz_estimativa[i,cont],y.calculado.matriz_incerteza[i,cont],y.calculado.gL[cont][i])+self.__quebra)
                f.close()
                cont+=1
        if export_y_xls: # xls format
            cont = 0
            wb = xlwt.Workbook()
            ws = wb.add_sheet('calculado-predicao')
            for i in range(y.calculado.NE):
                 ws.write(i, 0, y.calculado.matriz_estimativa[i, cont]), ws.write(i, 1, y.calculado.matriz_incerteza[i, cont]), ws.write(i, 2, y.calculado.gL[cont][i])
            for symb in y.simbolos:
                wb.save(self.__base_path+folder+symb+'-calculado-predicao'+'.xls')
        # covariance matrix
        if export_cov_y:
            # with open(self.__base_path+folder+'y-calculado-matriz-covariancia_fl'+self.__fluxo+'.txt','wt') as f:
            with open(self.__base_path+folder+'y-calculado-matriz-covariancia'+'.txt','wt') as f:
                for i in range(y.NV*y.calculado.NE):
                    for j in range(y.NV*y.calculado.NE):
                        f.write('{:.5g} '.format(y.calculado.matriz_covariancia[i,j]))
                    f.write(self.__quebra)
            f.close()

    def optimization(self):
        return self.__base_path
