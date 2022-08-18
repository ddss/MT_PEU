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
import math # usado no test de 'nan'
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
            # Criação do título
            f.write('<center>')
            f.write('<h1> PARÂMETROS </h1>')
            f.write('</center>')
            # Estimativa dos parâmetros
            f.write('<hr />')
            f.write( '<table border rules = all>\n')
            f.write('<tr>\n')
            f.write(('<td><b>Simbolos</b></td>'+ ' <td><b>{:^10}</b></td> '*parametros.NV).format(*parametros.simbolos)+self.__quebra)
            f.write('</tr>\n')
            f.write('<tr>\n')
            f.write(('<td><b>Estimativa</b></td>'+ ' <td>{:^10.3e}</td> '*parametros.NV).format(*parametros.estimativa)+self.__quebra)
            f.write('</tr>\n')

            if parametros.matriz_covariancia is not None:
                # Matriz de covariância, incerteza e matriz de correlação

                f.write('<tr>\n')
                f.write(('<td><b>Variância</b></td>'+ '<td>{:^10.3e}</td> '*parametros.NV).format(*[parametros.matriz_covariancia[i,i] for i in range(parametros.NV)]) + self.__quebra)
                f.write('</tr>\n')
                f.write('<tr>\n')
                f.write(('<td><b>Incerteza</b></td>'+ '<td>{:^10.3e}</td> '*parametros.NV).format(*[parametros.matriz_incerteza[0,i] for i in range(parametros.NV)]) + self.__quebra)
                f.write('</tr>\n')
                f.write('</table>\n')
                f.write(self.__quebra)


                def constroi_matriz (matriz_nome):
                   #construção de matriz com colchetes, cria uma borda  na tabela para que no interpretador do HTML
                   #pareça com o  colchetes da matriz
                    f.write('<table>')
                    f.write('<tr><td> &#9484 </td>')
                    f.write('<td> </td>' * parametros.NV)
                    f.write('<td>  &#9488 </td> </tr>\n')

                    for id in range(parametros.NV): #Construção da matriz génerica nxn
                        f.write(' <tr><td> &#9474 </td>')
                        for id2 in range(parametros.NV):
                            f.write('<td> {:^10.3e} </td> '.format(matriz_nome[id, id2]))
                        f.write('<td> &#9474 </td> </tr>\n')

                    f.write('<tr><td> &#9492 </td>')
                    f.write('<td> </td>' * parametros.NV)
                    f.write('<td>  &#9496 </td> </tr>\n')
                    f.write('</table>\n')


                f.write('<h3>Matriz de covariância:</h3>'+self.__quebra)
                constroi_matriz(parametros.matriz_covariancia)


                f.write('<h3>Matriz de correlação:</h3>'+self.__quebra)
                constroi_matriz(parametros.matriz_correlacao)

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
            f.write( '<p> Valor da função objetivo no ponto ótimo : {:.3g} </p>'  .format(pontoOtimo))
            f.write(self.__quebra)
            f.write(('<h3>RESTRIÇÕES : </h3>'))

            f.write('<table border rules = all>\n')
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
                f.write('<center>\n') # Centraliza o objeto no HTML
                f.write('<h1> PREDIÇÃO </h1>\n')
                f.write('</center>\n')
                f.write('<hr />\n')

                f.write('<h2> GRANDEZAS DEPENDENTES </h2 >\n')
                f.write('<h3>Coeficientes de correlação:</h3> \n'+self.__quebra)



                f.write('<table border rules = all > \n') #Inicia a tabela no HTML
                f.write('<tr>\n')
                f.write('<td><b> Símbolos </b> </td>\n') #Escreve o nome símbolos apenas na primeira célula da  tabela
                f.write(('<td><b> {} </b> </td/>\n'*y.NV).format(*y.simbolos)) #Escreve os símbolos na tabela
                f.write('<tr>\n')
                f.write('<td> Coeficiente de determinação  </td> \n')
                for id3 in range(y.NV):
                    # id3 corresponde aos elementos da lista de simbolos usados para endereçar  os coeficientes no dicionário
                    f.write(( '<td> {:.3f} </td>\n').format(estatisticas['R2'][y.simbolos[id3]]))
                f.write('</tr>\n')
                f.write('<tr>\n')
                f.write('<td> Coeficiente de determinação ajustado </td>\n')
                for id3 in range(y.NV):
                    ## id3 corresponde aos elementos da lista de simbolos usados para endereçar  os coeficientes no dicionário
                    f.write(( '<td> {:.3f} </td>\n').format(estatisticas['R2ajustado'][y.simbolos[id3]]))
                f.write('</tr>\n')
                f.write('</table>\n')

                # Objective function
                f.write('<h3>Função objetivo (FO):</h3>'+self.__quebra)
                # O valor da função objetivo é selecionado para ficar na esquerda , na direita ou no centro dos valores de chi2min e chi2max .
                # &#935 ---> chi : como o HTML escreve
                if float(estatisticas['FuncaoObjetivo']['chi2max'])>float(estatisticas['FuncaoObjetivo']['FO']) and float(estatisticas['FuncaoObjetivo']['FO'])>float(estatisticas['FuncaoObjetivo']['chi2min']):
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write(' <td><b> &#935<sup>2</sup> min</b> </td> <td> <b> FO </b></td> <td><b> &#935<sup>2</sup> max</b></td> ')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['FO'])+self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2max']) + self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')

                elif float(estatisticas['FuncaoObjetivo']['FO'])<float(estatisticas['FuncaoObjetivo']['chi2min']):
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write('<td><b> FO  </b> </td> <td><b>  &#935<sup>2</sup> min </b> </td>  <td> <b>&#935<sup>2</sup> max </b></td> ')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['FO']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2max'])+self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')

                else :
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write(' <td><b>  &#935<sup>2</sup> min </b> </td>  <td> <b>&#935<sup>2</sup> max </b></td> <td><b> FO  </b> </td>')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['chi2max']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas['FuncaoObjetivo']['FO']) + self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')
                f.write(self.__quebra)
                f.write('<ul>')
                f.write( '<li> <i> Informação : </i>  a função objetivo deve estar entre &#935<sup>2</sup> min e &#935<sup>2</sup> max.</li>' + self.__quebra)
                f.write('</ul>')
                f.write(self.__quebra)

               # RESIDUAL ANALYSIS
                def Matriz_HTML (nome_teste,resíduo_nome=None):
                    #Função escreve tabela automática , o objetivo dela é escrever automaticamente as tabelas com seus respectivos testes
                    if resíduo_nome is None:
                        # Parte I da função que escreve as tabelas em Normalidade (normaltest,shapiro,anderson,kstest) e Média(ttest,ztest).
                        f.write('<table border rules="all">')
                        f.write('<tr>\n')
                        f.write('<td><b>Testes com p-valores </b></td> ')
                        f.write(('<td> <b> Resíduos para {} </b> </td> <td> <b> Aceita Ho </b> </td> \n'*y.NV).format(*y.simbolos))
                        f.write('<tr>\n')
                        # Cria 2 células para cada variável de saída e desloca para a direita
                        for teste in y._Grandeza__nomesTestes[nome_teste].keys():# Roda uma vez para cada teste , algumas das análises tem mais de um teste por isso o for
                            if not isinstance(y._Grandeza__nomesTestes[nome_teste][teste],dict):#testa se o determinado argumento é um diciónario
                                f.write('<tr>\n')
                                f.write('<td>{}</td> '.format(teste))
                            for symb in y.simbolos:
                                if isinstance(y.estatisticas[symb][nome_teste][teste],float):# testa se o determinado argumento é um float
                                    f.write('<td>{:^8.3f}</td>'.format(y.estatisticas[symb][nome_teste][teste])+' ')
                                    if float(1 - PA) < float(y.estatisticas[symb][nome_teste][teste]): #teste se aceita H0 ou não
                                        f.write('<td> Sim </td>\n  ')
                                    else:
                                        f.write('<td> Não </td>\n ')
                                elif y.estatisticas[symb][nome_teste][teste] is None:
                                    f.write('<td>{:^8}</td>'.format('N/A')+' ')
                                    f.write('<td> - </td>\n ')
                            f.write('</tr>')
                        f.write('</table>\n')
                        f.write('<ul>\n')
                        f.write('<li> <i> Ho( Hipótese nula ): </i> </b> {} </li> \n'.format(
                            y._Grandeza__TestesInfo[nome_teste][teste]['H0']))
                        f.write(
                            '<li>   <p>  <i> Informação : </i>  p-valores devem ser maiores do que o nível de '
                            'significância (1-PA) </p> <p>    para não rejeitar a hipótese nula (Ho).</li>' + self.__quebra)
                        f.write('</ul>')
                    elif nome_teste == 'Durbin Watson':
                        # Parte II da função , escreve as tabelas em Autocorrelação(Durbin Watson).
                        f.write('<table border rules="all">')
                        f.write('<tr>')
                        f.write('<td>  <b>  {:<}:   </b> </td>'.format('Durbin Watson') + (
                                    '<td> <b> Resíduos para {:^8} </b> ' * y.NV).format(
                            *y.simbolos) + '</td> </tr>' + self.__quebra)
                        if isinstance(y._Grandeza__nomesTestes[resíduo_nome][nome_teste], dict):
                                f.write('<tr> <td> {:<33}</td>'.format('estatistica'))
                                for symb in y.simbolos:
                                    if isinstance(
                                            y._Grandeza__nomesTestes[resíduo_nome][nome_teste]['estatistica'],
                                            float):
                                        f.write('<td>{:^8.3f}</td>'.format(
                                            y.estatisticas[symb][resíduo_nome][nome_teste]['estatistica']))
                                    else:
                                        f.write('<td> N/A </td>')
                                f.write('</tr>')
                        f.write('</table >\n')
                        f.write(
                            '<ul>  <li> <i> Informação : </i> </b> <p>  \n <p> <b>'
                            ' (i) </b> Se a estatística do teste estiver próxima de 0 indica autocorrelação positiva</p>\n         '
                            ' <p><b> (ii) </b>  Se a estatística do teste estiver próxima de 4 indica autocorrelação negativa</p>\n         '
                            ' <p><b> (iii) </b> Se a estatística do teste estiver próxima de 2 indica que não há autocorrelação.</li></ul>' + self.__quebra)
                        f.write(self.__quebra)

                    else:
                        # Parte III da função , escreve as tabelas em Autocorrelação(Ljung-Box) e Homocedasticidade(white test,Bresh Pagan).
                        f.write('<p><u><i> {} </u></i></p>\n'.format('Testes com p-valores'))
                        f.write('<table border rules="all">')
                        f.write('<tr>')
                        f.write('<td>  <b>  {:<}:   </b> </td>'.format(nome_teste) + (
                                '<td><b> Resíduos para {:^8} </b> <td><b> Aceita Ho </b></td>' * y.NV).format(
                            *y.simbolos) + '</td> ')
                        f.write('</tr>\n')
                        for teste in y._Grandeza__nomesTestes[resíduo_nome].keys():
                            if teste == nome_teste:
                                if isinstance(y._Grandeza__nomesTestes[resíduo_nome][teste], dict):
                                    for key in y._Grandeza__nomesTestes[resíduo_nome][teste].keys():
                                        f.write('<tr> <td> {:<33}</td>'.format(key))
                                        for symb in y.simbolos:
                                            if isinstance(y.estatisticas[symb][resíduo_nome][teste][key],
                                                          float) and not math.isnan(y.estatisticas[symb][resíduo_nome][teste][key]):
                                                f.write('<td>{:^8.3f}</td>'.format(
                                                    y.estatisticas[symb][resíduo_nome][teste][key]))
                                                if float(1 - PA) < float(
                                                        y.estatisticas[symb][resíduo_nome][teste][key]):
                                                    f.write('<td> Sim </td>\n  ')
                                                else:
                                                    f.write('<td> Não </td>\n ')
                                            else:
                                                f.write('<td>{:^8}</td>'.format('N/A') + ' ')
                                                f.write('<td> - </td>\n ')
                                        f.write('</tr>')

                        f.write('</table>\n')

                        f.write('<ul>\n')
                        f.write('<li> <i> Ho( Hipótese nula ): </i> </b> {} </li> \n'.format(
                            y._Grandeza__TestesInfo[resíduo_nome][teste][key]['H0']))

                        f.write(
                            '<li>   <p>  <i> Informação : </i>  p-valores devem ser maiores do que o nível de significância (1-PA) </p> <p>    para não rejeitar a hipótese nula (Ho).</li>' + self.__quebra)
                        f.write('</ul>')
                # RESIDUAL ANALYSIS
                f.write('<h3>Análise de resíduos:</h3>' + self.__quebra)
                # RESIDUAL ANALYSIS - normal
                f.write('<h4>Normalidade:</h4>'+self.__quebra)
                Matriz_HTML('residuo-Normalidade')
                # RESIDUAL ANALYSIS - mean test
                f.write('<h4>    Média: </h4>'+self.__quebra)
                Matriz_HTML('residuo-Media')
                # RESIDUAL ANALYSIS - autocorrelation tests
                f.write('<h4>    Autocorrelação:</h4>'+self.__quebra)
                Matriz_HTML('Ljung-Box','residuo-Autocorrelacao')
                Matriz_HTML('Durbin Watson', 'residuo-Autocorrelacao')
                # RESIDUAL ANALYSIS - homocedasticity test
                f.write('<h4> Homocedasticidade: </h4>'+self.__quebra)
                Matriz_HTML('white test','residuo-Homocedasticidade')
                Matriz_HTML( 'Bresh Pagan','residuo-Homocedasticidade')

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
