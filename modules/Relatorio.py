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

    def __init__(self,base_path=None, base_dir=None,**kwargs):
        '''
        Classe para escrita de relatórios sobre estimação de parãmetros

        ========
        Entradas
        ========

        * base_path: caminho base
        * base_dir: diretório no caminho base que os arquivos serão salvos
        '''
        self.__quebra = kwargs.get('quebra') if kwargs.get('quebra') is not None else "\n"

        if base_path is None:
            base_path = getcwd()

        if base_dir is None:
            base_dir = sep + 'Report' + sep

        if base_path is not None:
            Validacao_Diretorio(base_path,base_dir)

        self.__base_path = base_path + base_dir

    def Parametros(self,parametros, pontoOtimo):
        '''
        Escrita sobre a etapa a estimativa dos parâmetros e sua uncertainty

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
                # Matriz de covariância, uncertainty e matriz de correlação
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
                f.write('</table>\n')
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
            f.write(('<h3>INTERVALOS DE ABRANGÊNCIA : </h3>'))

            f.write('<table border rules = all>\n')
            f.write('<tr>\n')
            f.write(
                ('<td>Simbolos</td>' + '<td>{:^10}</td>' * parametros.NV).format(*parametros.simbolos) + self.__quebra)
            f.write('</tr>\n')
            if parametros.interval_lb is not None:
                f.write('<tr>\n')
                f.write(('<td>lb:</td>' + '<td>{:^10.3e}</td>' * parametros.NV).format(
                    *parametros.interval_lb) + self.__quebra)
                f.write('</tr>\n')
            else:
                f.write('<tr>\n')
                f.write(('<td>lb</td>' + '<td>{:^10}</td>' * parametros.NV).format(
                    *['N/A'] * parametros.NV) + self.__quebra)
                f.write('</tr>\n')
            if parametros.interval_up is not None:
                f.write('<tr>\n')
                f.write(('<td>ub</td>' + '<td>{:^10.3e}</td>' * parametros.NV).format(
                    *parametros.interval_up) + self.__quebra)
                f.write('</tr>\n')
                f.write('</table>\n')
            else:
                f.write('<tr>\n')
                f.write(('<td>ub:</td>' + '<td>{:^10}</td>' * parametros.NV).format(
                    *['N/A'] * parametros.NV) + self.__quebra)
                f.write('</table>\n')

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
                f.write('</table>\n')
            else:
                f.write('<tr>\n')
                f.write(('<td>Limite inferior</td>'+ '<td>{:^10}</td>'*parametros.NV).format(*['N/A']*parametros.NV) + self.__quebra)
                f.write('</table>\n')
            f.close()
    def Grandezas(self, z, estatisticas, dataType, **kwargs):
        u'''
        Grandezas(self, z, estatisticas,**kwargs)

        ============================================================================
        Write the prediction and residual analysis results in the prediction report.
        ============================================================================

        - Parameters
        ------------
        z : grandeza class instance
            instance containing the information relating to the variables.
        estatisticas : dict
            dictionary with the R2, adjusted R2, and FO (objective function) values.
        dataType
        - keywords
        -----------

        export_z : bool
            exports the calculated data of z, its uncertainty, and degrees of freedom in a txt with comma separation.
        export_z_xls : bool
            exports the calculated data of z, its uncertainty, and degrees of freedom in a xls.
        export_cov_z : bool
            exports the covariance matrix of z.
        - References
        -------------

        [1] https://docs.python.org/2/tutorial/inputoutput.html

        [2] https://docs.python.org/2/library/string.html#formatstrings
        '''
        # ---------------------------------------------------------------------
        # VALIDATION
        # ---------------------------------------------------------------------
        if not isinstance(kwargs.get('export_z'),bool) and kwargs.get('export_z') is not None:
            raise TypeError('A keyword export_z deve ser booleana')
        if not isinstance(kwargs.get('export_cov_z'),bool) and kwargs.get('export_cov_z') is not None:
            raise TypeError('A keyword export_cov_z deve ser booleana')
        if not isinstance(kwargs.get('export_z_xls'), bool) and kwargs.get('export_z_xls') is not None:
            raise TypeError('A keyword export_z_xls deve ser booleana')
        if kwargs.get('export_z_xls') is None:
            export_z_xls = False
        else:
            export_z_xls = kwargs.get('export_z_xls')
        if kwargs.get('export_z') is None:
            export_z = False
        else:
            export_z = kwargs.get('export_z')
        if kwargs.get('export_cov_z') is None:
            export_cov_z = False
        else:
            export_cov_z = kwargs.get('export_cov_z')

        PA = kwargs.get('PA')
        # ---------------------------------------------------------------------
        # REPORT FILE WRITING
        # ---------------------------------------------------------------------

        #------------------------------------------------------------
        if estatisticas is not None:
            with open(self.__base_path+'grandezas-report-{}'.format(dataType)+'.html','wt') as f:
                # TITLE:
                f.write('<center>\n') # Centraliza o objeto no HTML
                f.write('<h1> PREDIÇÃO </h1>\n')
                f.write('</center>\n')
                f.write('<hr />\n')

                f.write('<h2> GRANDEZAS </h2 >\n')
                f.write('<h3>Coeficientes de correlação:</h3> \n'+self.__quebra)
                f.write('<table border rules = all > \n') #Inicia a tabela no HTML
                f.write('<tr>\n')
                f.write('<td><b> Símbolos </b> </td>\n') #Escreve o nome símbolos apenas na primeira célula da  tabela
                f.write(('<td><b> {} </b> </td/>\n' * z.NV).format(*z.simbolos)) #Escreve os símbolos na tabela
                f.write('<tr>\n')
                f.write('<td> Coeficiente de determinação  </td> \n')
                for id3 in range(z.NV):
                    # id3 corresponde aos elementos da lista de simbolos usados para endereçar  os coeficientes no dicionário
                    f.write(( '<td> {:.3f} </td>\n').format(estatisticas[dataType]['R2'][z.simbolos[id3]]))
                f.write('</tr>\n')
                f.write('<tr>\n')
                f.write('<td> Coeficiente de determinação ajustado </td>\n')
                for id3 in range(z.NV):
                    ## id3 corresponde aos elementos da lista de simbolos usados para endereçar  os coeficientes no dicionário
                    f.write(( '<td> {:.3f} </td>\n').format(estatisticas[dataType]['R2adjusted'][z.simbolos[id3]]))
                f.write('</tr>\n')
                f.write('</table>\n')

                # Objective function
                f.write('<h3>Função objetivo (FO):</h3>'+self.__quebra)
                # O valor da função objetivo é selecionado para ficar na esquerda , na direita ou no centro dos valores de chi2min e chi2max .
                # &#935 ---> chi : como o HTML escreve
                if float(estatisticas[dataType]['ObjectiveFunction']['chi2max'])>float(estatisticas[dataType]['ObjectiveFunction']['FO']) and float(estatisticas[dataType]['ObjectiveFunction']['FO'])>float(estatisticas[dataType]['ObjectiveFunction']['chi2min']):
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write(' <td><b> &#935<sup>2</sup> min</b> </td> <td> <b> FO </b></td> <td><b> &#935<sup>2</sup> max</b></td> ')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['FO'])+self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2max']) + self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')

                elif float(estatisticas[dataType]['ObjectiveFunction']['FO'])<float(estatisticas[dataType]['ObjectiveFunction']['chi2min']):
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write('<td><b> FO  </b> </td> <td><b>  &#935<sup>2</sup> min </b> </td>  <td> <b>&#935<sup>2</sup> max </b></td> ')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['FO']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2max'])+self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')

                else :
                    f.write('<table border rules = all>\n')
                    f.write('<tr>\n')
                    f.write(' <td><b>  &#935<sup>2</sup> min </b> </td>  <td> <b>&#935<sup>2</sup> max </b></td> <td><b> FO  </b> </td>')
                    f.write('</tr>\n')
                    f.write('<tr>\n')
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2min']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['chi2max']) + self.__quebra)
                    f.write('<td> {:.3f}</td>'.format(estatisticas[dataType]['ObjectiveFunction']['FO']) + self.__quebra)
                    f.write('</tr>\n')
                    f.write('</table>\n')
                f.write(self.__quebra)
                f.write('<ul>')
                f.write( '<li> <i> Informação : </i>  a função objetivo deve estar entre &#935<sup>2</sup> min e &#935<sup>2</sup> max.</li>' + self.__quebra)
                f.write('</ul>')
                f.write(self.__quebra)

               # RESIDUAL ANALYSIS
                def Matriz_HTML (nome_teste, residuo_nome=None):
                    #Função escreve tabela automática , o objetivo dela é escrever automaticamente as tabelas com seus respectivos testes
                    if residuo_nome is None:
                        # Parte I da função que escreve as tabelas em Normalidade (normaltest,shapiro,anderson,kstest) e Média(ttest,ztest).
                        f.write('<table border rules="all">')
                        f.write('<tr>\n')
                        f.write('<td><b>Testes com p-valores </b></td> ')
                        f.write(('<td> <b> Resíduos para {} </b> </td> <td> <b> Aceita Ho </b> </td> \n' * z.NV).format(*z.simbolos))
                        f.write('<tr>\n')
                        # Cria 2 células para cada variável de saída e desloca para a direita
                        for teste in z._Grandeza__nomesTestes[nome_teste].keys():# Roda uma vez para cada teste , algumas das análises tem mais de um teste por isso o for
                            if not isinstance(z._Grandeza__nomesTestes[nome_teste][teste], dict):#testa se o determinado argumento é um diciónario
                                f.write('<tr>\n')
                                f.write('<td>{}</td> '.format(teste))
                            for symb in z.simbolos:
                                if isinstance(z.estatisticas[dataType][symb][nome_teste][teste], float):# testa se o determinado argumento é um float
                                    f.write('<td>{:^8.3f}</td>'.format(z.estatisticas[dataType][symb][nome_teste][teste]) + ' ')
                                    if float(1 - PA) < float(z.estatisticas[dataType][symb][nome_teste][teste]): #teste se aceita H0 ou não
                                        f.write('<td> Sim </td>\n  ')
                                    else:
                                        f.write('<td> Não </td>\n ')
                                elif z.estatisticas[dataType][symb][nome_teste][teste] is None:
                                    f.write('<td>{:^8}</td>'.format('N/A')+' ')
                                    f.write('<td> - </td>\n ')
                            f.write('</tr>')
                        f.write('</table>\n')
                        f.write('<ul>\n')
                        f.write('<li> <i> Ho( Hipótese nula ): </i> </b> {} </li> \n'.format(
                            z._Grandeza__TestesInfo[nome_teste][teste]['H0']))
                        f.write(
                            '<li>   <p>  <i> Informação : </i>  p-valores devem ser maiores do que o nível de '
                            'significância (1-PA) </p> <p>    para não rejeitar a hipótese nula (Ho).</li>' + self.__quebra)
                        f.write('</ul>')
                    elif nome_teste == 'Durbin Watson':
                        # Parte II da função , escreve as tabelas em Autocorrelação(Durbin Watson).
                        f.write('<table border rules="all">')
                        f.write('<tr>')
                        f.write('<td>  <b>  {:<}:   </b> </td>'.format('Durbin Watson') + (
                                    '<td> <b> Resíduos para {:^8} </b> ' * z.NV).format(
                            *z.simbolos) + '</td> </tr>' + self.__quebra)
                        if isinstance(z._Grandeza__nomesTestes[residuo_nome][nome_teste], dict):
                                f.write('<tr> <td> {:<33}</td>'.format('estatistica'))
                                for symb in z.simbolos:
                                    if isinstance(
                                            z._Grandeza__nomesTestes[residuo_nome][nome_teste]['estatistica'],
                                            float):
                                        f.write('<td>{:^8.3f}</td>'.format(
                                            z.estatisticas[dataType][symb][residuo_nome][nome_teste]['estatistica']))
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
                                '<td><b> Resíduos para {:^8} </b> <td><b> Aceita Ho </b></td>' * z.NV).format(
                            *z.simbolos) + '</td> ')
                        f.write('</tr>\n')
                        for teste in z._Grandeza__nomesTestes[residuo_nome].keys():
                            if teste == nome_teste:
                                if isinstance(z._Grandeza__nomesTestes[residuo_nome][teste], dict):
                                    for key in z._Grandeza__nomesTestes[residuo_nome][teste].keys():
                                        f.write('<tr> <td> {:<33}</td>'.format(key))
                                        for symb in z.simbolos:
                                            if isinstance(z.estatisticas[dataType][symb][residuo_nome][teste][key],
                                                          float) and not math.isnan(z.estatisticas[dataType][symb][residuo_nome][teste][key]):
                                                f.write('<td>{:^8.3f}</td>'.format(
                                                    z.estatisticas[dataType][symb][residuo_nome][teste][key]))
                                                if float(1 - PA) < float(
                                                        z.estatisticas[dataType][symb][residuo_nome][teste][key]):
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
                            z._Grandeza__TestesInfo[residuo_nome][teste][key]['H0']))

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
        if export_z: # txt format
            cont = 0
            for symb in z.simbolos:
                with open(self.__base_path+symb+'-evaluated-{}'.format(dataType)+'.txt','wt') as f:
                    for i in range(z.evaluated[dataType].NE):
                        f.write('{:.5g},{:.5g},{:.5g}'.format(z.evaluated[dataType].matriz_estimativa[i,cont], z.evaluated[dataType].matriz_incerteza[i,cont], z.evaluated[dataType].gL[cont][i]) + self.__quebra)
                f.close()
                cont+=1
        if export_z_xls: # xls format
            cont = 0
            wb = xlwt.Workbook()
            ws = wb.add_sheet('evaluated-{}'.format(dataType))
            for i in range(z.evaluated[dataType].NE):
                 ws.write(i, 0, z.evaluated[dataType].matriz_estimativa[i, cont]), ws.write(i, 1, z.evaluated[dataType].matriz_incerteza[i, cont]), ws.write(i, 2, z.evaluated[dataType].gL[cont][i])
            for symb in z.simbolos:
                wb.save(self.__base_path+symb+'-evaluated-{}'.format(dataType)+'.xls')
        # covariance matrix
        if export_cov_z:
            # with open(self.__base_path+'z-evaluated-matriz-covariancia_fl'+self.__fluxo+'.txt','wt') as f:
            with open(self.__base_path+'z-evaluated-{}-matriz-covariancia'.format(dataType)+'.txt','wt') as f:
                for i in range(z.NV * z.evaluated[dataType].NE):
                    for j in range(z.NV * z.evaluated[dataType].NE):
                        f.write('{:.5g} '.format(z.evaluated[dataType].matriz_covariancia[i,j]))
                    f.write(self.__quebra)
            f.close()

    def optimization(self):
        return self.__base_path
