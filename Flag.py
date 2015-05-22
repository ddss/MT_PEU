# -*- coding: utf-8 -*-
"""
Classe auxiliar para controle de Flags

@author(es): Daniel, Francisco,
@GrupoPesquisa: PROTEC
@LinhadePesquisa: GI-UFBA
"""

class flag:

    def __init__(self):
        u'''Classe para padronizar o uso de flags.

        =======
        Entrada
        =======

        * caracteristicas (List): lista contendo as características das flags

        =========
        Atributos
        =========

        * **info**: dicionário que informa a situação atual das flags. As chaves são as características das flags e o conteúdo seu status.

        IMPORTANTE: Quando iniciada, as flags possuem status como FALSE.

        =======
        Métodos
        =======

        * **ToggleActive(caracteristica)** : muda o status da caracteristica da flag para TRUE
        * **ToggleInactive(caracteristica)**: muda o status da caracteristica da flag para FALSE

        =======
        Exemplo
        =======
        Exemplo: ::

        >>> Fl = flag()
        >>> Fl.setCaracteristica(['estimacao','reconciliacao'])
        >>> Fl.info
        >>> Fl.ToggleActive('estimacao')
        >>> Fl.info
        >>> Fl.ToggleInactive('estimacao')
        >>> Fl.info
        >>> Fl.setCaracteristica(['validacao'])
        >>> Fl.info
        >>> Fl.ToggleActive(['validacao'])
        >>> Fl.info
        '''

        # ---------------------------------------------------------------------
        # INICIALIZAÇÃO
        # ---------------------------------------------------------------------
        self._caracteristicas_disponiveis = []

        self.info = {}

    def setCaracteristica(self,caracteristica):
        '''uMétodo para incluir uma lista de características

        =======
        Entrada
        =======

        * caracteristica (List): lista de características a serem incluídas

        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if not isinstance(caracteristica,list):
            raise TypeError(u'a entrada "características" deve ser do tipo list.')

        # ---------------------------------------------------------------------
        # INCLUSÃO DA(S) NOVA(S) CARACTERÍSTICAS
        # ---------------------------------------------------------------------
        # Inclusão na lista de características disponíveis
        self._caracteristicas_disponiveis.extend(caracteristica)

        # Definição do status para o valor default de falso.
        for elemento in caracteristica:
            self.info[elemento] = False


    def __validacao(self,caracteristica):
        u'''Validação das entradas
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        if isinstance(caracteristica,str):

            if caracteristica not in self._caracteristicas_disponiveis:
                raise NameError(u'A caracteristica "'+str(caracteristica)+'" não está disponível. Características disponíveis: '+', '.join(self._caracteristicas_disponiveis)+'.')

        elif isinstance(caracteristica,list):

            teste = [isinstance(elemento,str) for elemento in caracteristica]

            if False in teste:
                raise TypeError('As características devem ser strings.')

            diferenca = set(caracteristica).difference(set(self._caracteristicas_disponiveis))
            if len(diferenca) != 0:
                raise NameError(u'Característica(s) indisponível(is): '+', '.join(diferenca) +'. Características disponíveis: '+', '.join(self._caracteristicas_disponiveis)+'.')

        else:
            raise TypeError(u'A caracteristica deve ser uma lista ou um string.')

        # ---------------------------------------------------------------------
        # AÇÃO
        # ---------------------------------------------------------------------
        # TODO: Permitir somente a inclusão de lista. Retirar este IF.

        if not isinstance(caracteristica,list):
            caracteristica = [caracteristica]

        # TODO: relocar este código para ToggleActive e ToggleInactive.
        self.__caracteristica = caracteristica # indica qual característica será modificada (Active/Inactive)

    def __Toggle(self):
        u'''
         Método interno para realizar ação de mudança de status
        '''

        for elemento in self.__caracteristica:
            self.info[elemento]    = self.__togglestatus

    def ToggleActive(self,caracteristica):
        '''
        Irá marcar a flag como TRUE
        =======
        Entrada
        =======

        * característica (lista de strings ou string): o que a flag está indicando
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        self.__validacao(caracteristica)

        self.__togglestatus = True
        self.__Toggle()

    def ToggleInactive(self,caracteristica):
        '''
        Irá marcar a flag como FALSE
        =======
        Entrada
        =======

        * característica (lista de strings ou string): o que a flag está indicando
        '''
        # ---------------------------------------------------------------------
        # VALIDAÇÃO
        # ---------------------------------------------------------------------
        self.__validacao(caracteristica)

        self.__togglestatus = False
        self.__Toggle()