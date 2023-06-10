# encoding: utf8
import random

import numpy as np

from turtlesim_env_base import TurtlesimEnvBase


class Dqn(object):
    """Bazowa klasa Dqn"""

    def __init__(self, env: TurtlesimEnvBase, id_prefix, seed):
        random.seed(seed)
        np.random.seed(seed)

        self.env = env
        self.id_prefix = id_prefix  # przyrostek identyfikatora modelu

        self.DISCOUNT = .8  # D dyskonto dla nagrody w następnym kroku   #z jaką dokładnością ufamy tej decyzji w następnym kroku
        self.EPS_INIT = 1.0  # *  ε początkowy       -nie ruszać
        self.EPS_DECAY = .99  # *E spadek ε           -nie ruszać
        self.EPS_MIN = .05  # *e ε minimalny        -nie ruszać        #mówi w ilu procentach wykonamy ruch losowy
        self.REPLAY_MEM_SIZE_MAX = 20_000  # M rozmiar cache decyzji
        self.REPLAY_MEM_SIZE_MIN = 1_500  # m zapełnienie warunkujące uczenie (4_000)
        self.MINIBATCH_SIZE = 32  # B liczba decyzji w próbce uczącej
        self.TRAINING_BATCH_SIZE = self.MINIBATCH_SIZE // 4
        self.UPDATE_TARGET_EVERY = 20  # U co ile treningów aktualizować model wolnozmienny
        self.EPISODES_MAX = 4000  # *P liczba epizodów uczących, ile razy będzeimy trenować sieć zanim zacznie z niej korzystać   -nie ruszać
        self.CTL_DIM = 10  # liczba możliwych akcji (tj. sterowań, decyzji) są to możliwe ruchy         -    niedawać jako prędkość w scenariuszu 0 bo może się zaciąć
        self.TRAIN_EVERY = 4  # T co ile kroków uczenie modelu szybkozmiennego
        self.SAVE_MODEL_EVERY = 20  # *  co ile epizodów zapisywać model # TODO-STUDENCI

        self.model = None
        self.target_model = None
        self.replay_memory = None

    def xid(self) -> str:
        """Sygnatura eksperymentu, tj. wartości parametrów w jednym łańcych znaków.
        Używane do nazywania plików z wynikami.
            2 litery - parametr środowiska
            1 litera - parametr klasy uczącej

        :return: str
        """
        return f'{self.id_prefix}-Gr{self.env.GRID_RES}_Cr{self.env.CAM_RES}_Sw{self.env.SPEED_RWRD_RATE}' \
               f'_Sv{self.env.SPEED_RVRS_RATE}_Sf{self.env.SPEED_FINE_RATE}_Dr{self.env.DIST_RWRD_RATE}' \
               f'_Oo{self.env.OUT_OF_TRACK_FINE}_Cd{self.env.COLLISION_DIST}_Ms{self.env.MAX_STEPS}' \
               f'_Pb{self.env.PI_BY}_D{self.DISCOUNT}_E{self.EPS_DECAY}_e{self.EPS_MIN}_M{self.REPLAY_MEM_SIZE_MAX}' \
               f'_m{self.REPLAY_MEM_SIZE_MIN}_B{self.MINIBATCH_SIZE}_U{self.UPDATE_TARGET_EVERY}' \
               f'_P{self.EPISODES_MAX}_T{self.TRAIN_EVERY}'

    @staticmethod
    def ctl2act(decision: int):
        """Zakodowanie wybranego sterowania (0-5).
        Na potrzeby środowiska: (prędkość, skręt)
        prędkość/skręt -.1rad 0 .1rad

        :param decision: liczba całkowita od 0 do 5
        """
        v = .4 if decision >= 3 else .2
        w = .25 * (decision % 3 - 1)
        return [v, w]
