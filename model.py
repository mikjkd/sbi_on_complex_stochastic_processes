# %%
# creo un modello M(theta, f)
# theta ha lungezza fissata k < K
# il numero di processi stocastici è 2 k = 2
# ha dei metodi per generare timeseries di lunghezza T
from typing import List

from process import ProbabilisticCharacterization, StrangeProcessf


class Model:
    def __init__(self, theta: List[ProbabilisticCharacterization], f):
        self.theta = theta
        self.process = None
        pc1 = self.theta[0]
        pc2 = self.theta[1]

        self.process = StrangeProcessf(pc1=pc1, pc2=pc2, change_rate=0.7, f=f)
    def generate(self, T):
        # necessario il reset al fine di mantenere la lungezza della ts fissa
        # in quanto generate_n(T) fa un append di T elementi alla vecchia serie
        self.process.reset()
        return self.process.generate_n(T)