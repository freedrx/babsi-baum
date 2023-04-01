from sklvq import GMLVQ

from sklearn.preprocessing import StandardScaler

SUPPORTED_MODELS = ['GMLVQ']

class LVQBuilder:
    """
    The class is responsible for building and training of LVQ based models
    """
    def __init__(self, lvq_type: str, prototypes_per_class: int) -> None:
        if lvq_type not in SUPPORTED_MODELS:
            raise TypeError('Entered model type is not supported')
        self.lvq_type = lvq_type
        self.prototypes_per_class = prototypes_per_class
        self.model = None

    def build(self):
        if self.lvq_type == 'GMLVQ':
            self.model = GMLVQ(prototype_n_per_class=self.prototypes_per_class)
        return self
    
    def train(self, X, y, scale_needed=True):
        if scale_needed:
            X = StandardScaler().fit_transform(X)
        self.model.fit(X, y)
        return self

    def get_model(self):
        return self.model