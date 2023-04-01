from sklvq import GMLVQ

class GMLVQProperties:
    """
    This class is responsible for gathering necessary data from GMLVQ model. 
    """

    def __init__(self, prototypes, relevance_matrix, features, classes) -> None:
        if relevance_matrix.shape[0] != len(features):
            raise TypeError('Features do not correspond to relevance matrix shape')
        self.prototypes = prototypes
        self.relevance_matrix = relevance_matrix
        self.features = features
        self.classes = classes

    
    @classmethod
    def build_from_GMLVQ(cls, model: GMLVQ, features):
        return GMLVQProperties(
            relevance_matrix=model.lambda_, 
            prototypes=[
                {'vector': vector, 'label': label.item()} 
                for vector, label in zip(
                    model.prototypes_,
                    model.prototypes_labels_
                )
            ],
            classes=model.classes_.tolist(),
            features=features
        )
