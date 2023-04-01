from sklearn.pipeline import make_pipeline
from babsi import BBTree
class BBPipeline:
    """
    This class contains model template that could be helpful by creating of pipeline based on BB-Tree
    """
    @classmethod
    def build_pipeline(cls, *args, tree: BBTree):
        return make_pipeline(*args, tree)
        
    