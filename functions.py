import numpy as np
import math
import copy
class BabsiFunctions:
    """
    This class contains side functionality is used by entire bunch of Babsi classes
    """
    @classmethod
    def get_relevance_indexes(cls, relevance_matrix):
        """
        Method sorts the retrieved list of feature weights
        """
        return np.flip(np.argsort(cls.get_attribute_weights(relevance_matrix=relevance_matrix)))
    
    @classmethod
    def get_attribute_weights(cls, relevance_matrix):
        """
        Method retrieves a diagonal of GMLVQ matrix
        """
        return np.diag(relevance_matrix)
    
    @classmethod
    def set_math_inf_frames(cls, frames):
        """
        For correct JSON generation infinity based values are replaced by specific strings
        """
        if frames[0] == 'neg-inf':
            frames[0] = -math.inf
        if frames[-1] == 'pos-inf':
            frames[-1] = math.inf   
        return frames        
    
    @classmethod
    def set_string_inf(cls, frames):
        """
        For correct deserialization process the strings representing infiity values are converted to floats
        """
        frames_copy = copy.deepcopy(frames)
        if frames_copy[0] == -math.inf:
            frames_copy[0] = 'neg-inf'
        if frames_copy[-1] == math.inf:
            frames_copy[-1] = 'pos-inf' 
        return frames_copy   

    @classmethod
    def get_current_dimension_idx(cls, indices):
        """
        Index of current node dimension is retrieved
        """
        return indices[0].item()   

