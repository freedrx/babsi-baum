import numpy as np

class BabsiNode:
    """
    The class contains basic functionality of BB-Tree node objects
    """
    def __init__(self, depth, dimension_idx, frames=None, is_leaf=True, class_percentage_vector=None, mismatched_count=0) -> None:
        
        self.depth = depth
        self.dimension_idx = dimension_idx
        self.frames = frames
        self.is_leaf = is_leaf
        self.mismatched_count = mismatched_count
        self.children = [] if not is_leaf else None
        self.class_percentage_vector = class_percentage_vector
    
    def toggle_leaf(self):
        # the method switches node to leaf state and vice versa
        if self.is_leaf:
            self.is_leaf = False
            self.children = []
        else:
            self.is_leaf = True
            self.children = None

    def update_dimension(self, new_idx):
        # update of dimension
        if new_idx:
            self.dimension_idx = new_idx
    
    def update_frames(self, new_frames):
        # update of feature space fractions
        if new_frames:
            self.frames = new_frames

    def majority_criteria(self, labels, class_map):
        # the amount of each class samples are checked and saved
        class_entries = self.get_class_entries(labels, class_map)
        #distribution vector is defined
        self.class_percentage_vector = np.array([v/sum(class_entries.values()) for v in class_entries.values()])
        
        u, i = np.unique(self.class_percentage_vector, return_inverse=True)
        # count of mismatched elements is found
        if u[np.bincount(i) > 1].shape[0] > 0:
            self.mismatched_count = sum(class_entries.values())
        else:
            self.mismatched_count = labels.shape[0] - class_entries[np.argmax(self.class_percentage_vector)]

    @classmethod
    def get_unique_labels(cls, labels):
        return np.unique(labels)

    @classmethod
    def get_class_entries(cls, labels, class_map):
        # the method returns amount of entrances for each class label
        return {class_idx: (0 if class_idx not in cls.get_unique_labels(labels) else np.count_nonzero(labels == class_idx)) for class_idx in class_map}

    def define_frames(self, prototypes):
        """
        The method defined points in feature space where the fractions need to be done
        """
        # prototypes are sorted ascending
        sorted_prototypes = self.sorted_prototypes_by_dimension(prototypes, self.dimension_idx)
        # mean distance between two arbitrary prototypes is calculated
        mean_dist = (
            sorted_prototypes[-1]['vector'][self.dimension_idx] - sorted_prototypes[0]['vector'][self.dimension_idx])/(len(sorted_prototypes) - 1)
        new_frames = []

        # two neighbor prototypes are picked
        # when distance between them is greater as average distance of the bunch of prototypes
        # the fraction is done in the middle of their distance
        for i in range(1, len(sorted_prototypes) -1):
            down = sorted_prototypes[i-1]['vector'][self.dimension_idx]
            up = sorted_prototypes[i]['vector'][self.dimension_idx]
            mean = (up+down)/2
            if up-down >= mean_dist and mean > self.frames[0] and mean < self.frames[1]:
                new_frames.append(mean)

        # fractions are updated
        for i in range(len(new_frames)):
            self.frames.insert(i + 1, new_frames[i])

    @classmethod
    def sorted_prototypes_by_dimension(self, prototypes, dimension):
        #method sorts prototypes taking into account specific dimension
        return sorted(prototypes, key=lambda d: d['vector'][dimension]) 