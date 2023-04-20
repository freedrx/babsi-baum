import math
import numpy as np
from functions import BabsiFunctions
from node import BabsiNode
import copy
from props import GMLVQProperties
from typing import List
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
class BBTree(BaseEstimator, TransformerMixin):
    """
    This class contains implementation of BB-Tree algorithm
    """
    def __init__(self, max_depth: int, model_properties: GMLVQProperties):
        if max_depth < 1:
            raise ValueError('Max depth greater than 0 should be specified')
        if (model_properties.relevance_matrix != model_properties.relevance_matrix.T).all():
            raise ValueError('Relevance matrix should be symmetric')
        if len(model_properties.classes) < 2:
            raise ValueError('At least 2 unique classes should be represented')
        for feature_name in model_properties.features:
            if any(not c.isalnum() for c in feature_name.replace(' ', '')):
                raise TypeError('Feature names must contain alphanumeric chars and blank spaces solely')
        self.max_depth = max_depth
        
        self.relevance_matrix  = model_properties.relevance_matrix
        self.prototypes = model_properties.prototypes

        #for both class list and feature list the mappings are created. It facilitates the training process
        self.class_map = {i:model_properties.classes[i] for i in range(len(model_properties.classes))}
        self.features = model_properties.features
        
        self.scaler = StandardScaler()
        self.root = None
    
    def fit(self, entries, labels, decimal_places=None):
        """
        The input data is checked and normalized. Besides it is allowed to pick necessary amount of decimal places for generated rules
        feauture frames are defined and root node is created
        """
        if decimal_places is not None and (decimal_places < 0 or type(decimal_places) != int):
            raise TypeError('Decimals have to be integers greater than zero')
        if entries.shape[1] != self.relevance_matrix.shape[1]:
            raise TypeError('entries do not correspond to relevance matrix')
        if (set(BabsiNode.get_unique_labels(labels=labels)) != set(self.class_map.values())):
            raise TypeError('All fixed classes have to be represented in input data')
        if not np.issubdtype(entries.dtype, np.number):
            raise TypeError('BB Tree supports numeric data only')
        entries=self.scaler.fit_transform(entries)

        # points of specific feature where feature space is fractured
        feature_frames = {f'frame({i})':[-math.inf, math.inf] for i in range(entries.shape[1])}
        
        #min and max elements of each feature in dataset
        thresholds = [{'min': v1, 'max': v2} for v1, v2 in zip(np.amin(entries, axis=0), np.amax(entries, axis=0))]
        
        # generating of root object
        self.root = self._build_node(
            entries=entries, 
            labels=self.map_labels(labels=labels), 
            prototypes=self.prototypes, 
            relevance_indices=BabsiFunctions.get_relevance_indexes(self.relevance_matrix), 
            feature_frames=feature_frames, 
            thresholds=thresholds, 
            depth=0,
            decimal_places=decimal_places
        )

    def _build_node(self, entries, labels, prototypes, relevance_indices, feature_frames, thresholds, depth, decimal_places):

        while(True):
            node = BabsiNode(
                depth=depth,
                dimension_idx= BabsiFunctions.get_current_dimension_idx(relevance_indices),
                frames=feature_frames[f'frame({BabsiFunctions.get_current_dimension_idx(relevance_indices)})'].copy()
            )
            if not self.check_stop_criteria(node, labels):
                if len(prototypes) > 1:
                    node.define_frames(prototypes)
                # when feature space is not fractured the method creates no node for current feature span 
                # and retrieves another feature to analyse
                if len(node.frames) == 2:
                    relevance_indices = np.roll(relevance_indices, -1)
                    depth = depth + 1 if depth > 0 else depth
                    continue
                # when desired amount of decimal places is gotten
                # the feature frames are transormed
                if decimal_places is not None:
                    new_frames = []
                    new_frames.append(node.frames[0])
                    for frame in node.frames[1:-1]:
                        if frame not in [math.inf, -math.inf]:
                            temp_sample = np.zeros(len(self.features))
                            temp_sample[node.dimension_idx] = frame
                            inverted = round(self.scaler.inverse_transform(np.array([temp_sample]))[0][node.dimension_idx], decimal_places)
                            temp_sample[node.dimension_idx] = inverted
                            new_frames.append(self.scaler.transform(np.array([temp_sample]))[0][node.dimension_idx])
                    new_frames.append(node.frames[-1])
                    node.frames = new_frames
                
                # for each feautre fraction the child node is initialised
                for i in range(len(node.frames) - 1):
                    thresholds_to_update = copy.deepcopy(thresholds)
                    frames_to_update = copy.deepcopy(feature_frames) 
                    frames_to_update[f'frame({node.dimension_idx})'] = [node.frames[i], node.frames[i+1]]
                    
                    # guillotine cut for getting data is located in specified feature fraction
                    # and prototypes are representing these data points
                    new_entries, new_labels, active_prototypes = self.guillotine_cut(
                        inference_data=entries, 
                        labels=labels,
                        prototypes=prototypes,
                        dimension=node.dimension_idx,
                        min_frame=node.frames[i], 
                        max_frame=node.frames[i+1], 
                        relevance_matrix=self.relevance_matrix, 
                        thresholds=thresholds_to_update
                    )
                    if new_entries.shape[0] > 0:
                        
                        new_props = {
                            'relevance_indices': copy.deepcopy(np.roll(relevance_indices, -1)),
                            'feature_frames': frames_to_update,
                            'thresholds': thresholds_to_update,
                            'depth': node.depth + 1,
                        }
                        node.children.append(
                            {
                                'frames': (node.frames[i], node.frames[i+1]),
                                'node': self._build_node(new_entries, new_labels, active_prototypes, **new_props, decimal_places=decimal_places)
                            }
                        )
                # if no child nodes are generated the corresponding node is converted to the leaf
                if len(node.children) == 0:
                    node.toggle_leaf()
                # parent node is replaced by child node when parent node has only one child in its list
                elif len(node.children) == 1:
                    node = node.children[0]['node']
                break
            else:
                break
        return node
        

    @classmethod
    def guillotine_cut(cls, inference_data, labels, prototypes, dimension, min_frame, max_frame, thresholds, relevance_matrix):
        # function represents the main functionality of guillotine cut
        # dataset is splitted in smaller sets for each child node
        filter = (inference_data[:, dimension] > min_frame) & (inference_data[:, dimension] <= max_frame)
        new_entries = inference_data[filter]
        new_labels = labels[filter]
        if new_entries.shape[0] > 0:
            thresholds[dimension]['min'] = new_entries[:, dimension].min()
            thresholds[dimension]['max'] = new_entries[:, dimension].max()

            distances = [
                [((entry - prototype['vector'])
                        .dot(relevance_matrix.T)
                        .dot(relevance_matrix)
                        .dot(np.array([entry - prototype['vector']]).T))
                        .item()
                for prototype in prototypes] 
            for entry in new_entries]

            prototype_indices = list(set([dist_item.index(min(dist_item)) for dist_item in distances]))
            return new_entries, new_labels, [prototypes[i] for i in prototype_indices]
        return new_entries, new_labels, prototypes

    def check_stop_criteria(self, node, labels):
        node.majority_criteria(labels, self.class_map)
        # check whether all labels in specific node are instances of solely one class
        if node.mismatched_count == 0:
            self.stop()
            return True
        
        # depth check
        if node.depth >= self.max_depth:
            self.stop()
            return True
        node.toggle_leaf()
        return False
    
    def stop(self):
        pass

    def predict(self, inference_data: np.ndarray):
        # function returns labels of inference data
        outputs = self.get_outputs(node=self.root, inference_data=self.scaler.transform(inference_data), expected_output='labels')
        return self.map_indices(outputs.flatten())
    
    def predict_proba(self, inference_data: np.ndarray):
        # function returns distribution vectors for each inference data point
        return self.get_outputs(node=self.root, inference_data=self.scaler.transform(inference_data), expected_output='vectors')
    
    def explain(self, inference_data):
        # function retrieves both labels and explanation to the model decisions
        predictions, explanations = self.get_outputs(node=self.root, inference_data=self.scaler.transform(inference_data), expected_output='labels', explainable=True)
        return self.map_indices(predictions.flatten()), [rule_list[0].split('$$$$$') for rule_list in explanations.tolist()]

    def get_outputs(self, node: BabsiNode, inference_data: np.ndarray, expected_output, explainable=False):
        if expected_output not in ['labels', 'vectors']:
            raise ValueError('Inappropriate output type')
        if inference_data.shape[1] != len(self.features):
            raise ValueError('Shapes do not match')
        if inference_data.ndim == 1:
            inference_data = np.array([np.array(inference_data)])
        if node.is_leaf:
            #the predicitons are done in the case current node is a leaf
            res = (
                np.full((inference_data.shape[0],1), np.argmax(node.class_percentage_vector)), 
                None
            ) if expected_output == 'labels' else (
                np.tile(node.class_percentage_vector, (inference_data.shape[0],1)),
                None
            )
            return res[0] if not explainable else res
        else:
            indices = []
            cuts = []
            if explainable:
                explanations = []

            for i in range(len(node.frames) -1):
                # for feauture space fraction the presence of corresponding node is examined
                inferior_node = None
                filter = (inference_data[:, node.dimension_idx] > node.frames[i]) & (inference_data[:, node.dimension_idx] <= node.frames[i+1])
                sample_indices = np.where(filter)[0]
                if sample_indices.shape[0] == 0:
                    cuts.append(np.array([]))
                    if explainable:
                        explanations.append(np.array([]))
                    continue
                indices.append(sample_indices)
                for child in node.children:
                    if (node.frames[i], node.frames[i+1]) == child['frames']:
                        inferior_node = child
                        break
                if inferior_node is not None:
                    # for found inferior node the recursion step is done
                    res = self.get_outputs(inferior_node['node'], inference_data[filter], expected_output=expected_output, explainable=explainable)
                    if explainable:
                        # explanations for current dimension are created
                        explanations.append(
                            np.tile(
                                self.create_rule(
                                    frames=(node.frames[i], node.frames[i+1]), 
                                    dim_idx=node.dimension_idx
                                ), 
                                (len(indices[-1]), 1)
                            )
                        )
                        # both predictions and explanations are put to the general list for the whole bunch of points
                        cut, explanation_block = res
                        cuts.append(cut)
                        if explanation_block is not None:
                            explanation_block = np.core.defchararray.add('$$$$$', explanation_block)
                            explanations[i] = np.core.defchararray.add(explanations[i], explanation_block)
                    else:
                        cuts.append(res)
                else:
                    if explainable:
                        explanations.append(
                            np.tile(
                                self.create_rule(
                                    frames=(node.frames[i], node.frames[i+1]), 
                                    dim_idx=node.dimension_idx
                                ), 
                                (len(indices[-1]), 1)
                            )
                        )

                    cuts.append(
                        np.full((inference_data[filter].shape[0],1), np.argmax(node.class_percentage_vector))
                    if expected_output == 'labels' else 
                        np.tile(node.class_percentage_vector, (inference_data[filter].shape[0],1))
                    )
            # because the whole dataset is splitted in multiple parts it is 
            # necessary to merge it based on their indices

            cuts = [array for array in cuts if array.shape[0] > 0]
            flat_indices = [item for sublist in indices for item in sublist]
            flat_cuts = np.concatenate(cuts)
            sorted_cuts = flat_cuts[np.argsort(flat_indices)]
            # taking indices into account the explanations are merged
            if explainable:
                explanations = [array for array in explanations if array.shape[0] > 0]
                flat_explanations = np.concatenate(explanations)
                sorted_explanations = flat_explanations[np.argsort(flat_indices)]
            return (sorted_cuts, sorted_explanations) if explainable else sorted_cuts    

    def score(self, inference_data, labels):
        # function for validation
        if inference_data.shape[1] != self.relevance_matrix.shape[1]:
            raise TypeError('entries do not correspond to relevance matrix')
        res = self.map_indices(np.argmax(self.predict_proba(inference_data), axis=1))
        accuracy = res[res == labels].shape[0]/res.shape[0]
        return accuracy


    def index_to_class(self, idx):
        return self.class_map[idx]
    
    def index_to_feature(self, idx):
        return self.features[idx]
    
    def map_labels(self, labels: NDArray):
        new_labels = np.zeros(labels.shape, dtype=int)
        for k,v in self.class_map.items():
            new_labels[np.where(labels == v)] = k
        return new_labels
    
    def map_indices(self, indices):
        new_indices = np.zeros(indices.shape, dtype=object)
        for k,v in self.class_map.items():
            new_indices[np.where(indices == k)] = v
        return new_indices        
    
    def __str__(self) -> str:
        return 'BBTree object'

    def get_node_count(self):
        if self.root is None:
            raise ValueError('Tree has not been trained')
        return self._node_count(self.root.children) + 1

    def _node_count(self, children):
        counter = 0
        for child in children:
            if not child['node'].is_leaf:
                counter += self._node_count(child['node'].children)
        counter += len(children)
        return counter

    def create_rule(self, frames, dim_idx):
        # function generates rules based on given frame range
        output = []

        for frame in frames:
            if frame in [math.inf, -math.inf]:
                output.append(frame)
            else:
                temp_sample = np.zeros(len(self.features))
                temp_sample[dim_idx] = frame
                output.append(self.scaler.inverse_transform(np.array([temp_sample]))[0][dim_idx]) 
        return f'Value of {self.index_to_feature(dim_idx)} is in range from {output[0]} to {output[1]}'