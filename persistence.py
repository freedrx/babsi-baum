from babsi import BBTree
import json
import numpy as np
from node import BabsiNode
from functions import BabsiFunctions
from props import GMLVQProperties

class BabsiPersistence:
    """
    This class contains the functionality is used to dump and load BB-Tree models
    """
    @classmethod
    def dump(cls, tree: BBTree, filename:str, path:str):
        with open(f'{path}/{filename}.json', 'w', encoding ='utf8') as json_file:
            json.dump(cls.tree_to_dict(tree=tree), json_file, ensure_ascii = False)
        
        
    @classmethod
    def load(cls, filename:str, path:str):
        with open(f'{path}/{filename}.json', 'r') as json_file:
            tree_dict = json.load(json_file)
        return cls.dict_to_tree(tree_dict)
    
    @classmethod
    def dict_to_tree(cls, tree_dict):
        """
        Based on a given dict the BB-Tree is deserialized
        """
        cls.deserialise_vectors(vectors=tree_dict['GMLVQ_props']['prototypes'])
        tree = BBTree(
            max_depth=tree_dict['max_depth'],
            model_properties=GMLVQProperties(
                prototypes=cls.deserialise_vectors(tree_dict['GMLVQ_props']['prototypes']),
                relevance_matrix=np.array(tree_dict['GMLVQ_props']['relevance_matrix']),
                classes=np.array(tree_dict['GMLVQ_props']['classes']),
                features=tree_dict['GMLVQ_props']['features']
            )
        )

        tree.scaler.mean_ = np.array(tree_dict['scaler_props']['mean'])
        tree.scaler.scale_ = np.array(tree_dict['scaler_props']['scale'])

        tree.root = BabsiNode(
            dimension_idx=tree_dict['tree']['dimension_idx'],
            depth=tree_dict['tree']['depth'],
            is_leaf=False,
            class_percentage_vector=np.array(tree_dict['tree']['vector']),
            frames=BabsiFunctions.set_math_inf_frames(tree_dict['tree']['frames']),
        )
        tree.root.children = cls.children_to_node(tree_dict['tree']['children'], depth=tree.root.depth+1)
        return tree

    @classmethod
    def children_to_node(cls, children, depth):
        """
        Every child dictionary is processed and replaced by Babsi-Node-Object
        """
        children_list = []
        for values in children.values():
            child, relays_count = cls.get_node(values['object'], relays_count_needed=True)
            child_node = BabsiNode(
                depth=depth + relays_count,
                dimension_idx=child['dimension_idx'],
                class_percentage_vector=np.array(child['vector']),
                is_leaf= True if child['type'] == 'leaf' else False,
                mismatched_count=child['mismatched_count']
            )
            if child['type'] == 'node':
                child_node.frames=BabsiFunctions.set_math_inf_frames(child['frames'])
                child_node.children = cls.children_to_node(child['children'], depth=depth+relays_count+1)
            children_list.append({
                'frames': tuple(BabsiFunctions.set_math_inf_frames(values['span'])),
                'node': child_node
            })
        return children_list
            

    @classmethod
    def tree_to_dict(cls, tree: BBTree):
        """
        Based on the given BB-Tree the dictionary is created.
        Every field contains numpy arrays is converted to the list
        """
        return {
            'structure name': 'Babsi Baum',
            'max_depth': tree.max_depth,
            'GMLVQ_props': {
                'prototypes': cls.serialise_vectors(tree.prototypes),
                'relevance_matrix': tree.relevance_matrix.tolist(),
                'classes': list(tree.class_map.values()),
                'features': tree.features
            },
            'scaler_props': {
                'mean': tree.scaler.mean_.tolist(),
                'scale': tree.scaler.scale_.tolist()
            },
            'tree': {
                'type': 'root',
                'vector': tree.root.class_percentage_vector.tolist(),
                'dimension_idx': tree.root.dimension_idx,
                'depth': tree.root.depth,
                'mismatched_count': tree.root.mismatched_count,
                'children': cls.children_to_dict(tree.root),
                'frames': BabsiFunctions.set_string_inf(tree.root.frames)
            }
        }

    @classmethod
    def children_to_dict(cls, node: BabsiNode):
        """
        Each BB-Node is replaced by dictionary and packed in relays if necessary
        """
        children = {}
        
        for i in range(len(node.children)):
            relays = node.children[i]['node'].depth - node.depth - 1
            children[i]= {
                'span': BabsiFunctions.set_string_inf(list(node.children[i]['frames'])),
                'object': cls.generate_node(relays)
            }
            child = cls.get_node(children[i]['object'])
            child.update({
                'vector': node.children[i]['node'].class_percentage_vector.tolist(),
                'dimension_idx': node.children[i]['node'].dimension_idx,
                'mismatched_count': node.children[i]['node'].mismatched_count
            })
            if node.children[i]['node'].is_leaf:
                child.update(type='leaf')
            else:
                child.update({
                    'frames': BabsiFunctions.set_string_inf(node.children[i]['node'].frames),
                    'children': cls.children_to_dict(node.children[i]['node'])
                })
        return children
            

    @classmethod
    def generate_node(cls, relays_amount):
        """
        This method generates node entities and packs them in relay objects 
        """
        if relays_amount > 0:
            return {
                'type': 'relay',
                'child': cls.generate_node(relays_amount=relays_amount-1)
            }
        else:
            return {'type': 'node'}
    
    @classmethod
    def get_node(cls, child, relays_count_needed=False):
        """
        This method ignores relay wrappers and retrieves dictionary with node data
        """
        res_tpl = cls._get_node(child=child,count= 0 if relays_count_needed else None)
        return res_tpl if relays_count_needed else res_tpl[0]


    @classmethod
    def _get_node(cls, child, count=None):
        if child['type'] == 'relay':
            if count is not None:
                count += 1
            return cls._get_node(child=child['child'], count=count)
        else:
            return child, count
        
    @classmethod

    def serialise_vectors(cls, vectors):
        """
        The method serializes numpy vectors
        """
        for i in range(len(vectors)):
            vectors[i]['vector'] = vectors[i]['vector'].tolist()
        return vectors
    
    @classmethod
    def deserialise_vectors(cls, vectors):
        """
        The method deserializes numpy vectors
        """
        for i in range(len(vectors)):
            vectors[i]['vector'] = np.array(vectors[i]['vector'])
        return vectors

