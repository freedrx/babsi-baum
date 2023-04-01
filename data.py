import pandas as pd
from sklearn.preprocessing import StandardScaler

class BabsiDataset:
    """
    The class is responsible for retrieving data from specific sources and defining of feature labels
    """
    @classmethod
    def load_csv(cls, path, filename, delimiter=',', header_present=True, is_scaled=False, label_attr_name=None):
        if filename[-3:] != 'csv':
            raise TypeError('filename is not a csv file')
        if label_attr_name and not header_present:
            raise TypeError('Not possible to extract label column name by having no attributes')
        data = pd.read_csv(f'{path}/{filename}', delimiter=delimiter, header=0 if header_present else None)
        
        if data.shape[0] == 0:
            raise ValueError('csv file is empty')
        
        if header_present:
            if label_attr_name is None:
                raise ValueError('label column name is not specified')

            if label_attr_name not in data.columns:
                raise ValueError(f'csv file has no {label_attr_name} column')
            label = data.pop(label_attr_name)
        else:
            label = data.pop(len(data.columns) - 1)
            
        attributes = [i if type(i) == str else str(i) for i in list(data.columns)]
        X = data.to_numpy()
        del data

        y = label.to_numpy()
        del label

        if not is_scaled:
            X = StandardScaler().fit_transform(X)
        
        return X, y, attributes