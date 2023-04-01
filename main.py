from babsi import BBTree
from persistence import BabsiPersistence
from props import GMLVQProperties
from data import BabsiDataset
from builder import LVQBuilder
from persistence import BabsiPersistence
from sklearn.preprocessing import StandardScaler

JSON_PATH = 'There should be your specified path'
JSON_FILENAME = 'There should be your specified filename'
CSV_PATH = ''
CSV_FILENAME = ''

X, y, features = BabsiDataset.load_csv(
    path=CSV_PATH, 
    filename=CSV_FILENAME,
    delimiter=',',
    header_present=True,
    label_attr_name='label'
)

model = LVQBuilder(
    lvq_type='GMLVQ', 
    prototypes_per_class=2
).build().train(X, y).get_model()

properties = GMLVQProperties.build_from_GMLVQ(model=model, features=features)

tree = BBTree(
    max_depth=4,
    model_properties=properties
)

tree.fit(X, y)
labels, explanation = tree.explain(X)
score = tree.score(X, y)

BabsiPersistence.dump(path=JSON_PATH, filename=JSON_FILENAME, tree=tree)
new_tree = BabsiPersistence.load(path=JSON_PATH, filename=JSON_FILENAME)

