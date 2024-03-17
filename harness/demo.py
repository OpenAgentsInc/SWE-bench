from .dataset import get_dataset
from .scripts import make_test_spec

dataset = get_dataset()
test1 = make_test_spec(dataset[0])
print(test1)
