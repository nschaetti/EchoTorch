

# Imports
import torch
import echotorch

# data_indexer = echotorch.DataIndexer(["f1", "f2"])
# print(data_indexer)
# print("keys: {}".format(data_indexer.keys))
# print("Indices: {}".format(data_indexer.indices))
# print("")
#
# print("to_index('f1'): {}".format(data_indexer.to_index('f1')))
# print("to_index(['f1', 'f2']): {}".format(data_indexer.to_index(['f1', 'f2'])))
# print("to_index('test1': 'f1', 'test2': 'f2'): {}".format(data_indexer.to_index({'test1': 'f1', 'test2': 'f2'})))
# print("to_index(tuple('f1', f2')): {}".format(data_indexer.to_index(slice('f1', 'f2'))))
# print("")
#
# print("to_keys(0): {}".format(data_indexer.to_keys(0)))
# print("to_keys([0, 1]): {}".format(data_indexer.to_keys([0, 1])))
# print("to_keys('test1': 0, 'test2': 1): {}".format(data_indexer.to_keys({'test1': 0, 'test2': 1})))
# print("")

data_tensor = echotorch.DataTensor(torch.randn(20, 2, 3), [None, ['f1', 'f2'], None])
print(data_tensor)
print(data_tensor[:, ['f1'], :])
print(data_tensor[0])
print(data_tensor[[0, 1, 2]])
print(data_tensor[:, 'f1', :])
print(data_tensor[:, 0, :])
print(data_tensor[:, ['f1', 'f2']])
print(data_tensor[:, [0, 'f2']])
print(data_tensor[:, ['f1', 'f1']])
