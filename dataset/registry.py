_dataset_entrypoints = {}

def register_dataset(fn):
    print('running register_dataset(%s)' % fn)
    _dataset_entrypoints[fn.__name__] = fn
    return fn

def dataset_entrypoint(dataset):
    return _dataset_entrypoints[dataset]