from .registry import dataset_entrypoint

def create_dataset(dataset, datadir, datapth):
    create_fn = dataset_entrypoint(dataset)
    d = create_fn(datadir, datapth)
    return d
