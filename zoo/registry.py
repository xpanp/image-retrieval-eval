_model_entrypoints = {}

def register_model(fn):
    print('running register_model(%s)' % fn)
    _model_entrypoints[fn.__name__] = fn
    return fn

def model_entrypoint(model_name):
    return _model_entrypoints[model_name]