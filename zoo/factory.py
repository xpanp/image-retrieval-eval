from .registry import model_entrypoint

def create_model(method, model, ckp):
    create_fn = model_entrypoint(method)
    model = create_fn(model, ckp)
    return model
