from .registry import model_entrypoint

def create_model(args, method, model, ckp):
    create_fn = model_entrypoint(method)
    model = create_fn(args, model, ckp)
    return model
