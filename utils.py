import json

def load_model(model_name):
    if model_name=="timexlstm":
        from models.timexlstm import xLSTMLit as model
    elif model_name=="itransformer":
        from models.itransformer import iTransformerLit as model
    elif model_name=="samformer":
        from models.samformer import SAMformerLit as model
    elif model_name=="timemixer":
        from models.timemixer import TimeMixerLit as model
    elif model_name=="vaformer":
        from models.vaformer import VAformerLit as model
    elif model_name=="gpt4ts":
        from models.gpt4ts import gpt4tsLit as model
    elif model_name=="moment":
        from models.moment import MomentLit as model
    elif model_name=="toto":
        from models.toto import TotoLit as model
    elif model_name=="patchtst":
        from models.patchtst import PatchTSTLit as model
    return model

def load_results(filename="mse.json"):
    with open(filename, 'r') as f:
        results = json.load(f)
    return results

def save_results(filename, dataset, context_horizon, target_horizon, model, score):
    results = load_results(filename)
    
    results.setdefault(dataset, {})
    results[dataset].setdefault(str(context_horizon), {})
    results[dataset][str(context_horizon)].setdefault(str(target_horizon), {})
    results[dataset][str(context_horizon)][str(target_horizon)][model] = score

    with open(filename, "w") as f:
        json.dump(results, f, indent=2)