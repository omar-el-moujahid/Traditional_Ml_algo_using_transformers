from pathlib import Path
def get_config():
    return {
        "batch_size": 8 ,
        "num_epochs": 20 ,
        "lr" : 10**-4,
        "lang_src": "en",
        "lang_tgt": "fr",
        "seq_len": 350 , 
        "d_model": 512,
        "src_seq_len": 350,
        "tgt_seq_len": 350,
        "model_folder" : "weights",
        "model_baseename": "tmodel_",
        "preload": None , ## retrun the model if it cruch 
        "tokinezer_file": "tokenizer_(0).json",
        "experiment_name" : "runs/tmodel" # save the losses
    } 
    
def get_weights_file_path(config , epoch : str):
    model_folder=config["model_folder"]
    model_baseename = config["model_baseename"]
    model_filename = f"{model_baseename}{epoch}.pt"
    return str(Path(".")/model_folder/model_filename)

# import transformers
# print(transformers.__version__)