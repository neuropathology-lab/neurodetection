import pathlib
import platform
from fastai.vision.all import load_learner

platf = platform.system()
if platf == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath
elif platf == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

def load_is_neuron(model_name):
    this_file = pathlib.Path(__file__) 
    # Necessary cause fastai models have pathlib paths embedded in them and they will thrown NotImplementedErrors if a model is loaded on a different platform than it was trained on
    is_neuron_model_path = this_file.parent.parent / "models" / model_name
    is_neuron_model = load_learner(is_neuron_model_path)
    return is_neuron_model

