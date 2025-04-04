import pathlib
import platform
from fastai.vision.all import load_learner

platf = platform.system()
if platf == 'Linux': 
    pathlib.WindowsPath = pathlib.PosixPath
elif platf == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

def loadIsNeuron(model_name):
    this_file = pathlib.Path(__file__) 
    # Necessary cause fastai models have pathlib paths embedded in them and they will thrown NotImplementedErrors if a model is loaded on a different platform than it was trained on
    model_path = this_file.parent / "models" / model_name
    model = load_learner(model_path)

    return model

