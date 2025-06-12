import pathlib
import platform
from fastai.vision.all import load_learner

# Detect operating system platform
platf = platform.system()

# Patch incompatible pathlib types for cross-platform model loading
if platf == 'Linux':
    pathlib.WindowsPath = pathlib.PosixPath
elif platf == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath


def loadIsNeuron(model_name):
    """
    Loads a FastAI-trained model from the 'models' directory.
    """

    this_file = pathlib.Path(__file__)

    # Construct path to model file
    model_path = this_file.parent / "models" / model_name

    # Load the model
    model = load_learner(model_path)

    return model