import pandas as pd
import matplotlib.pyplot as plt
from aicspylibczi import CziFile
from utils import convertCziToUsableArray

def plotPositivesOnImage(czi_path, output_csv_file):
    czi = CziFile(str(czi_path))
    output_df = pd.read_csv(output_csv_file)
    img = convertCziToUsableArray(czi)

    output_df = output_df[output_df["is_neuron"] == "Positive"]

    plt.imshow(img)
    plt.scatter(output_df["center_col"], output_df["center_row"], color ='r')
    plt.show()
        
if __name__ == '__main__':
    czi_path = ""

    df = "./out_dir/UK2018_ah_1_classified_objects.csv"

    plotPositivesOnImage(czi_path, df)
