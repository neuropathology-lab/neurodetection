from setuptools import setup, find_packages

setup(
    name='neurodetection',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "aicspylibczi>=3.2.1",
        "cellpose>=3.1.1.1",
        "fastai>=2.7.17",
        "imageio>=2.37.0",
        "matplotlib>=3.10.1",
        "numpy>=1.20.0,<2.1",
        "opencv_python>=4.11.0.86",
        "pandas>=2.2.3",
        "Pillow>=11.1.0",
        "scikit_learn>=1.6.1",
        "scipy>=1.15.2",
        "scikit-image>=0.25.2",
        "tqdm>=4.67.1"
    ],
    entry_points={
        'console_scripts': [
            'detect_neurons_tif = neurodetection.detect_neurons_tif:detect_neurons_tif_cli'
        ]
    },
)