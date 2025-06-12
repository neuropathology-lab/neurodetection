from setuptools import setup, find_packages

setup(
    name='neurodetection',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'neurodetection': ['models/isneuron_hematoxylin.pkl', 'models/isneuron_ptdp.pkl']
    },
    install_requires=[
        "aicspylibczi==3.2.1",
        "cellpose==3.1.1.1",
        "fastai>=2.7.17,<3.0",
        "imageio>=2.37.0,<3.0",
        "matplotlib>=3.10.1,<4.0",
        "numpy>=1.24,<2.0",
        "opencv-python>=4.11.0.86,<5.0",
        "pandas>=2.2.3,<3.0",
        "Pillow>=7.1.0,<11.0",
        "scikit-learn>=1.6.1,<1.7",
        "scipy>=1.7.0,<1.14.0",
        "scikit-image>=0.25.2,<1.0",
        "tqdm>=4.67.1,<5.0",
        "streamlit>=1.37.1,<2.0",
        "gensim>=4.3.3,<5.0"
    ],
    entry_points={
        'console_scripts': [
            'detectNeurons = neurodetection.detectNeurons:detectNeurons_cli'
        ]
    },
    author_email='neuropathology.kuleuven@outlook.com',
    description='Detects pyramidal neurons in IHC/HE stained human brain tissue sections.',
    url='https://github.com/neuropathology-lab/neurodetection',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)