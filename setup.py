from setuptools import find_packages, setup

with open('README.md', 'r') as f:
  readme_content = f.read()
  
  
setup(
  name = "misato-dataset",
  version = "0.0.7",
  description="UNOFFICIAL Misato dataset pypi package. For instructions on dataset download see official GitHub page (https://github.com/t7morgen/misato-dataset).",
  package_dir={"": "src"},
  packages=find_packages(where="src"),
  long_description=readme_content,
  long_description_content_type="text/markdown",
  url="https://github.com/t7morgen/misato-dataset",
  
  author="Jean Charle Yaacoub", 
  author_email="jyaacoub21@gmail.com",

  license="LGPL-2.1",
  classifiers=[
      "Programming Language :: Python :: 3.10",
      "License :: OSI Approved :: GNU Lesser General Public License v2 (LGPLv2)",
      "Operating System :: OS Independent",
      ],
  install_requires=[
      "matplotlib",
      "h5py",
      "pandas",
      "numpy",
      "torch",
      "torch_geometric",
      "torch_scatter",
      "torch_sparse",
      "pytorch-lightning",
      "joblib",
      
      # "matplotlib==3.7.2",
      # "h5py==3.8.0",
      # "pandas==2.1.0",
      # "numpy==1.25.2",
      # "torch==2.0.1",
      # "torch_geometric==2.4.0",
      # "torch_scatter==2.1.1",
      # "torch_sparse==0.6.17",
      # "pytorch-lightning>=1.8",
      # "joblib==1.3.2",
  ],
  extra_require={
      "dev": ['twine>=4.0.2']
    },
  python_requires=">=3.10"
)
