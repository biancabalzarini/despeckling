conda update -n base -c defaults conda
conda env create -f environment.yml
source activate despeckling
python -m pip install -r requirements.txt
pip install wheel
pip install --only-binary :all: pyreadr