run:
	@python main.py

format:
	black .

download:
	mkdir -p Datasets/PneumoniaMNIST
	mkdir -p Datasets/PathMNIST
	cd Datasets/PneumoniaMNIST && wget https://zenodo.org/records/6496656/files/pneumoniamnist.npz
	cd Datasets/PathMNIST && wget https://zenodo.org/records/6496656/files/pathmnist.npz

create-env:
	conda env create -f environment.yml

update-env:
	conda install --file ./requirements.txt
	conda env export --from-history > environment.yml

clean-all: clean-datasets clean

clean-datasets:
	rm -rf Datasets/*

clean:
	rm -rf A/__pycache__
	rm -rf B/__pycache__
