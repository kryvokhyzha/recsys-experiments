install_packages: requirements.txt
	conda install -y -c conda-forge tensorflow==2.7.0
	conda install -y -c conda-forge lightfm==1.16
	conda install -y -c conda-forge umap-learn
	conda install -y -c conda-forge jupyterlab
	pip install -r requirements.txt
