<!-- install libs  -->
# Sphinx
pip3 install myst-parser pandoc sphinx-rtd-theme sphinxcontrib-napoleon nbsphinx sphinx 
# pypi
python3 -m pip install --upgrade twine build


# Package Deploy
## Test Pypi
### build
python3 -m build
### test deploy
python3 -m twine upload --repository testpypi dist/*
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-package-YOUR-USERNAME-HERE

# Prod deploy
python3 -m twine upload --repository pypi dist/*
python3 -m pip install pyqsm


download pandocs
https://pandoc.org/installing.html
install pandocs
sudo dpkg -i <deb_file_location>

create docs folder 
mkdir docs 

generate sphinx process files 
sphinx-generate 

deploy package to pypi
https://packaging.python.org/en/latest/tutorials/packaging-projects/

generate docs for package 
sphinx-apidoc -o docs/source pyQSM 

generate html files 
sphinx-build -M html docs/source docs/build

Test locally 
sphinx-autobuild  docs/source docs/build
