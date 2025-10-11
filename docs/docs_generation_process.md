<!-- install libs  -->

pip3 install myst-parser pandoc sphinx-rtd-theme sphinxcontrib-napoleon nbsphinx sphinx build twine


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
