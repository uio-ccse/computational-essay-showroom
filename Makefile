install:
	cd .jupyter-book && make install

clone: 
	git clone https://github.com/jupyter/jupyter-book .jupyter-book
	cd .jupyter-book && git checkout 5bcbf9aafcea177b10377de81778e4d281cba70f && cd ..

build: clone build_noclone

build_noclone:
	rm -rf .jupyter-book/content
	rm -rf .jupyter-book/_build/*
	cp -r content .jupyter-book/
	cp -r assets .jupyter-book/
	cp _includes/* .jupyter-book/_includes/
	cp toc.yml .jupyter-book/_data/
	cp _config.yml .jupyter-book/_config.yml
	cd .jupyter-book && make book

clean: 
	rm -rf .jupyter-book

serve: install 
	cd .jupyter-book && make serve
