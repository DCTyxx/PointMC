.PHONY: clean build

build: pykdtree/kdtree.c pykdtree/_kdtree_core.c
	python setup.py build

pykdtree/kdtree.c: pykdtree/kdtree.pyx
	cython pykdtree/kdtree.pyx

clean:
	rm -rf build/
	rm -f pykdtree/*.so
