.PHONY: all build setup preprocess train validate test sample

version ?= default

all: build

build:
	bash ./hack/build.sh $(version)

setup:
	bash ./hack/setup.sh

preprocess:

train:

validate:

test:

sample:
