.ONESHELL:
SHELL := /bin/bash

all: release
.PHONY: all release pypi clean scrub

release: pypi

pypi: dist
	uv publish

dist: clean
	uv build

clean:
	rm -rf dist

scrub:
	ruff format
	nbdev-clean --fname notebooks


