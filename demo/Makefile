TARGET := cache.out chrono.out monitor.out mplext.out quickui.out
PYTHON := /opt/Tools/anaconda/3/bin/python
PYTHONPATH := /home/andreoli/src/distrib

all: ${TARGET}

%.out: %.py
	export PYTHONPATH=${PYTHONPATH} ; ${PYTHON} make.py $(patsubst %.py,%,$<)
