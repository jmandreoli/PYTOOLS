QTUICBIN = /opt/MLS_fast/anaconda/envs/py3k/bin/pyuic4

# Edit UI design files (extension .ui) with
# designer <filename> in same directory as $QTUICBIN

all: $(addsuffix ui.py,$(basename $(wildcard *.ui))) $(addsuffix ui.py,$(basename $(wildcard plugins/*/*.ui)))
%ui.py: %.ui
	${QTUICBIN} -o $@ $<
