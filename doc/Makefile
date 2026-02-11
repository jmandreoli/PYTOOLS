# Creation date:        2022-01-15
# Contributors:         Jean-Marc Andreoli
# Language:             python
# Purpose:              Doc generation script
#
# Fill in the following variables and save as Makefile. Alternatively
# you can set these variables from the command line, and also
# from the environment for the first two.
#
SPHINXOPTS    ?=
SPHINXBUILD   ?=
SOURCEDIR     = .
BUILDDIR      = _build
# needed for _resource
RSCMAKE       =
SOFFICE       = soffice
#
# There should be no need to change below this line
#--------------------------------------------------------------------------------------------------
override SPHINXOPTS    = -T # local
override SPHINXBUILD   = /home/andreoli/Tools/anaconda/bin/sphinx-build # local
override RSCMAKE       = /home/andreoli/Tools/anaconda/bin/python -Wignore -m myutil.make # local
#
# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
#
.PHONY: help Makefile
#
RSCDIR = $(BUILDDIR)/resource
rsc: $(addprefix $(RSCDIR)/mod_,cache.out cache-timeline.png cache-workflow.png polling.out html.out) ;
#
$(RSCDIR)/%.out:: %.py
	@echo Generating $@
	@mkdir -p $(RSCDIR)
	@export PYTHONPATH=$(realpath ../.pythonpath):$(PYTHONPATH); ${RSCMAKE} $< $@
#
$(RSCDIR)/%.png:: %.odg
	@echo Generating $@ ...
	@mkdir -p $(RSCDIR)
	@${SOFFICE} --headless --convert-to png --outdir $(RSCDIR) $<
#
MAKEFILE:: Makefile                                # local
	@echo Generating $@ ...                          # local
	@sed -e 's/^.*\# local$$//g' -e '/^$$/d' $< > $@ # local

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
