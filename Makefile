PYTHON ?= python3

.PHONY: smoke selftest selftest-quick test clean pilot

smoke:
	PYTHON=$(PYTHON) bash run_smoke_test.sh

selftest:
	$(PYTHON) -m fingerprint.selftest

selftest-quick:
	$(PYTHON) -m fingerprint.selftest --quick

test: selftest-quick smoke

# Example pilot run (minutes to hours on a laptop). The beta:radius pairs are
# illustrative only: set them from your own measured scale (e.g. sqrt(t0)).
pilot:
	$(PYTHON) -m fingerprint --sizes 8 12 --betas 2.3 2.4 --n-therm 200 \
	  --n-configs 200 --n-sweeps 10 --n-or 3 --flow-radius 2.3:1.6 2.4:2.0 \
	  --seeds 1 2 3 4 --outdir pilot_output

clean:
	rm -rf smoke_output pilot_output output __pycache__ fingerprint/__pycache__
