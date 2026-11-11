.PHONY: test
# Requires the current extension built and installed in the active Python environment.
test:
	LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --no-default-features
	python -m unittest discover -s tests
