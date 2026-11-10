.PHONY: test test-python
test:
	LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --no-default-features

# Requires the built extension installed in the active Python environment.
test-python:
	python -m unittest discover -s tests
