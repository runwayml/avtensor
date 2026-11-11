.PHONY: test
# Requires the current extension built and installed in the active Python environment.
test:
	set -e; \
	torch_lib_dir="$$(python -c 'from pathlib import Path; import torch; print(Path(torch.__file__).parent / "lib")')"; \
	export LD_LIBRARY_PATH="$$torch_lib_dir$${LD_LIBRARY_PATH:+:$$LD_LIBRARY_PATH}"; \
	LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --no-default-features; \
	python -m unittest discover -s tests
