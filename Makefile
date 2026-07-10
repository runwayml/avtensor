.PHONY: test
test:
	LIBTORCH_BYPASS_VERSION_CHECK=1 cargo test --no-default-features
