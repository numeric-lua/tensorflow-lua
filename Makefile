.PHONY: test lint bench fetch clean
test:
	./scripts/test.sh
lint:
	./scripts/lint.sh
bench:
	./scripts/bench.sh
fetch:
	./scripts/fetch_libtensorflow.sh
clean:
	rm -rf third_party/libtensorflow tests/.tmp
