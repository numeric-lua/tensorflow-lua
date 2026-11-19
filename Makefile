.PHONY: test test-fetch lint bench package fetch clean
test: test-fetch
	./scripts/test.sh
test-fetch:
	./tests/test_fetch_libtensorflow.sh
lint:
	./scripts/lint.sh
bench:
	./scripts/bench.sh
package:
	./scripts/package.sh
fetch:
	./scripts/fetch_libtensorflow.sh
clean:
	rm -rf third_party/libtensorflow tests/.tmp
