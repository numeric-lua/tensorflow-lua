.PHONY: test lint bench package fetch clean
test:
	./scripts/test.sh
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
