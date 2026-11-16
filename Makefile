.PHONY: test lint fetch clean
test:
	./scripts/test.sh
lint:
	./scripts/lint.sh
fetch:
	./scripts/fetch_libtensorflow.sh
clean:
	rm -rf third_party/libtensorflow tests/.tmp
