.PHONY: test fetch clean
test:
	./scripts/test.sh
fetch:
	./scripts/fetch_libtensorflow.sh
clean:
	rm -rf third_party/libtensorflow tests/.tmp
