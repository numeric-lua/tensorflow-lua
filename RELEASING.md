# Releasing tensorflow-lua

Release engineering is coordinator-authorized. Do not tag, publish, or change
package metadata without explicit approval.

## Preflight

1. Bump `tf.VERSION`, the API test, CI package assertion, README status, and
   `docs/api.md` version.
2. Update `CHANGELOG.md` and `RELEASE_NOTES.md`.
3. Run:

   ```sh
   TENSORFLOW_ROOT="$PWD/third_party/libtensorflow" ./scripts/test.sh
   ./scripts/lint.sh
   ```

4. Confirm the GitHub Actions matrix is green on `main` (Ubuntu
   22.04/24.04 × libtensorflow 2.16.2/2.18.0).
5. Verify a clean install outside the checkout:

   ```sh
   luarocks install --tree=/tmp/tf-release --deps-mode=none tensorflow-lua <version>
   ```

## Artifacts

1. Add `tensorflow-lua-<version>-1.rockspec` from the current SCM rockspec,
   with `source.tag = "v<version>"`.
2. Verify the versioned rockspec builds:

   ```sh
   ./scripts/package.sh tensorflow-lua-<version>-1.rockspec
   ```

3. Commit and push the release artifacts; wait for CI.
4. Create and push the annotated tag:

   ```sh
   git tag -a v<version> -m "tensorflow-lua <version>"
   git push origin v<version>
   ```

5. Pack and test the source rock:

   ```sh
   luarocks pack tensorflow-lua-<version>-1.rockspec
   luarocks install --tree=/tmp/tf-release --deps-mode=none \
     ./tensorflow-lua-<version>-1.src.rock
   ```

## Publish

1. Create the GitHub release with the rockspec and source rock:

   ```sh
   gh release create v<version> \
     --title "tensorflow-lua <version>" \
     --notes-file RELEASE_NOTES.md \
     --latest \
     tensorflow-lua-<version>-1.rockspec \
     tensorflow-lua-<version>-1.src.rock
   ```

2. Upload to LuaRocks:

   ```sh
   luarocks upload tensorflow-lua-<version>-1.rockspec \
     tensorflow-lua-<version>-1.src.rock
   ```

3. Verify:

   ```sh
   luarocks search tensorflow-lua
   luarocks install --tree=/tmp/tf-published --deps-mode=none \
     tensorflow-lua <version>
   ```

4. Remove local `.src.rock`/`.all.rock` artifacts. Never move or recreate an
   existing release tag.

## 0.6.x patch checklist

- [ ] patch only fixes, docs, CI, or low-risk correctness changes;
- [ ] changelog entry is under a new patch version;
- [ ] no public API removals;
- [ ] test matrix green on every supported TensorFlow version;
- [ ] patch rockspec revision incremented (`0.6.1-1`, not `0.6.0-2`);
- [ ] LuaRocks package install verified after upload;
- [ ] existing tags untouched.
