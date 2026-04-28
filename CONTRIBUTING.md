This fork is maintained as a system-focused C++ codebase. Guidelines:

- Build/test locally before proposing changes:
  - `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release`
  - `cmake --build build -j$(nproc)`
  - `ctest` or `./build/tests/unittests` if enabled
- For Debian/Ubuntu environments, the non-interactive installer is in `scripts/install.sh`.
- Use non-marketing language in docs; keep changes minimal and technical.
