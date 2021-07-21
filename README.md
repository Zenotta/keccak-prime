# Keccak Prime

The implementation of Zenotta's Keccak Prime function.

To make use of CPU acceleration for AES, provide the following compilation flags:

```
RUSTFLAGS="-Ctarget-cpu=sandybridge -Ctarget-feature=+aes,+sse2,+sse4.1,+ssse3"
```
