# Keccak Prime

The implementation of Zenotta's Keccak Prime function.

To make use of CPU acceleration for AES, provide the following compilation flags:

```
RUSTFLAGS="-Ctarget-cpu=sandybridge -Ctarget-feature=+aes,+sse2,+sse4.1,+ssse3"
```

## GPU compute

To access GPU, we use the Vulkan Compute API.

### Linux

Most likely, you don't need to do anything to make it work on Linux.
If you have an AMD or Intel GPU, just install Mesa.
If you use an NVIDIA GPU, install the proprietary driver.

### Windows

### macOS

`brew install molten-vk`
