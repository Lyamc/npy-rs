# npy-rs
[![crates.io version](https://img.shields.io/crates/v/npy.svg)](https://crates.io/crates/npy) [![Documentation](https://docs.rs/npy/badge.svg)](https://docs.rs/npy/)

Numpy format (*.npy) serialization and deserialization.

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Usage

To use **npy-rs**, two dependencies must be specified in `Cargo.toml`:

```toml
npy = "0.5"
npy-derive = "0.5"
```

The `npy-derive` dependency is only needed for
[structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html)
serialization.

Data can now be imported from a `*.npy` file:

```rust
use npy::NpyData;
use std::io::Read;

let mut buf = vec![];
std::fs::File::open("data.npy").unwrap().read_to_end(&mut buf).unwrap();
let data: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();

```

and exported to a `*.npy` file:

```rust
npy::to_file("data.npy", data).unwrap();
```
