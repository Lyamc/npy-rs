# npy-rs
[![crates.io version](https://img.shields.io/crates/v/npy.svg)](https://crates.io/crates/npy) [![Documentation](https://docs.rs/npy/badge.svg)](https://docs.rs/npy/)

Numpy format (*.npy) serialization and deserialization.

[**NPY**](https://docs.scipy.org/doc/numpy-dev/neps/npy-format.html) is a simple binary data format.
It stores the type, shape and endianness information in a header,
which is followed by a flat binary data field. This crate offers a simple, mostly type-safe way to
read and write *.npy files. Files are handled using iterators, so they don't need to fit in memory.

## Key Features

- **Nom 8 Support**: Updated to the latest `nom` parser combinator library.
- **NPY 2.0 Support**: Added support for parsing NPY format version 2.0 (headers with 4-byte length fields).
- **Type Safety**: Leverages Rust's type system for safe deserialization.
- **Memory Efficiency**: Processes files using iterators, suitable for large datasets.

## Usage

To use **npy-rs**, specify the dependency in `Cargo.toml`:

```toml
[dependencies]
npy = "0.5"
npy-derive = "0.5"
```

The `npy-derive` dependency is only needed for
[structured array](https://docs.scipy.org/doc/numpy/user/basics.rec.html)
serialization.

### Importing Data

Data can be imported from a `*.npy` file:

```rust
use npy::NpyData;
use std::io::Read;

let mut buf = vec![];
std::fs::File::open("data.npy").unwrap().read_to_end(&mut buf).unwrap();
let data: Vec<f64> = NpyData::from_bytes(&buf).unwrap().to_vec();
```

### Exporting Data

Data can be exported to a `*.npy` file:

```rust
npy::to_file("data.npy", data).unwrap();
```

## License

This project is licensed under the MIT license.

