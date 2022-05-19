# QSimulator.jl

[![CI](https://github.com/spherical-tensor/QSimulator.jl/actions/workflows/CI.yml/badge.svg?branch=master)](https://github.com/spherical-tensorQSimulator.jl/actions/workflows/CI.yml)

QSimulator Fork from BBN-Q's original.

Package for simulating time dynamics of quantum systems with a focus on superconducting qubits.

## Documentation

To build and navigate the documentation locally, navigate into the `docs` directory and run the following 
```bash
julia --project=@. make.jl
julia --project=@.  -e 'using LiveServer; serve(dir="./build")'
```

## Installation

```
(v1.6) pkg> add https://github.com/spherical-tensor/QSimulator.jl
```

## Unit tests

```julia
Pkg.test("QSimulator")
```

## Benchmarks
We can track the code performance between commits by running the benchmarking suite in
`benchmark/benchmarks.jl` using [PkgBenchmark](https://github.com/JuliaCI/PkgBenchmark.jl).

Please the documentation for more details.
