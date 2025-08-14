## About

Fluidisk is a simple fluid simulation based on the paper [position based fluids](https://blog.mmacklin.com/project/pbf/).

https://github.com/user-attachments/assets/44b16558-040f-42fa-bae7-08482cf03ef0

## Running the project

As a Rust project it requires Rust to be installed, you can [get it here](https://www.rust-lang.org/learn/get-started). Then run it with `cargo`.

```
cargo run --release
```

## TODOs

- [ ] Derive related simulation parameters like number of particles, target density and kernel cutoff distance
      from each other or perhaps from a simpler set of more intuitive parameters.
- [ ] Show some info about the constraint solving convergence.
- [ ] Add collision detection and resolution to enforce hard boundaries.
- [ ] Render the particles so that they look like an actual fluid.
- [ ] Add vorticity confiment and XSPH viscosity.
- [ ] Add SPH as an optional backend.
- [ ] Support calculations on a GPU.
- [x] Normalize the SPH kernels for 2d or find better ones specifically for 2d.

## Theory

Position based fluids puts the venerable [smoothed-particle hydrodynamics](https://en.wikipedia.org/wiki/Smoothed-particle_hydrodynamics) (SPH) in the context of [position based dynamics](https://matthias-research.github.io/pages/publications/posBasedDyn.pdf), in the sense that the SPH density estimates are enforced through constraint solving in the position space.

This particular method was chosen because it is both relatively simple to implement and provides visually appealing results of the fluid interface. But it has to be noted that it is less physically grounded than most other methods, including SPH, hybrid particle/grid methods ([PIC](https://en.wikipedia.org/wiki/Particle-in-cell), [FLIP](https://ui.adsabs.harvard.edu/abs/1986JCoPh..65..314B/abstract)), and especially the traditional grid based methods ([PISO](https://en.wikipedia.org/wiki/PISO_algorithm), [SIMPLE](https://en.wikipedia.org/wiki/SIMPLE_algorithm)).
