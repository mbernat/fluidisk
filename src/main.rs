use glam::{Vec2, vec2};
use macroquad::{
    color::{BLACK, WHITE},
    input::{KeyCode, is_key_down},
    shapes::draw_circle,
    time::get_frame_time,
    window::{clear_background, next_frame},
};
use std::f32::consts::PI;

#[derive(Clone)]
struct Particle {
    pos: Vec2,
    vel: Vec2,
}

struct World {
    particles: Vec<Particle>,
    gravity: Vec2,
    // desired fluid density
    rho: f32,
    // kernel cutoff distance
    h: f32,
    // constraint force mixing
    cfm: f32,

    solver_iterations: usize,
}

impl World {
    fn step(&mut self, dt: f32) {
        let mut pos = vec![];
        for p in &mut self.particles {
            let center = vec2(400.0, 300.0);
            let radius = 250.0;
            let diff = p.pos - center;
            let (normal, length) = diff.normalize_and_length();
            let spring_force = if length > radius {
                -1000.0 * (length - radius) * normal
            } else {
                Vec2::ZERO
            };

            let dir = if is_key_down(KeyCode::Left) {
                -1.0
            } else if is_key_down(KeyCode::Right) {
                1.0
            } else {
                0.0
            };

            let rotation_force = 5.0 * dir * (p.pos - center).perp();

            let press = if is_key_down(KeyCode::Down) { 1.0 } else { 0.0 };
            let press_force = 5000.0 * press * normal;

            let drag_force = -1.0 * p.vel;
            let force = spring_force + rotation_force + press_force + drag_force;

            let mass = 1.0;
            p.vel += (force / mass + self.gravity) * dt;
            pos.push(p.pos + p.vel * dt);
        }

        let neighbors: Vec<Vec<usize>> = self
            .particles
            .iter()
            .enumerate()
            .map(|(i, _)| {
                self.particles
                    .iter()
                    .enumerate()
                    .filter_map(|(j, _)| {
                        let dist_squared = (pos[i] - pos[j]).length_squared();
                        let keep = i != j && dist_squared <= self.h.powi(2);
                        keep.then_some(j)
                    })
                    .collect()
            })
            .collect();

        let max_count = neighbors.iter().map(|n| n.len()).max().unwrap();
        println!("Max neighbor count: {max_count}");

        for _iter in 0..self.solver_iterations {
            let lambda: Vec<_> = neighbors
                .iter()
                .enumerate()
                .map(|(i, neighbors)| {
                    let rho: f32 = neighbors
                        .iter()
                        .copied()
                        .map(|j| Self::kernel(pos[i] - pos[j], self.h))
                        .sum();
                    let ci = rho / self.rho - 1.0;
                    let grads: Vec<Vec2> = neighbors
                        .iter()
                        .copied()
                        .map(|j| Self::kernel_grad(pos[i] - pos[j], self.h) / self.rho)
                        .collect();

                    let square_of_sum_of_grads: f32 = grads.iter().sum::<Vec2>().length_squared();
                    let sum_of_square_of_grads: f32 =
                        grads.iter().map(|g| g.length_squared()).sum();
                    let sum_grad_ci = square_of_sum_of_grads + sum_of_square_of_grads;
                    -ci / (sum_grad_ci + self.cfm)
                })
                .collect();

            let delta_pos: Vec<_> = neighbors
                .iter()
                .enumerate()
                .map(|(i, neighbors)| {
                    let sum_of_grads: Vec2 = neighbors
                        .iter()
                        .copied()
                        .map(|j| {
                            (lambda[i] + lambda[j]) * Self::kernel_grad(pos[i] - pos[j], self.h)
                        })
                        .sum();

                    sum_of_grads / self.rho
                })
                .collect();

            for (pos, delta_pos) in pos.iter_mut().zip(delta_pos.into_iter()) {
                *pos += delta_pos;

                // TODO: proper collision detection

                pos.x = pos.x.clamp(0.0, 800.0);
                pos.y = pos.y.clamp(0.0, 600.0);
            }
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.vel = (pos[i] - p.pos) / dt;

            // TODO: vorticity confinement & XSPH viscosity

            p.pos = pos[i];
        }
    }

    // TODO: 2d kernels or normalization; poly6 and spiky are normalized for 3d, so we multiply by `h` to correct for that

    fn kernel(x: Vec2, h: f32) -> f32 {
        h * poly6(x.length_squared(), h)
    }

    fn kernel_grad(x: Vec2, h: f32) -> Vec2 {
        let (norm, r) = x.normalize_and_length();
        h * spiky_diff(r, h) * norm
    }

    fn draw(&self) {
        for p in &self.particles {
            draw_circle(p.pos.x, p.pos.y, 2.0, WHITE);
            // draw_circle_lines(p.pos.x, p.pos.y, self.h, 1.0, GREEN);
        }
    }
}

fn poly6(r_squared: f32, h: f32) -> f32 {
    if r_squared <= h * h {
        315.0 / (64.0 * PI * h.powi(9)) * (h * h - r_squared).powi(3)
    } else {
        0.0
    }
}

fn _spiky(r: f32, h: f32) -> f32 {
    if r <= h {
        15.0 / (PI * h.powi(6)) * (h - r).powi(3)
    } else {
        0.0
    }
}

fn spiky_diff(r: f32, h: f32) -> f32 {
    if r <= h {
        -45.0 / (PI * h.powi(6)) * (h - r).powi(2)
    } else {
        0.0
    }
}

#[macroquad::main("Fluidisk")]
async fn main() {
    let mut world = World {
        particles: vec![],
        gravity: vec2(0.0, 500.0),
        rho: 1e-2,
        h: 20.0,
        cfm: 2e-1,

        solver_iterations: 4,
    };

    let count = 40;
    let sep = 10.0;
    let offset = (vec2(800.0, 600.0) - count as f32 * vec2(sep, sep)) / 2.0;
    for i in 0..count {
        for j in 0..count {
            world.particles.push(Particle {
                pos: vec2(i as f32 * sep, j as f32 * sep) + offset,
                vel: Vec2::ZERO,
            })
        }
    }

    loop {
        clear_background(BLACK);

        let dt = get_frame_time();
        world.step(dt);
        world.draw();

        next_frame().await
    }
}
