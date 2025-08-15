use egui::Slider;
use glam::{Vec2, vec2};
use macroquad::{
    color::{BLACK, BLUE, WHITE},
    input::{KeyCode, is_key_down},
    shapes::{draw_circle, draw_circle_lines},
    time::get_frame_time,
    window::{clear_background, next_frame},
};

use crate::{
    kernel::d2::{poly6, spiky_diff},
    partition::Partition,
};

mod kernel;
mod partition;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;
const DEBUG_NEIGHBORS: bool = false;

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
    // TODO: compute from density?
    // kernel cutoff distance
    h: f32,
    // constraint force mixing
    cfm: f32,
    solver_iterations: usize,
    particle_count_per_dim: usize,

    draw_kernel_cutoff: bool,
}

fn particles(count_per_dim: usize) -> Vec<Particle> {
    let mut particles = vec![];

    let sep = 10.0;
    let offset = (vec2(WIDTH, HEIGHT) - count_per_dim as f32 * vec2(sep, sep)) / 2.0;
    for i in 0..count_per_dim {
        for j in 0..count_per_dim {
            particles.push(Particle {
                pos: vec2(i as f32 * sep, j as f32 * sep) + offset,
                vel: Vec2::ZERO,
            })
        }
    }

    particles
}

impl World {
    fn new() -> Self {
        let particle_count_per_dim = 50;
        World {
            particles: particles(particle_count_per_dim),
            gravity: vec2(0.0, 500.0),
            rho: 2e-2,
            h: 20.0,
            cfm: 5e-2,
            solver_iterations: 4,
            particle_count_per_dim,
            draw_kernel_cutoff: false,
        }
    }

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

        let mut partition = Partition::new(WIDTH, HEIGHT, self.h);
        for p in pos.iter().copied().enumerate() {
            partition.add(p);
        }

        let neighbors: Vec<Vec<usize>> = self
            .particles
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let neighbors = partition.find_neighbors(i, pos[i]);

                if DEBUG_NEIGHBORS {
                    let slow_neighbors: Vec<_> = self
                        .particles
                        .iter()
                        .enumerate()
                        .filter_map(|(j, _)| {
                            let dist_squared = (pos[i] - pos[j]).length_squared();
                            let keep = i != j && dist_squared <= self.h.powi(2);
                            keep.then_some(j)
                        })
                        .collect();

                    assert_eq!(neighbors.len(), slow_neighbors.len());
                }

                neighbors
            })
            .collect();

        let _max_neighbor_count = neighbors.iter().map(|n| n.len()).max().unwrap();

        for iter in 0..self.solver_iterations {
            let ci_and_lambda: Vec<_> = neighbors
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
                    (ci, -ci / (sum_grad_ci + self.cfm))
                })
                .collect();

            // TODO: move elsewhere and add more statistics
            // TODO: egui histogram
            {
                println!("Iteration {iter:2}");

                let size = 0.1;
                let mut histogram = Partition::new(2.0, size, size);
                for (i, (ci, _)) in ci_and_lambda.iter().enumerate() {
                    histogram.add((i, (vec2(ci + 1.0, 0.0))));
                }

                for (i, column) in histogram.grid.iter().enumerate() {
                    let count = column[0].len();
                    let ci_low = (i as f32) * size - 1.0;
                    let ci_high = ci_low + size;
                    println!("ci {ci_low:.2} .. {ci_high:.2}: {count:4}");
                }

                let max_abs_ci = ci_and_lambda
                    .iter()
                    .max_by(|a, b| a.0.abs().partial_cmp(&b.0.abs()).unwrap())
                    .unwrap();
                println!("Max constraint error: {:.3}", max_abs_ci.0);
            }

            let delta_pos: Vec<_> = neighbors
                .iter()
                .enumerate()
                .map(|(i, neighbors)| {
                    let sum_of_grads: Vec2 = neighbors
                        .iter()
                        .copied()
                        .map(|j| {
                            (ci_and_lambda[i].1 + ci_and_lambda[j].1)
                                * Self::kernel_grad(pos[i] - pos[j], self.h)
                        })
                        .sum();

                    sum_of_grads / self.rho
                })
                .collect();

            for (pos, delta_pos) in pos.iter_mut().zip(delta_pos.into_iter()) {
                *pos += delta_pos;

                // TODO: proper collision detection

                pos.x = pos.x.clamp(0.0, WIDTH);
                pos.y = pos.y.clamp(0.0, HEIGHT);
            }
        }

        for (i, p) in self.particles.iter_mut().enumerate() {
            p.vel = (pos[i] - p.pos) / dt;

            // TODO: vorticity confinement & XSPH viscosity

            p.pos = pos[i];
        }
    }

    fn kernel(x: Vec2, h: f32) -> f32 {
        poly6(x.length_squared(), h)
    }

    fn kernel_grad(x: Vec2, h: f32) -> Vec2 {
        let (norm, r) = x.normalize_and_length();
        spiky_diff(r, h) * norm
    }

    fn input(&mut self) {
        egui_macroquad::ui(|egui_ctx| {
            egui::Window::new("controls").show(egui_ctx, |ui| {
                ui.label("Density");
                ui.add(Slider::new(&mut self.rho, 1e-3..=1e-1));

                ui.label("Kernel cutoff distance");
                ui.add(Slider::new(&mut self.h, 5.0..=100.0));

                ui.label("Constraint force mixing");
                ui.add(Slider::new(&mut self.cfm, 0.0..=0.1));

                ui.label("Solver iterations");
                ui.add(Slider::new(&mut self.solver_iterations, 1..=20));

                ui.label("Particle count per dim");
                ui.add(Slider::new(&mut self.particle_count_per_dim, 1..=100));

                ui.separator();

                ui.checkbox(&mut self.draw_kernel_cutoff, "Draw kernel cutoff?");

                if ui.button("Restart").clicked() {
                    self.particles = particles(self.particle_count_per_dim);
                }
            });
        });
    }
    fn draw(&self) {
        for p in &self.particles {
            draw_circle(p.pos.x, p.pos.y, 2.0, BLUE.with_alpha(0.5));

            if self.draw_kernel_cutoff {
                draw_circle_lines(p.pos.x, p.pos.y, self.h, 1.0, WHITE.with_alpha(0.1));
            }
        }
    }
}

#[macroquad::main("Fluidisk")]
async fn main() {
    let mut world = World::new();

    loop {
        world.input();
        let dt = get_frame_time();
        world.step(dt);

        clear_background(BLACK);
        world.draw();
        egui_macroquad::draw();

        next_frame().await
    }
}
