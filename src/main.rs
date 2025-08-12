use egui::Slider;
use glam::{Vec2, vec2};
use macroquad::{
    color::{BLACK, WHITE},
    input::{KeyCode, is_key_down},
    shapes::draw_circle,
    time::get_frame_time,
    window::{clear_background, next_frame},
};
use std::f32::consts::PI;

const WIDTH: f32 = 800.0;
const HEIGHT: f32 = 600.0;
const DEBUG_NEIGHBORS: bool = false;

struct Partition {
    // 2d grid of particle indices and positions contained within the given square cell with side `size`
    grid: Vec<Vec<Vec<(usize, Vec2)>>>,

    size: f32,
    x_dim: usize,
    y_dim: usize,
}

impl Partition {
    pub fn new(width: f32, height: f32, size: f32) -> Self {
        let x_dim = (width / size).ceil() as usize;
        let y_dim = (height / size).ceil() as usize;
        let grid = vec![vec![vec![]; y_dim]; x_dim];

        Self {
            grid,
            size,
            x_dim,
            y_dim,
        }
    }

    pub fn add(&mut self, p: (usize, Vec2)) {
        let cell = self.find_cell(p.1);
        self.grid[cell.0][cell.1].push(p);
    }

    fn find_cell(&self, pos: Vec2) -> (usize, usize) {
        let cell_x = (pos.x / self.size)
            .round()
            .clamp(0.0, self.x_dim as f32 - 1.0) as usize;
        let cell_y = (pos.y / self.size)
            .round()
            .clamp(0.0, self.y_dim as f32 - 1.0) as usize;

        (cell_x, cell_y)
    }

    fn find_neighbors(&self, index: usize, pos: Vec2) -> Vec<usize> {
        let cell = self.find_cell(pos);
        let mut neighbors = vec![];

        for dx in -1..=1 {
            for dy in -1..=1 {
                let cell_x = cell.0 as isize + dx;
                let cell_y = cell.1 as isize + dy;

                if cell_x < 0
                    || cell_x >= self.x_dim as isize
                    || cell_y < 0
                    || cell_y >= self.y_dim as isize
                {
                    continue;
                }

                let cell = &self.grid[cell_x as usize][cell_y as usize];
                neighbors.extend(cell.iter().filter_map(|(p_index, p_pos)| {
                    let dist_squared = (pos - p_pos).length_squared();
                    let keep = index != *p_index && dist_squared <= self.size.powi(2);
                    keep.then_some(p_index)
                }));
            }
        }

        neighbors
    }
}

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
        let particle_count_per_dim = 40;
        World {
            particles: particles(particle_count_per_dim),
            gravity: vec2(0.0, 500.0),
            rho: 1e-2,
            h: 20.0,
            cfm: 2e-1,
            solver_iterations: 4,
            particle_count_per_dim,
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

    // TODO: 2d kernels or normalization; poly6 and spiky are normalized for 3d, so we multiply by `h` to somewhat correct for that

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

fn input(world: &mut World) {
    egui_macroquad::ui(|egui_ctx| {
        egui::Window::new("controls").show(egui_ctx, |ui| {
            ui.label("Density");
            ui.add(Slider::new(&mut world.rho, 1e-3..=1e-1));

            ui.label("Kernel cutoff distance");
            ui.add(Slider::new(&mut world.h, 5.0..=100.0));

            ui.label("Constraint force mixing");
            ui.add(Slider::new(&mut world.cfm, 0.0..=1.0));

            ui.label("Solver iterations");
            ui.add(Slider::new(&mut world.solver_iterations, 1..=20));

            ui.label("Particle count per dim");
            ui.add(Slider::new(&mut world.particle_count_per_dim, 1..=100));

            ui.separator();

            if ui.button("Restart").clicked() {
                world.particles = particles(world.particle_count_per_dim);
            }
        });
    });
}

#[macroquad::main("Fluidisk")]
async fn main() {
    let mut world = World::new();

    loop {
        clear_background(BLACK);

        input(&mut world);

        let dt = get_frame_time();
        world.step(dt);
        world.draw();

        egui_macroquad::draw();

        next_frame().await
    }
}
