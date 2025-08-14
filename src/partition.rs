use glam::Vec2;

/// 2d grid of particle indices and positions contained within the given square cell.
/// Particles that are out of bounds get put into the closest cell.
pub struct Partition {
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

    pub fn find_neighbors(&self, index: usize, pos: Vec2) -> Vec<usize> {
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
