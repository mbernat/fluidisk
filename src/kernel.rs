pub mod d2 {
    use std::f32::consts::PI;

    pub fn poly6(r_squared: f32, h: f32) -> f32 {
        if r_squared <= h * h {
            4.0 / (PI * h.powi(8)) * (h * h - r_squared).powi(3)
        } else {
            0.0
        }
    }

    #[expect(unused)]
    fn spiky(r: f32, h: f32) -> f32 {
        if r <= h {
            10.0 / (PI * h.powi(5)) * (h - r).powi(3)
        } else {
            0.0
        }
    }

    pub fn spiky_diff(r: f32, h: f32) -> f32 {
        if r <= h {
            -30.0 / (PI * h.powi(5)) * (h - r).powi(2)
        } else {
            0.0
        }
    }
}

#[expect(unused)]
mod d3 {
    use std::f32::consts::PI;

    pub fn poly6(r_squared: f32, h: f32) -> f32 {
        if r_squared <= h * h {
            315.0 / (64.0 * PI * h.powi(9)) * (h * h - r_squared).powi(3)
        } else {
            0.0
        }
    }

    pub fn spiky(r: f32, h: f32) -> f32 {
        if r <= h {
            15.0 / (PI * h.powi(6)) * (h - r).powi(3)
        } else {
            0.0
        }
    }

    pub fn spiky_diff(r: f32, h: f32) -> f32 {
        if r <= h {
            -45.0 / (PI * h.powi(6)) * (h - r).powi(2)
        } else {
            0.0
        }
    }
}
