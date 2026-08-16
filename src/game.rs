use crate::{log_info, log_warning};

pub struct World {
    pub particles: Vec<Particle>,
    pub extent: glam::Vec2,
}

impl World {
    pub fn new(extent: glam::Vec2) -> Self {
        Self {
            particles: Vec::new(),
            extent,
        }
    }

    pub fn update(&mut self, dt: f32) {
        // Update all particles
        for i in 0..self.particles.len() {
            for j in 0..self.particles.len() {
                if i == j {
                    continue;
                }

                let other = self.particles[j];
                let particle = &mut self.particles[i];

                let r_sq = particle.pos.distance_squared(other.pos);
                let f_g = 6.67430e-11 * particle.mass * other.mass / r_sq;
                particle.add_force((other.pos - particle.pos) * f_g);
            }
        }

        self.particles.iter_mut().for_each(|particle| {
            particle.update(dt);
        });
    }

    pub fn add_particle(&mut self, particle: Particle) {
        self.particles.push(particle);
    }
}

#[derive(Clone, Copy)]
pub struct Particle {
    pub color: glam::Vec4,
    pub pos: glam::Vec2,
    pub old_pos: glam::Vec2,
    pub acc: glam::Vec2,
    pub radius: f32,
    pub mass: f32,
}

impl Particle {
    pub fn new(pos: glam::Vec2, radius: f32) -> Self {
        Self {
            color: glam::Vec4::new(pos.x / 512.0 * 0.5, pos.y / 512.0 * 0.5, 0.0, 0.0),
            pos,
            old_pos: pos,
            acc: glam::Vec2::ZERO,
            radius,
            mass: 1e12f32,
        }
    }

    pub fn update(&mut self, dt: f32) {
        let particle_vel = self.pos - self.old_pos;
        self.old_pos = self.pos;
        self.pos += particle_vel + self.acc * (dt * dt);
        self.acc = glam::Vec2::ZERO;
    }

    pub fn add_force(&mut self, force: glam::Vec2) {
        self.acc += force / self.mass;
    }
}
