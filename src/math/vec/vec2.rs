use std::ops::{Add, AddAssign, Div, DivAssign, Mul, Sub, SubAssign};

use crate::math::vec::ops::*;

#[derive(Default, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C, align(8))]
pub struct Vec2f {
    x: f32,
    y: f32,
}

impl Vec2f {
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

impl From<f32> for Vec2f {
    fn from(value: f32) -> Self {
        Self { x: value, y: value }
    }
}

impl From<[f32; 2]> for Vec2f {
    fn from(value: [f32; 2]) -> Self {
        Self {
            x: value[0],
            y: value[1],
        }
    }
}

impl Add for Vec2f {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(self.x + rhs.x, self.y + rhs.y)
    }
}

impl AddAssign for Vec2f {
    fn add_assign(&mut self, rhs: Self) {
        self.x += rhs.x;
        self.y += rhs.y;
    }
}

impl Sub for Vec2f {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(self.x - rhs.x, self.y - rhs.y)
    }
}

impl SubAssign for Vec2f {
    fn sub_assign(&mut self, rhs: Self) {
        self.x -= rhs.x;
        self.y -= rhs.y;
    }
}

impl Mul for Vec2f {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self::new(self.x * rhs.x, self.y * rhs.y)
    }
}

impl Mul<f32> for Vec2f {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        Self::new(self.x * rhs, self.y * rhs)
    }
}

impl Mul<Vec2f> for f32 {
    type Output = Vec2f;

    fn mul(self, rhs: Vec2f) -> Self::Output {
        Vec2f::new(self * rhs.x, self * rhs.y)
    }
}

impl Div for Vec2f {
    type Output = Self;
    fn div(self, rhs: Self) -> Self::Output {
        Self::new(self.x / rhs.x, self.y / rhs.y)
    }
}

impl Div<f32> for Vec2f {
    type Output = Self;
    fn div(self, rhs: f32) -> Self::Output {
        Self::new(self.x / rhs, self.y / rhs)
    }
}

impl DivAssign for Vec2f {
    fn div_assign(&mut self, rhs: Self) {
        self.x /= rhs.x;
        self.y /= rhs.y;
    }
}

impl Length for Vec2f {
    type Output = f32;

    fn length(self) -> Self::Output {
        f32::sqrt((self.x * self.x) + (self.y * self.y))
    }
}

impl Distance for Vec2f {
    type Output = f32;

    fn distance(a: Self, b: Self) -> Self::Output {
        (a - b).length()
    }
}

impl Normalized for Vec2f {
    fn normalized(self) -> Self {
        self / self.length()
    }
}

impl Reflect for Vec2f {
    fn reflect(incident: Self, normal: Self) -> Self {
        incident - (normal * 2.0 * Self::dot(incident, normal))
    }
}

impl Refract for Vec2f {
    fn refract(incident: Self, normal: Self, eta: f32) -> Self {
        let k =
            1.0 - (eta * eta) * (1.0 - (Self::dot(normal, incident) * Self::dot(normal, incident)));
        if k < 0.0 {
            return Self::from(0.0);
        } else {
            let eta_dot_n_i = eta * Self::dot(normal, incident);
            return (incident * eta) - (Self::from(eta_dot_n_i + f32::sqrt(k)) * normal);
        }
    }
}

impl Dot for Vec2f {
    type Output = f32;

    fn dot(a: Self, b: Self) -> Self::Output {
        (a.x * b.x) + (a.y * b.y)
    }
}

impl Cross for Vec2f {
    type Output = f32;

    fn cross(a: Self, b: Self) -> Self::Output {
        (a.x * b.y) - (a.y * b.x)
    }
}

impl Min for Vec2f {
    fn min(a: Self, b: Self) -> Self {
        Self::new(f32::min(a.x, b.x), f32::min(a.y, b.y))
    }
}
impl Max for Vec2f {
    fn max(a: Self, b: Self) -> Self {
        Self::new(f32::max(a.x, b.x), f32::max(a.y, b.y))
    }
}

impl Pow<Self> for Vec2f {
    fn pow(a: Self, b: Self) -> Self {
        Self::new(f32::powf(a.x, b.x), f32::powf(a.y, b.y))
    }
}

impl Pow<f32> for Vec2f {
    fn pow(a: Self, b: f32) -> Self {
        Self::new(f32::powf(a.x, b), f32::powf(a.y, b))
    }
}

impl Abs for Vec2f {
    fn abs(self) -> Self {
        Self::new(f32::abs(self.x), f32::abs(self.y))
    }
}

impl Reversed for Vec2f {
    fn reversed(self) -> Self {
        Self::new(-self.x, -self.y)
    }
}

impl Mix<Self> for Vec2f {
    fn mix(a: Self, b: Self, amount: Self) -> Self {
        Self::new(
            (a.x * (1.0 - amount.x)) + b.x * amount.x,
            (a.y * (1.0 - amount.y)) + b.y * amount.y,
        )
    }
}

impl Mix<f32> for Vec2f {
    fn mix(a: Self, b: Self, amount: f32) -> Self {
        Self::new(
            (a.x * (1.0 - amount)) + b.x * amount,
            (a.y * (1.0 - amount)) + b.y * amount,
        )
    }
}
