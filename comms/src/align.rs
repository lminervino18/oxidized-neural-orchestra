// Trait to generalize over primitive number types that are 1 byte aligned.
pub trait Align1: bytemuck::Pod {}

impl Align1 for u8 {}
impl Align1 for i8 {}
impl Align1 for u16 {}
impl Align1 for i16 {}
impl Align1 for u32 {}
impl Align1 for i32 {}
impl Align1 for u64 {}
impl Align1 for i64 {}
impl Align1 for u128 {}
impl Align1 for i128 {}
impl Align1 for f32 {}
impl Align1 for f64 {}
