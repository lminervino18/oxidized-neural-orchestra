use std::ops::{Deref, DerefMut};

use serde::{Deserialize, Deserializer, Serialize, de};

/// A float with a value between `0.0` and `1.0`.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(transparent)]
pub struct Float01 {
    value: f32,
}

impl Deref for Float01 {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}

impl DerefMut for Float01 {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}

impl Float01 {
    /// Creates a new `Float01`.
    ///
    /// # Args
    /// * `value` - The value to store.
    ///
    /// # Returns
    /// A new `Float01` instance if the given value is between `0.0` and `1.0`.
    pub fn new(value: f32) -> Option<Self> {
        (0.0..=1.0).contains(&value).then_some(Self { value })
    }
}

impl<'de> Deserialize<'de> for Float01 {
    /// Deserializes a `Float01` from a float value.
    ///
    /// # Args
    /// * `deserializer` - The deserializer to read from.
    ///
    /// # Returns
    /// A validated `Float01` instance.
    ///
    /// # Errors
    /// Returns a deserialization error if the value is outside `[0.0, 1.0]`.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = f32::deserialize(deserializer)?;
        Float01::new(value)
            .ok_or_else(|| de::Error::custom("Float01 value must be between 0.0 and 1.0"))
    }
}
