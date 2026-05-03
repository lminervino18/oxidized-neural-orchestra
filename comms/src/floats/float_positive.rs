use std::ops::Deref;

use serde::{Deserialize, Deserializer, Serialize, de};

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(transparent)]
pub struct FloatPositive {
    value: f32,
}

impl FloatPositive {
    /// Creates a new `FloatPositive`.
    ///
    /// # Args
    /// * `value` - The inner value.
    ///
    /// # Returns
    /// An option with `Some` value if the given `value` is a positive number, else `None`.
    pub fn new(value: f32) -> Option<Self> {
        (value.is_sign_positive() && value != 0.0).then_some(FloatPositive { value })
    }
}

impl<'de> Deserialize<'de> for FloatPositive {
    /// Deserializes a `FloatPositive` from a float value.
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
        FloatPositive::new(value)
            .ok_or_else(|| de::Error::custom("Float01 value must be between 0.0 and 1.0"))
    }
}

impl Deref for FloatPositive {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
