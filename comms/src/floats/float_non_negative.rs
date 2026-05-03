use std::ops::Deref;

use serde::{Deserialize, Deserializer, Serialize, de};

#[derive(Serialize, Debug, Clone, Copy)]
#[serde(transparent)]
pub struct FloatNonNegative {
    value: f32,
}

impl FloatNonNegative {
    /// Creates a new `FloatNonNegative`.
    ///
    /// # Args
    /// * `value` - The inner value.
    ///
    /// # Returns
    /// An option with `Some` value if the given `value` is not negative, else `None`.
    pub fn new(value: f32) -> Option<Self> {
        value
            .is_sign_positive()
            .then_some(FloatNonNegative { value })
    }
}

impl<'de> Deserialize<'de> for FloatNonNegative {
    /// Deserializes a `FloatNonNegative` from a float value.
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
        FloatNonNegative::new(value)
            .ok_or_else(|| de::Error::custom("FloatNonNegative value must be non negative"))
    }
}

impl Deref for FloatNonNegative {
    type Target = f32;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
