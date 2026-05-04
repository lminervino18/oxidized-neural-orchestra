use std::{cell::RefCell, rc::Rc};

use comms::specs::machine_learning::{DistributionSpec, ParamGenSpec};
use rand::{Rng, SeedableRng, rngs::StdRng};

use super::{ChainedParamGen, ConstParamGen, ParamGen, RandParamGen, Result};

/// Makes `callback`'s return type generic, when trying to resolve a concrete `RandParamGen` it avoids boxing all
/// parameter generators variants.
///
/// When trying to resolve for a `ChainedParamGen`, a sub `RandParamGen` must be returned as a boxed
/// `ParamGen`, so it must duplicate the distribution's static dispatch match.
///
/// To avoid this duplication, this macro generalizes what it's immediately done with the concrete parameter generator,
/// and thus avoids boxing parameter generators unnecessarily.
///
/// # Args
/// * `rng` - A random number generator.
/// * `dist_spec` - A specification for a distribution.
/// * `limit` - The limit of parameters that the `RandParamGen` can generate.
/// * `callback` - The closure to call passing in the created weigth gen.
macro_rules! with_distribution {
    ($rng:expr, $dist_spec:expr, $limit:expr, $callback:expr) => {
        match $dist_spec {
            DistributionSpec::Uniform { low, high } => {
                let param_gen = RandParamGen::uniform($rng, $limit, low, high)?;
                ($callback)(param_gen)
            }
            DistributionSpec::UniformInclusive { low, high } => {
                let param_gen = RandParamGen::uniform_inclusive($rng, $limit, low, high)?;
                ($callback)(param_gen)
            }
            DistributionSpec::XavierUniform { fan_in, fan_out } => {
                let param_gen = RandParamGen::xavier_uniform($rng, $limit, fan_in, fan_out)?;
                ($callback)(param_gen)
            }
            DistributionSpec::LecunUniform { fan_in } => {
                let param_gen = RandParamGen::lecun_uniform($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Normal { mean, std_dev } => {
                let param_gen = RandParamGen::normal($rng, $limit, mean, std_dev)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Kaiming { fan_in } => {
                let param_gen = RandParamGen::kaiming($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Xavier { fan_in, fan_out } => {
                let param_gen = RandParamGen::xavier($rng, $limit, fan_in, fan_out)?;
                ($callback)(param_gen)
            }
            DistributionSpec::Lecun { fan_in } => {
                let param_gen = RandParamGen::lecun($rng, $limit, fan_in)?;
                ($callback)(param_gen)
            }
        }
    };
}

/// Builds parameter generators given a specification.
#[derive(Default)]
pub struct ParamGenBuilder;

impl ParamGenBuilder {
    /// Creates a new `ParamGenBuilder`.
    ///
    /// # Returns
    /// A new `ParamGenBuilder` instance.
    pub fn new() -> Self {
        Self
    }

    /// Resolves the `ParamGen` for this server.
    ///
    /// # Args
    /// * `spec` - The specification of the parameter server.
    ///
    /// # Returns
    /// A new Server or a `RandErr` if the specification has
    /// invalid `RandParamGen` construction values.
    pub fn build<S>(&self, spec: ParamGenSpec, seed: S) -> Result<Box<dyn ParamGen>>
    where
        S: Into<Option<u64>>,
    {
        match spec {
            ParamGenSpec::Const { value, limit } => {
                let param_gen = ConstParamGen::new(value, limit);
                Ok(Box::new(param_gen))
            }
            ParamGenSpec::Rand {
                distribution,
                limit,
            } => {
                let rng = self.generate_rng(seed.into());
                with_distribution!(rng, distribution, limit, |param_gen| {
                    Ok(Box::new(param_gen) as Box<dyn ParamGen>)
                })
            }
            ParamGenSpec::Chained { ref specs } => {
                let rng = self.generate_rng(seed.into());
                let param_gen = self.resolve_chained(rng, specs)?;
                Ok(Box::new(param_gen))
            }
        }
    }

    /// Resolves the `ChainedParamGen` parameter generator.
    ///
    /// # Args
    /// * `rng` - A random number generator.
    /// * `specs` - A list of parameter generator specifications.
    ///
    /// # Returns
    /// A resolved `ChainedParamGen`.
    fn resolve_chained<R>(
        &self,
        rng: Rc<RefCell<R>>,
        specs: &[ParamGenSpec],
    ) -> Result<ChainedParamGen>
    where
        R: Rng + 'static,
    {
        let mut param_gens: Vec<Box<dyn ParamGen>> = Vec::with_capacity(specs.len());

        for spec in specs {
            match spec {
                ParamGenSpec::Const { value, limit } => {
                    let param_gen = ConstParamGen::new(*value, *limit);
                    param_gens.push(Box::new(param_gen));
                }
                ParamGenSpec::Rand {
                    distribution,
                    limit,
                } => {
                    with_distribution!(rng.clone(), *distribution, *limit, |param_gen| {
                        param_gens.push(Box::new(param_gen))
                    });
                }
                ParamGenSpec::Chained { specs } => {
                    let param_gen = self.resolve_chained(rng.clone(), specs)?;
                    param_gens.push(Box::new(param_gen))
                }
            }
        }

        Ok(ChainedParamGen::new(param_gens))
    }

    /// Generates a random number generator given (or not) a seed.
    ///
    /// # Args
    /// * `seed` - An optional seed for the rng.
    ///
    /// # Returns
    /// An clonable random number generator with interior mutability.
    fn generate_rng(&self, seed: Option<u64>) -> Rc<RefCell<StdRng>> {
        let rng = match seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_os_rng(),
        };

        Rc::new(RefCell::new(rng))
    }
}
