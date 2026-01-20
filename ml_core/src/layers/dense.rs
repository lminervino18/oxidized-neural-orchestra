/*
 * las matrices Z y A evidentemente no van a poder estar acá porque como tenemos
 * referencias mutables para los pesos y sesgos, en algún momento vamos a tener q matar al
 * modelo para poder escribir en los vectores raw de los mismos, y por tanto vamos a tener
 * q estar levantando el modelo muchas veces. Arrancar estos vectores es costoso porq hay
 * que allocar memoria. Las dos opciones que se me ocurren son:
 * - que quienes necesiten las matrices Z y A del modelo las computen, que es un poco
 * molesto porque significa que quien lo haga va a tener que conocere aún más la
 * implementación
 * - puta me olvidé de la otra que gagá q estoy. Volveré...
 * - (opción 3!!!) bueno otra es mantener la memoria de estas matrices en otro lado como se
 * está haciendo con los W y b, pero no sé hasta qué punto es copado tener que mantener eso
 * constantemente.
 * Otra cosa que hay que modificar en caso de que optemos por crear modelos constantemente
 * es evitar el checkeo de la dimensionalidad de la raw data (creo que `from_shape_ptr`
 * hace algo del estilo)
 ***/

use ndarray::{Array1, ArrayView1, ArrayViewMut1, ArrayViewMut2, ShapeError};

pub struct Dense<'a> {
    weights: ArrayViewMut2<'a, f32>,
    biases: ArrayViewMut1<'a, f32>,
    w_sums: Array1<f32>,
    activations: Array1<f32>,
}

impl<'a> Dense<'a> {
    pub fn new(
        dim_in: usize,
        dim_out: usize,
        weights_raw: &'a mut [f32],
        biases_raw: &'a mut [f32],
    ) -> Result<Self, ShapeError> {
        let weights = ArrayViewMut2::from_shape((dim_in, dim_out), weights_raw)?;
        let biases = ArrayViewMut1::from_shape(dim_out, biases_raw)?;
        let w_sums = Array1::<f32>::zeros(dim_out);
        let activations = Array1::<f32>::zeros(dim_out);

        Ok(Self {
            weights,
            biases,
            w_sums,
            activations,
        })
    }

    pub fn forward(&self, x: ArrayView1<f32>) -> Array1<f32> {
        let w = &self.weights;
        let b = &self.biases;

        w.dot(&x) + b
    }
}
