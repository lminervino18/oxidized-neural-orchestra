use std::path::Path;

use log::info;
use safetensors::tensor;

use crate::{
    Result,
    configs::{LayerConfig, ModelConfig},
};

/// The result of a completed training session.
///
/// Contains the trained model parameters alongside the model architecture,
/// allowing the weights to be saved to disk without requiring additional context.
#[derive(Debug)]
pub struct TrainedModel {
    pub params: Vec<f32>,
    pub model: ModelConfig,
    pub input_size: usize,
}

impl TrainedModel {
    /// Saves the trained model parameters to a `.safetensors` file.
    ///
    /// Each dense layer produces two tensors named `layer_N.weight` and
    /// `layer_N.bias`, following the PyTorch `state_dict` convention.
    /// The weight tensor has shape `[input_size, output_size]` and the
    /// bias tensor has shape `[output_size]`.
    ///
    /// # Args
    /// * `path` - The output file path (e.g. `"model.safetensors"`).
    ///
    /// # Errors
    /// Returns an `OrchErr` if the file cannot be written or the parameter
    /// buffer does not match the model architecture.
    pub fn save_safetensors(&self, path: impl AsRef<Path>) -> Result<()> {
        use safetensors::{Dtype, tensor::TensorView};

        let params_bytes: &[u8] = bytemuck::cast_slice(&self.params);
        let mut tensors: Vec<(String, TensorView)> = Vec::new();
        let mut offset = 0;
        let mut prev = self.input_size;

        for (i, layer) in self.model.layers.iter().enumerate() {
            let (w_count, b_count, w_shape, b_shape, out) = match layer {
                LayerConfig::Dense { output_size, .. } => {
                    let out = output_size.get();
                    let w_count = prev * out;
                    let b_count = out;

                    (w_count, b_count, vec![prev, out], vec![out], out)
                }
                LayerConfig::Conv {
                    input_dim,
                    kernel_dim,
                    stride,
                    padding,
                    ..
                } => {
                    let input_height = input_dim.1.get();
                    let input_width = input_dim.2.get();
                    let stride = stride.get();
                    let (filters, in_channels, kernel_size) =
                        (kernel_dim.0.get(), kernel_dim.1.get(), kernel_dim.2.get());

                    let output_height = (input_height + 2 * padding - kernel_size) / stride + 1;
                    let output_width = (input_width + 2 * padding - kernel_size) / stride + 1;

                    let w_count = filters * in_channels * kernel_size * kernel_size;
                    let b_count = filters;
                    let out = output_height * output_width * filters;

                    (
                        w_count,
                        b_count,
                        vec![filters, in_channels, kernel_size, kernel_size],
                        vec![filters],
                        out,
                    )
                }
            };

            let w_bytes = &params_bytes[offset * 4..(offset + w_count) * 4];
            let tensor_view = TensorView::new(Dtype::F32, w_shape, w_bytes)?;
            tensors.push((format!("layer_{i}.weight"), tensor_view));
            offset += w_count;

            let b_bytes = &params_bytes[offset * 4..(offset + b_count) * 4];
            let tensor_view = TensorView::new(Dtype::F32, b_shape, b_bytes)?;
            tensors.push((format!("layer_{i}.bias"), tensor_view));
            offset += b_count;

            prev = out;
        }

        tensor::serialize_to_file(tensors, &None, path.as_ref())?;
        info!("model saved to {}", path.as_ref().display());
        Ok(())
    }
}
