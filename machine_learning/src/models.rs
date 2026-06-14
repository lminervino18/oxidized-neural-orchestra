use crate::arch::{Sequential, layers::Layer};

const MNIST_IN_CHANNELS: usize = 1;
const MNIST_INPUT_HEIGHT: usize = 28;
const MNIST_INPUT_WIDTH: usize = 28;
const MNIST_LABEL_SIZE: usize = 10;

pub fn make_nielsen_mnist_model() -> Sequential {
    let conv_filters = 10;
    let conv_kernel_size = 5;
    let conv_stride = 1;
    let conv_padding = 0;

    let _conv_output_height =
        (MNIST_INPUT_HEIGHT + 2 * conv_padding - conv_kernel_size) / conv_stride + 1;
    let _conv_output_width =
        (MNIST_INPUT_WIDTH + 2 * conv_padding - conv_kernel_size) / conv_stride + 1;

    let max_pooling_filter_size = 2;
    let max_pooling_stride = 2;
    let max_pooling_padding = 0;

    let max_pooling_output_height = (_conv_output_height + 2 * max_pooling_padding
        - max_pooling_filter_size)
        / max_pooling_stride
        + 1; // max_pooling_stride + 1
    let max_pooling_output_width = (_conv_output_width + 2 * max_pooling_padding
        - max_pooling_filter_size)
        / max_pooling_stride
        + 1; // max_pooling_stride + 1

    // esta no está en pytorch
    let unflatten = Layer::two_d_to4d(MNIST_IN_CHANNELS, MNIST_INPUT_HEIGHT, MNIST_INPUT_WIDTH);
    let conv = Layer::conv2d(
        conv_filters,
        MNIST_IN_CHANNELS,
        conv_kernel_size,
        conv_stride,
        conv_padding,
    );
    let max_pooling = Layer::max_pooling(
        max_pooling_filter_size,
        max_pooling_stride,
        max_pooling_padding,
    );
    let flatten = Layer::four_d_to2d(
        conv_filters,
        max_pooling_output_height,
        max_pooling_output_width,
    );
    let dense1 = Layer::dense((
        conv_filters * max_pooling_output_height * max_pooling_output_width,
        100,
    ));
    let sigmoid = Layer::sigmoid(1.0);
    let dense2 = Layer::dense((100, MNIST_LABEL_SIZE));
    // esta es la única que no me queda claro qué onda con la spec q tengo armada en pytorch
    let softmax = Layer::softmax();

    let layers = vec![
        unflatten,
        conv,
        max_pooling,
        flatten,
        dense1,
        sigmoid,
        dense2,
        softmax,
    ];

    Sequential::new(layers)
}

pub fn some_other_mnist_model() -> Sequential {
    let conv_filters = 10; // nielsen quiere 20 pero no se lo damos
    let conv_kernel_size = 5;
    let conv_stride = 1;
    let conv_padding = 0;

    let conv_output_height =
        (MNIST_INPUT_HEIGHT + 2 * conv_padding - conv_kernel_size) / conv_stride + 1;
    let conv_output_width =
        (MNIST_INPUT_WIDTH + 2 * conv_padding - conv_kernel_size) / conv_stride + 1;

    let max_pooling_filter_size = 2;
    let max_pooling_stride = 2;
    let max_pooling_padding = 0;

    let max_pooling_output_height = (conv_output_height + 2 * max_pooling_padding
        - max_pooling_filter_size)
        / max_pooling_stride
        + 1;
    let max_pooling_output_width = (conv_output_width + 2 * max_pooling_padding
        - max_pooling_filter_size)
        / max_pooling_stride
        + 1;

    let layers = vec![
        Layer::two_d_to4d(MNIST_IN_CHANNELS, MNIST_INPUT_HEIGHT, MNIST_INPUT_WIDTH),
        Layer::conv2d(
            conv_filters,
            MNIST_IN_CHANNELS,
            conv_kernel_size,
            conv_stride,
            conv_padding,
        ),
        Layer::max_pooling(
            max_pooling_filter_size,
            max_pooling_stride,
            max_pooling_padding,
        ),
        Layer::four_d_to2d(
            conv_filters,
            max_pooling_output_height,
            max_pooling_output_width,
        ),
        Layer::dense((
            conv_filters * max_pooling_output_height * max_pooling_output_width,
            100,
        )),
        Layer::sigmoid(1.),
        Layer::dense((100, MNIST_LABEL_SIZE)),
        Layer::softmax(),
    ];

    Sequential::new(layers)
}
