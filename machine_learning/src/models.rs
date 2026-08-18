use crate::arch::{Sequential, layers::Layer};

const MNIST_IN_CHANNELS: usize = 1;
const MNIST_INPUT_HEIGHT: usize = 28;
const MNIST_INPUT_WIDTH: usize = 28;
const MNIST_LABEL_SIZE: usize = 10;

fn conv_output_dim(
    kernel_size: usize,
    input_dim: (usize, usize),
    stride: usize,
    padding: usize,
) -> (usize, usize) {
    let output_h = (input_dim.0 + 2 * padding - kernel_size) / stride + 1;
    let output_w = (input_dim.1 + 2 * padding - kernel_size) / stride + 1;
    (output_h, output_w)
}

pub fn make_nielsen_mnist_model() -> Sequential {
    let conv_filters = 10;
    let conv_kernel_size = 5;
    let conv_stride = 1;
    let conv_padding = 0;

    let (_conv_output_height, _conv_output_width) = conv_output_dim(
        conv_kernel_size,
        (MNIST_INPUT_HEIGHT, MNIST_INPUT_WIDTH),
        conv_stride,
        conv_padding,
    );

    let max_pooling_filter_size = 2;
    let max_pooling_stride = 2;
    let max_pooling_padding = 0;

    let (max_pooling_output_height, max_pooling_output_width) = conv_output_dim(
        max_pooling_filter_size,
        (_conv_output_height, _conv_output_width),
        max_pooling_stride,
        max_pooling_padding,
    );

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

pub fn make_lenet5_mnist_model() -> Sequential {
    let unflatten1 = Layer::two_d_to4d(MNIST_IN_CHANNELS, MNIST_INPUT_HEIGHT, MNIST_INPUT_WIDTH);

    let conv1_filters = 6;
    let conv1_kernel_size = 5;
    let conv1_stride = 1;
    let conv1_padding = 2;

    let conv1 = Layer::conv2d(
        conv1_filters,
        MNIST_IN_CHANNELS,
        conv1_kernel_size,
        conv1_stride,
        conv1_padding,
    );

    let (conv1_output_height, conv1_output_width) = conv_output_dim(
        conv1_kernel_size,
        (MNIST_INPUT_HEIGHT, MNIST_INPUT_WIDTH),
        conv1_stride,
        conv1_padding,
    );

    let flatten1 = Layer::four_d_to2d(conv1_filters, conv1_output_height, conv1_output_width);
    let tanh1 = Layer::tanh(1.0);
    let unflatten2 = Layer::two_d_to4d(conv1_filters, conv1_output_height, conv1_output_width);

    let max_pooling1_filter_size = 2;
    let max_pooling1_stride = 2;
    let max_pooling1_padding = 0;

    let max_pooling1 = Layer::max_pooling(
        max_pooling1_filter_size,
        max_pooling1_stride,
        max_pooling1_padding,
    );

    let (max_pooling1_output_height, max_pooling1_output_width) = conv_output_dim(
        max_pooling1_filter_size,
        (conv1_output_height, conv1_output_width),
        max_pooling1_stride,
        max_pooling1_padding,
    );

    let conv2_filters = 16;
    let conv2_kernel_size = 5;
    let conv2_stride = 1;
    let conv2_padding = 0;

    let conv2 = Layer::conv2d(
        conv2_filters,
        conv1_filters,
        conv2_kernel_size,
        conv2_stride,
        conv2_padding,
    );

    let (conv2_output_height, conv2_output_width) = conv_output_dim(
        conv2_kernel_size,
        (max_pooling1_output_height, max_pooling1_output_width),
        conv2_stride,
        conv2_padding,
    );

    let flatten2 = Layer::four_d_to2d(conv2_filters, conv2_output_height, conv2_output_width);
    let tanh2 = Layer::tanh(1.0);
    let unflatten3 = Layer::two_d_to4d(conv2_filters, conv2_output_height, conv2_output_width);

    let max_pooling2_filter_size = 2;
    let max_pooling2_stride = 2;
    let max_pooling2_padding = 0;

    let max_pooling2 = Layer::max_pooling(
        max_pooling2_filter_size,
        max_pooling2_stride,
        max_pooling2_padding,
    );

    let (max_pooling2_output_height, max_pooling2_output_width) = conv_output_dim(
        max_pooling2_filter_size,
        (conv2_output_height, conv2_output_width),
        max_pooling2_stride,
        max_pooling2_padding,
    );

    let flatten3 = Layer::four_d_to2d(
        conv2_filters,
        max_pooling2_output_height,
        max_pooling2_output_width,
    );

    let dense1 = Layer::dense((
        max_pooling2_output_height * max_pooling2_output_width * conv2_filters,
        120,
    ));
    let tanh3 = Layer::tanh(1.0);
    let dense2 = Layer::dense((120, 84));
    let tanh4 = Layer::tanh(1.0);
    let dense3 = Layer::dense((84, 10));
    let softmax = Layer::softmax();

    let layers = [
        unflatten1,
        conv1,
        flatten1,
        tanh1,
        unflatten2,
        max_pooling1,
        conv2,
        flatten2,
        tanh2,
        unflatten3,
        max_pooling2,
        flatten3,
        dense1,
        tanh3,
        dense2,
        tanh4,
        dense3,
        softmax,
    ];

    Sequential::new(layers.to_vec())
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
