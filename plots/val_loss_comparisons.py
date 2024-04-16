from utils import str_to_float, plot_multiple_lines, plot_zoom_multiple_lines


def plot_latent_comparison():
    loss_values = []
    zoom_loss_values = []

    filenames = [
        "loss_ep_75_lat_3_bs_8_full_test_data.csv",
        "loss_ep_75_lat_9_bs_8_full_test_data.csv",
        "loss_ep_75_lat_16_bs_8_full_test_data.csv",
        "loss_ep_75_lat_32_bs_8_full_test_data.csv",
        "loss_ep_75_lat_64_bs_8_full_test_data.csv",
        "loss_ep_75_lat_128_bs_8_full_test_data.csv",
    ]
    loss_colors = [
        'darkturquoise',
        'plum',
        'goldenrod',
        'coral',
        'lightgreen',
        'royalblue'
    ]

    loss_labels = [
        'Latent Dimension 3',
        'Latent Dimension 9',
        'Latent Dimension 16',
        'Latent Dimension 32',
        'Latent Dimension 64',
        'Latent Dimension 128',
    ]

    for f in filenames:
        path = f'../results/test_latents/{f}'
        with open(path, 'r') as fd:
            rows = fd.readlines()
            metrics = list(map(str_to_float, rows))
            loss_values.append(metrics)
            zoom_loss_values.append(metrics[-40:])

    plot_multiple_lines(
        loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Latent Layer Dimension",
        "Epoch",
        "Loss (mse)",
        "A_latent_analysis_test_data_loss.png"
    )
    plot_zoom_multiple_lines(
        zoom_loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Latent Layer Dimension",
        "Epoch",
        "Loss (mse)",
        "A_latent_analysis_test_data_loss_zoom.png"
    )


def plot_architecture_comparison():
    loss_values = []
    zoom_loss_values = []

    filenames = [
        "loss_ep_75_kf_len_3_bs_8_full_test_data.csv",
        "loss_ep_75_kf_len_4_bs_8_full_test_data.csv",
        "loss_ep_75_kf_len_5_bs_8_full_test_data.csv",
    ]
    loss_colors = [
        'darkturquoise',
        'plum',
        'goldenrod',
    ]

    loss_labels = [
        'Filters 16, 32, 64',
        'Filters 16, 32, 64, 128',
        'Filters 16, 32, 64, 128, 256',
    ]

    for f in filenames:
        path = f'../results/test_architectures/{f}'
        with open(path, 'r') as fd:
            rows = fd.readlines()
            metrics = list(map(str_to_float, rows))
            loss_values.append(metrics)
            zoom_loss_values.append(metrics[-40:])

    plot_multiple_lines(
        loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Filters for Architecture",
        "Epoch",
        "Loss (mse)",
        "A_arch_analysis_test_data_loss.png"
    )
    plot_zoom_multiple_lines(
        zoom_loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Filters for Architecture",
        "Epoch",
        "Loss (mse)",
        "A_arch_analysis_test_data_loss_zoom.png"
    )


def plot_adam_comparison():
    loss_values = []
    zoom_loss_values = []

    filenames = [
        "loss_ep_75_lr_1e-05_bs_8_full_test_data.csv",
        "loss_ep_75_lr_0.0001_bs_8_full_test_data.csv",
        "loss_ep_75_lr_0.001_bs_8_full_test_data.csv",
        "loss_ep_75_lr_0.01_bs_8_full_test_data.csv",
        "loss_ep_75_lr_0.1_bs_8_full_test_data.csv",
    ]
    loss_colors = [
        'darkturquoise',
        'plum',
        'goldenrod',
        'coral',
        'lightgreen',
    ]

    loss_labels = [
        'Adam LR 0.00001',
        'Adam LR 0.0001',
        'Adam LR 0.001',
        'Adam LR 0.01',
        'Adam LR 0.1',
    ]

    for f in filenames:
        path = f'../results/test_adam_lr/{f}'
        with open(path, 'r') as fd:
            rows = fd.readlines()
            metrics = list(map(str_to_float, rows))
            loss_values.append(metrics)
            zoom_loss_values.append(metrics[-40:])

    plot_multiple_lines(
        loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Adam Learning Rate",
        "Epoch",
        "Loss (mse)",
        "A_adam_lr_analysis_test_data_loss.png"
    )
    plot_zoom_multiple_lines(
        zoom_loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Adam Learning Rate",
        "Epoch",
        "Loss (mse)",
        "A_adam_lr_analysis_test_data_loss_zoom.png"
    )


def plot_relu_alpha_comparison():
    loss_values = []
    zoom_loss_values = []

    filenames = [
        "loss_ep_75_relu_alpha_0.01_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_0.05_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_0.1_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_0.3_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_0.6_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_0.9_bs_8_full_test_data.csv",
        "loss_ep_75_relu_alpha_1.3_bs_8_full_test_data.csv",
    ]
    loss_colors = [
        'darkturquoise',
        'plum',
        'goldenrod',
        'coral',
        'lightgreen',
        'royalblue',
        'mediumpurple'
    ]

    loss_labels = [
        'Leaky Relu Alpha 0.01',
        'Leaky Relu Alpha 0.05',
        'Leaky Relu Alpha 0.1',
        'Leaky Relu Alpha 0.3',
        'Leaky Relu Alpha 0.6',
        'Leaky Relu Alpha 0.9',
        'Leaky Relu Alpha 1.3',
    ]

    for f in filenames:
        path = f'../results/test_leaky_relu_alpha/{f}'
        with open(path, 'r') as fd:
            rows = fd.readlines()
            metrics = list(map(str_to_float, rows))
            loss_values.append(metrics)
            zoom_loss_values.append(metrics[-40:])

    plot_multiple_lines(
        loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Leaky Relu Alpha",
        "Epoch",
        "Loss (mse)",
        "A_relu_alpha_analysis_test_data_loss.png"
    )
    plot_zoom_multiple_lines(
        zoom_loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Leaky Relu Alpha",
        "Epoch",
        "Loss (mse)",
        "A_relu_alpha_analysis_test_data_loss_zoom.png"
    )


def plot_bs_comparison():
    loss_values = []
    zoom_loss_values = []

    filenames = [
        "loss_ep_75_bs_4_full_test_data.csv",
        "loss_ep_75_bs_8_full_test_data.csv",
        "loss_ep_75_bs_12_full_test_data.csv",
        "loss_ep_75_bs_32_full_test_data.csv",
    ]
    loss_colors = [
        'darkturquoise',
        'plum',
        'goldenrod',
        'coral',
    ]

    loss_labels = [
        'Batch Size 4',
        'Batch Size 8',
        'Batch Size 12',
        'Batch Size 32',
    ]

    for f in filenames:
        path = f'../results/test_batch_size/{f}'
        with open(path, 'r') as fd:
            rows = fd.readlines()
            metrics = list(map(str_to_float, rows))
            loss_values.append(metrics)
            zoom_loss_values.append(metrics[-40:])

    plot_multiple_lines(
        loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Batch Size",
        "Epoch",
        "Loss (mse)",
        "A_batch_size_analysis_test_data_loss.png"
    )
    plot_zoom_multiple_lines(
        zoom_loss_values,
        loss_labels,
        loss_colors,
        "Test Loss vs. Batch Size",
        "Epoch",
        "Loss (mse)",
        "A_batch_size_analysis_test_data_loss_zoom.png"
    )


if __name__ == '__main__':
    # plot_latent_comparison()
    # plot_architecture_comparison()
    # plot_adam_comparison()
    plot_bs_comparison()
    # plot_relu_alpha_comparison()

    print("In order to run comparisons make sure you have appropriate files for it")