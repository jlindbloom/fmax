import matplotlib.pyplot as plt


def plot_train_test(fcast_model):
    """Given a forecasting model, plots the testing vs. training data.
    """
    fig, axs = plt.subplots(figsize=(13,5))
    axs.plot(fcast_model.train_index, fcast_model.train_data, label="Training")
    axs.plot(fcast_model.test_index, fcast_model.test_data, label="Testing")
    axs.legend()
    axs.set_title("Training vs. Testing Data")
    return fig