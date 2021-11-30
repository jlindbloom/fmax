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



def plot_posterior_predictive(fcast_model):
    """Simple plot of the posterior predictive of a forecast model.
    """

    sample_paths = fcast_model.ppc['running_max']
    index = fcast_model.fcast_index

    # Calculate the 1%, 10%, 50%, 90%, and 99% quantiles
    lower_bound_one = np.quantile(sample_paths, q=0.01, axis=0)
    lower_bound_ten = np.quantile(sample_paths, q=0.1, axis=0)
    medians = np.quantile(sample_paths, q=0.5, axis=0)
    upper_bound_ninety = np.quantile(sample_paths, q=0.9, axis=0)
    upper_bound_ninety_nine = np.quantile(sample_paths, q=0.99, axis=0)

    # Plot
    fig, axs = plt.subplots(1, 2, figsize=(13,8))

    # Plot sample paths on the left
    axs[0].plot(index, sample_paths[:10000,:].T, alpha=0.05)
    axs[0].plot(fcast_model.train_index, fcast_model.train_data, color="red", label="Training")
    axs[0].plot(fcast_model.test_index, fcast_model.test_data, color="black", label="Testing")
    axs[0].legend()
    axs[0].set_xlabel("Period")
    axs[0].set_ylabel("Record")
    axs[0].set_title("Many Posterior Predictive Sample Paths")


    # Plot CI on the right
    axs[1].fill_between(index, lower_bound_one, upper_bound_ninety_nine, alpha=0.4, label="99% CI", color="C0")
    axs[1].fill_between(index, lower_bound_ten, upper_bound_ninety, alpha=0.7, label="80% CI", color="C0")
    axs[1].plot(index, medians, label="Median")
    axs[1].plot(fcast_model.train_index, fcast_model.train_data, color="red", label="Training")
    axs[1].plot(fcast_model.test_index, fcast_model.test_data, color="black", label="Testing")
    axs[1].legend()
    axs[1].set_xlabel("Period")
    axs[1].set_ylabel("Record")
    axs[1].set_title("Credible Interval Over Posterior Predictive Sample Paths")

    fig.tight_layout()

    return fig