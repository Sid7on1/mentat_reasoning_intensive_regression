import numpy as np
import sklearn.metrics as skm
import logging

logger = logging.getLogger(__name__)

# Configuration
VELOCITY_THRESHOLD = 1e-6  # Threshold for considering velocity changes significant
FLOW_THEORY_CONST = 0.5  # Constant in Flow Theory equation

class EvaluationMetrics:
    """
    Class for calculating various evaluation metrics.

    This class implements CCC (Concordance Correlation Coefficient) and NMSE
    (Normalized Mean Square Error) metrics for assessing the performance of
    reasoning-intensive regression models.
    """

    def __init__(self):
        self._y_true = None
        self._y_pred = None

    def concordance_correlation_coefficient(self, y_true, y_pred):
        """
        Compute the Concordance Correlation Coefficient (CCC).

        The CCC measures the agreement between two sets of variables, in this case,
        the true and predicted values. It combines aspects of correlation and
        agreement on the line of perfect agreement.

        Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

        Returns:
        float: CCC value, ranging from -1 to 1.
        """
        self._y_true = y_true
        self._y_pred = y_pred

        # Compute CCC using sklearn's implementation
        return skm.concordance_correlation_coefficient(y_true, y_pred)

    def normalized_mean_square_error(self, y_true, y_pred):
        """
        Calculate the Normalized Mean Square Error (NMSE).

        The NMSE is a relative measure of the prediction error, computed as the
        ratio of the mean square error to the variance of the true values.

        Parameters:
        y_true (array-like): True target values.
        y_pred (array-like): Predicted values.

        Returns:
        float: NMSE value.
        """
        # Store true and predicted values for later use
        self._y_true = y_true
        self._y_pred = y_pred

        # Compute MSE
        mse = np.mean((y_true - y_pred) ** 2)

        # Compute variance of true values
        var_y = np.var(y_true)

        # Calculate NMSE
        nmse = mse / var_y

        return nmse

    def compute_all_metrics(self):
        """
        Compute both CCC and NMSE metrics.

        This method wraps the individual metric computation methods and returns
        the results as a dictionary.
        """
        metrics = {}

        ccc = self.concordance_correlation_coefficient(self._y_true, self._y_pred)
        metrics['ccc'] = ccc

        nmse = self.normalized_mean_square_error(self._y_true, self._y_pred)
        metrics['nmse'] = nmse

        return metrics

    def plot_prediction_distributions(self, ax=None):
        """
        Plot the distribution of true and predicted values for visualization.

        Parameters:
        ax (matplotlib.Axes): Axis object to plot on. If None, a new figure and axis
                            will be created.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(self._y_true, bins=30, alpha=0.5, label='True Values')
        ax.hist(self._y_pred, bins=30, alpha=0.5, label='Predicted Values')

        ax.set_xlabel('Values')
        ax.set_ylabel('Frequency')
        ax.legend()

        return ax

class VelocityEvaluationMetrics(EvaluationMetrics):
    """
    Class for evaluating velocity changes with additional time-based metrics.
    """

    def __init__(self, t_prev, t_curr):
        super().__init__()
        self.t_prev = t_prev
        self.t_curr = t_curr

    def _compute_velocity(self, y):
        """Compute the velocity changes between consecutive time steps."""
        velocity = np.abs(y[self.t_curr] - y[self.t_prev])
        return velocity

    def significant_velocity_changes(self, y_true, y_pred):
        """
        Identify significant velocity changes between consecutive time steps.

        This method uses the defined VELOCITY_THRESHOLD to determine whether the
        velocity changes are significant or not.

        Parameters:
        y_true (array-like): True target values at current time step.
        y_pred (array-like): Predicted values at current time step.

        Returns:
        array-like: Boolean mask indicating significant velocity changes.
        """
        velocity_true = self._compute_velocity(y_true)
        velocity_pred = self._compute_velocity(y_pred)

        # Combine velocity changes for true and pred values
        velocities = np.stack((velocity_true, velocity_pred), axis=1)

        # Apply threshold to consider significant changes
        changes = np.max(velocities, axis=1) > VELOCITY_THRESHOLD

        return changes

    def plot_velocity_scatter(self, ax=None):
        """
        Plot a scatter plot of true vs. predicted velocity changes.

        Parameters:
        ax (matplotlib.Axes): Axis object to plot on. If None, a new figure and axis
                            will be created.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots()

        velocity_true = self._compute_velocity(self._y_true)
        velocity_pred = self._compute_velocity(self._y_pred)

        ax.scatter(velocity_true, velocity_pred)

        ax.set_xlabel('True Velocity Changes')
        ax.set_ylabel('Predicted Velocity Changes')
        ax.set_title('Velocity Changes Scatter Plot')

        return ax

def main():
    # Example usage
    y_true = np.random.rand(100)
    y_pred = np.random.rand(100)

    metrics = EvaluationMetrics()

    # Compute metrics
    ccc = metrics.concordance_correlation_coefficient(y_true, y_pred)
    nmse = metrics.normalized_mean_square_error(y_true, y_pred)

    logger.info(f"CCC: {ccc:.2f}")
    logger.info(f"NMSE: {nmse:.4f}")

    # Plotting distributions
    fig, ax = plt.subplots()
    metrics.plot_prediction_distributions(ax)
    plt.show()

    # Velocity evaluation
    t_prev = 50
    t_curr = 51
    velocity_metrics = VelocityEvaluationMetrics(t_prev, t_curr)

    changes = velocity_metrics.significant_velocity_changes(y_true, y_pred)
    logger.info(f"Significant Velocity Changes: {changes.sum()} instances")

    fig, ax = plt.subplots()
    velocity_metrics.plot_velocity_scatter(ax)
    plt.show()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()