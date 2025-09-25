import time
from sklearn.metrics import f1_score


def get_performance_metrics():
    """Calculates and returns performance metrics."""
    start_time = time.time()

    # Mock data for demonstration
    y_true = [0, 1, 1, 0, 1, 0]
    y_pred = [0, 1, 0, 0, 1, 1]

    f1 = f1_score(y_true, y_pred)
    end_time = time.time()
    time_taken = end_time - start_time

    return {
        "f1_score": f1,
        "time_taken": time_taken
    }