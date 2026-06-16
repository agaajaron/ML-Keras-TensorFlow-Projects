# %%
from config import *


def get_run_logdir(root_logdir="my_logs"):
    import time
    os.makedirs(root_logdir, exist_ok=True)
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


def plot_loss_curves(history, epochs, title="Training vs Validation Loss", convergence_epoch=None):
    N = epochs
    plt.figure(figsize=(8, 6))
    plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
    if convergence_epoch:
        plt.axvline(x=convergence_epoch, color="red", label=f"convergence ~{convergence_epoch}")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


def evaluate_r2(model, scaler, scaled_x, scaled_y):
    preds = model.predict(scaled_x)
    y_pred = scaler.inverse_transform(preds.reshape(-1, 1))
    y_true = scaler.inverse_transform(scaled_y.reshape(-1, 1))
    score = r2_score(y_true, y_pred)
    print(f"R² = {score:.3f}")
    return score
