import numpy as np
import matplotlib.pyplot as plt
from three_layer_nn import ThreeLayerNet
from data_loader import load_CIFAR10, preprocess_data

def accuracy(y_pred, y_true):
    """计算预测准确率"""
    return np.mean(y_pred == y_true)

def sgd_update(params, grads, learning_rate):
    for key in params:
        params[key] -= learning_rate * grads[key]

def train(model, X_train, y_train, X_val, y_val,
          learning_rate=1e-3, learning_rate_decay=0.95,
          num_epochs=10, batch_size=200, verbose=True):
    num_train = X_train.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    best_val_acc = 0.0
    best_params = {}

    for epoch in range(num_epochs):
        idx = np.arange(num_train)
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train = y_train[idx]

        for i in range(iterations_per_epoch):
            batch_indices = np.random.choice(num_train, batch_size)
            X_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            loss, grads = model.loss(X_batch, y_batch)
            sgd_update(model.params, grads, learning_rate)

        y_val_pred = model.predict(X_val)
        val_acc = accuracy(y_val_pred, y_val)
        print(f"Epoch {epoch + 1}/{num_epochs}, Validation Accuracy: {val_acc:.4f}")
        model.val_accuracy_history.append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params = {k: v.copy() for k, v in model.params.items()}

        learning_rate *= learning_rate_decay

    model.params = best_params
    model.plot_loss_and_accuracy()
    return model, best_val_acc

def hyperparameter_search(X_train, y_train, X_val, y_val, input_size, output_size):
    learning_rates = [1e-3, 5e-4]
    hidden_sizes = [50, 100]
    regs = [0.0, 1e-3]
    results = {}
    best_val_acc = -1
    best_model = None
    all_curves = []  # 用于收集每次训练的曲线数据

    for lr in learning_rates:
        for hs in hidden_sizes:
            for reg in regs:
                print(f"Training with learning_rate={lr}, hidden_size={hs}, reg={reg}")
                model = ThreeLayerNet(input_size=input_size, hidden_size=hs,
                                      output_size=output_size, activation='relu',
                                      weight_scale=1e-2, reg=reg)
                model, val_acc = train(model, X_train, y_train, X_val, y_val,
                                       learning_rate=lr, num_epochs=5, batch_size=200, verbose=False)
                results[(lr, hs, reg)] = val_acc
                print(f"验证集准确率: {val_acc:.4f}")
                label = f"lr={lr}, hs={hs}, reg={reg}"
                curves = {
                    "label": label,
                    "loss_history": model.loss_history.copy(),
                    "val_accuracy_history": model.val_accuracy_history.copy()
                }
                all_curves.append(curves)
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model = model
    print(f"最佳验证集准确率: {best_val_acc:.4f}")
    return best_model, results, all_curves

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_acc = accuracy(y_pred, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    return test_acc
