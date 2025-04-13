import numpy as np
import matplotlib.pyplot as plt

def visualize_weights(model):
    """
    可视化模型第一层隐藏层的权重。
    将权重矩阵从 (3072, hidden_size) 重排为 (hidden_size, 32, 32, 3)
    """
    hidden_size = model.params['b1'].shape[0]
    W1 = model.params['W1'].reshape(32, 32, 3, hidden_size).transpose(3, 0, 1, 2)

    plt.figure(figsize=(12, 6))
    num_to_show = min(10, hidden_size)
    for i in range(num_to_show):
        plt.subplot(2, 5, i + 1)
        w = W1[i]
        w_min, w_max = np.min(w), np.max(w)
        w = (w - w_min) / (w_max - w_min)
        plt.imshow(w)
        plt.axis('off')
    plt.suptitle("Visualization of First Hidden Layer Weights")
    plt.tight_layout()
    plt.show()
