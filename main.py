from data_loader import load_CIFAR10, preprocess_data
from training import hyperparameter_search, evaluate_model
from visualization import visualize_weights
import pickle

if __name__ == '__main__':
    cifar10_dir = './cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    num_training = 45000
    num_validation = 5000
    X_val = X_train[num_training:num_training + num_validation]
    y_val = y_train[num_training:num_training + num_validation]
    X_train = X_train[:num_training]
    y_train = y_train[:num_training]

    X_train, X_val, X_test = preprocess_data(X_train, X_val, X_test)
    input_size = X_train.shape[1]  # 3072
    output_size = 10

    # 进行超参数查找，返回最佳模型、验证结果和所有训练曲线数据
    best_model, results, all_curves = hyperparameter_search(X_train, y_train, X_val, y_val, input_size, output_size)

    evaluate_model(best_model, X_test, y_test)

    # 可视化权重
    visualize_weights(best_model)

    with open('best_model.pkl', 'wb') as f:
        pickle.dump(best_model.params, f)
    print("训练完成，最佳模型已保存为 best_model.pkl。")
