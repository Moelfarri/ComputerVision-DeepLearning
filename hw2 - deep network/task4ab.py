import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    neurons_per_layer = [32, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)

    ###Model with 32 neurons in hidden layer###
    model = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history, val_history = trainer.train(num_epochs)
    
    
 
    
    print("32 neurons")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model.forward(X_val)))
    
    
    ###Model with 128 neurons in hidden layer###
    neurons_per_layer = [128, 10]
    model1 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer1 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model1, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history1, val_history1 = trainer1.train(num_epochs)
    
    
 
    
    print("128 neurons")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    
    #Plotting
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .6])
    utils.plot_loss(train_history["loss"],"Model with 32 neurons")
    utils.plot_loss(train_history1["loss"], "Model with 128 neurons")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training loss")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history["accuracy"], "Model with 32 neurons")
    utils.plot_loss(train_history1["accuracy"], "Model with 128 neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")
    plt.legend()
    #plt.savefig("task4ab_training_comparison.png")
    plt.show()
    
    
    
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .6])
    utils.plot_loss(val_history["loss"],"Model with 32 neurons")
    utils.plot_loss(val_history1["loss"], "Model with 128 neurons")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation loss")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(val_history["accuracy"], "Model with 32 neurons")
    utils.plot_loss(val_history1["accuracy"], "Model with 128 neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task4ab_validation_comparison.png")
    plt.show()
    
    
    ######Naked model - no improvments######
    neurons_per_layer = [64, 10]
    model_naked = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer_naked = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model_naked, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history_naked, val_history_naked = trainer_naked.train(num_epochs)
    
    print("just basic")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model_naked))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model_naked))
    print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model_naked.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model_naked.forward(X_train)))    
    
    
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .6])
    utils.plot_loss(train_history["loss"],"Training - 32 neurons")
    utils.plot_loss(train_history_naked["loss"], "Training - 64 neurons")
    utils.plot_loss(train_history1["loss"], "Training - 128 neurons")
    utils.plot_loss(val_history["loss"],"Validation - 32 neurons")
    utils.plot_loss(val_history_naked["loss"],"Validation - 64 neurons")
    utils.plot_loss(val_history1["loss"], "Validation - 128 neurons")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation loss")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history["accuracy"], "Training - 32 neurons")
    utils.plot_loss(train_history_naked["accuracy"], "Training - 64 neurons")
    utils.plot_loss(train_history1["accuracy"], "Training - 128 neurons")
    utils.plot_loss(val_history["accuracy"], "Validation - 32 neurons")
    utils.plot_loss(val_history_naked["accuracy"], "Validation - 64 neurons")
    utils.plot_loss(val_history1["accuracy"], "Validation - 128 neurons")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    plt.savefig("task4ab_validation_training_comparison.png")
    plt.show()