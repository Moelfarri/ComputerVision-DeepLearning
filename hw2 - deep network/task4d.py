import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = 0.02
    batch_size = 32
    neurons_per_layer = [64, 10]
    momentum_gamma = .9  # Task 3 hyperparameter
    shuffle_data = True

    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = False

    # Load dataset
    X_train, Y_train, X_val, Y_val = utils.load_full_mnist()
    X_train = pre_process_images(X_train)
    X_val = pre_process_images(X_val)
    Y_train = one_hot_encode(Y_train, 10)
    Y_val = one_hot_encode(Y_val, 10)
    
    
    ######1nd model - network from task 3######
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
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
    
    
    
    ######2nd model - network from task 4d ######
    neurons_per_layer = [60, 60, 10]
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
    
    
    
    
    print("model from 4d")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))
    print("Final Validation Cross Entropy Loss:",
          cross_entropy_loss(Y_val, model1.forward(X_val)))
    
    #Plotting training and validation loss from 4d model:
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history1["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history1["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.03])
    utils.plot_loss(train_history1["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history1["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task4d_two_hidden_layers.png")
    plt.show()
 
    
     
  
    #Plotting training/validation - loss/accuracy comparing the two models:
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],"Training - 1 hidden layer")
    utils.plot_loss(train_history1["loss"],"Training - 2 hidden layers")
    utils.plot_loss(val_history["loss"],"Validation - 1 hidden layer")
    utils.plot_loss(val_history1["loss"],"Validation - 2 hidden layers")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training/Validation Loss")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history["accuracy"], "Training - 1 hidden layer")
    utils.plot_loss(train_history1["accuracy"], "Training - 2 hidden layers")
    utils.plot_loss(val_history["accuracy"],"Validation - 1 hidden layer")
    utils.plot_loss(val_history1["accuracy"],"Validation - 2 hidden layers")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training/Validation Accuracy")
    plt.legend()
    #plt.savefig("task4d_training_accuracy_comparison.png")
    plt.show()
    
 