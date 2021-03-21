import utils
import matplotlib.pyplot as plt
from task2a import pre_process_images, one_hot_encode, SoftmaxModel, cross_entropy_loss
from task2 import SoftmaxTrainer, calculate_accuracy


if __name__ == "__main__":
    # hyperparameters DO NOT CHANGE IF NOT SPECIFIED IN ASSIGNMENT TEXT
    num_epochs = 50
    learning_rate = .1
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

    
    ######Naked model - no improvments######
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
    
    
    
    ######1nd model - improved weights######
    use_improved_sigmoid = False
    use_improved_weight_init = True
    use_momentum = False
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
    
    print("Improved weights")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model))
    print("Final Validation Cross Entropy Loss:",cross_entropy_loss(Y_val, model.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model.forward(X_train)))
    
    #Plotting
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task3a_improvedweight.png")
    plt.show()
 
    
    ######2nd model - improved weights & improved sigmoid######
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = False
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

    
    print("Improved weights & improved sigmoid")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model1))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model1))
    print("Final Validation Cross Entropy Loss:",cross_entropy_loss(Y_val, model1.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model1.forward(X_train)))
    
    #Plotting
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
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history1["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history1["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task3b_improvedweight_and_improvedsigmoid.png")
    plt.show()
    
    
    ######3nd model - improved weights & improved sigmoid & momentum######
    use_improved_sigmoid = True
    use_improved_weight_init = True
    use_momentum = True
    learning_rate = 0.02
    
    model2 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer2 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model2, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history2, val_history2 = trainer2.train(num_epochs)

    
    print("Improved weights & improved sigmoid & momentum")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model2))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model2))
    print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model2.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model2.forward(X_train)))
    
    #Plotting
    plt.figure(figsize=(20, 12))
    plt.subplot(1, 2, 1)
    plt.ylim([0., .5])
    utils.plot_loss(train_history2["loss"],
                    "Training Loss", npoints_to_average=10)
    utils.plot_loss(val_history2["loss"], "Validation Loss")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Cross Entropy Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.91, 1.01])
    utils.plot_loss(train_history2["accuracy"], "Training Accuracy")
    utils.plot_loss(val_history2["accuracy"], "Validation Accuracy")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Accuracy")
    plt.legend()
    #plt.savefig("task3c_improvedweight_and_improvedsigmoid_and_momentum.png")
    plt.show()
    
    
    ######4nd model - improved sigmoid ######
    use_improved_sigmoid = True
    use_improved_weight_init = False
    use_momentum = False
    learning_rate = .1
    
    model3 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer3 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model3, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history3, val_history3 = trainer3.train(num_epochs)

    
    print("just improved sigmoid")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model3))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model3))
    print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model3.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model3.forward(X_train)))
    
    
    ######5nd model - Momentum ######
    use_improved_sigmoid = False
    use_improved_weight_init = False
    use_momentum = True
    learning_rate = 0.02
    
    model4 = SoftmaxModel(
        neurons_per_layer,
        use_improved_sigmoid,
        use_improved_weight_init)
    trainer4 = SoftmaxTrainer(
        momentum_gamma, use_momentum,
        model4, learning_rate, batch_size, shuffle_data,
        X_train, Y_train, X_val, Y_val,
    )
    train_history4, val_history4 = trainer4.train(num_epochs)

    
    print("just momentum")
    print("Train accuracy:", calculate_accuracy(X_train, Y_train, model4))
    print("Validation accuracy:", calculate_accuracy(X_val, Y_val, model4))
    print("Final Validation Cross Entropy Loss:", cross_entropy_loss(Y_val, model4.forward(X_val)))
    print("Final Train Cross Entropy Loss:", cross_entropy_loss(Y_train, model4.forward(X_train)))    
    
    
    #Plot Validation Buildup:
    plt.figure(figsize=(20, 12))
    #plot loss
    plt.subplot(1, 2, 1)
    plt.ylim([0, 0.56])
    utils.plot_loss(val_history_naked["loss"], "Basic model")
    utils.plot_loss(val_history["loss"], "Model with IW")
    utils.plot_loss(val_history3["loss"], "Model with IS")
    utils.plot_loss(val_history4["loss"], "Model with M")
    utils.plot_loss(val_history1["loss"], "Model with IW & IS")
    utils.plot_loss(val_history2["loss"], "Model with IW & IS & M")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.885, 0.975])
    utils.plot_loss(val_history_naked["accuracy"], "Basic model")
    utils.plot_loss(val_history["accuracy"], "Model with IW")
    utils.plot_loss(val_history3["accuracy"], "Model with IS")
    utils.plot_loss(val_history4["accuracy"], "Model with M")
    utils.plot_loss(val_history1["accuracy"], "Model with IW & IS")
    utils.plot_loss(val_history2["accuracy"], "Model with IW & IS & M")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Validation Accuracy")
    plt.legend()
    #plt.savefig("task3_buildup_validation_accuracy_loss.png")
    plt.show()
    
    
    
    #Plot Training Buildup:
    plt.figure(figsize=(20, 12))
    #plot loss
    plt.subplot(1, 2, 1)
    plt.ylim([0, 0.9])
    utils.plot_loss(train_history_naked["loss"], "Basic model")
    utils.plot_loss(train_history["loss"], "Model with IW")
    utils.plot_loss(train_history3["loss"], "Model with IS")
    utils.plot_loss(train_history4["loss"], "Model with M")
    utils.plot_loss(train_history1["loss"], "Model with IW & IS")
    utils.plot_loss(train_history2["loss"], "Model with IW & IS & M")
    plt.legend()
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Loss - Average")
    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.ylim([0.95, 1.01])
    utils.plot_loss(train_history_naked["accuracy"], "Basic model")
    utils.plot_loss(train_history["accuracy"], "Model with IW")
    utils.plot_loss(train_history3["accuracy"], "Model with IS")
    utils.plot_loss(train_history4["accuracy"], "Model with M")
    utils.plot_loss(train_history1["accuracy"], "Model with IW & IS")
    utils.plot_loss(train_history2["accuracy"], "Model with IW & IS & M")
    plt.xlabel("Number of Training Steps")
    plt.ylabel("Training Accuracy")
    plt.legend()
    #plt.savefig("task3c_buildup_training_accuracy_loss.png")
    plt.show()
    
    
 
    
 

     