import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
#from pytorch_model_summary import summary
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots


#(Conv-ReLu-Pool)xN -> LinearxM -> softmax model
class ModelArchitecture1(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        super().__init__()
        num_filters = [32,64,128]
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[0]),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[0],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[0]),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.2),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[1]),
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[1],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[1]),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.3),
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[2]),
            nn.Conv2d(
                in_channels=num_filters[2],
                out_channels=num_filters[2],
                kernel_size=3,
                stride=1,
                padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[2]),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Dropout(p=0.4)
        )

        self.num_output_features = num_filters[-1]*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features,128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features) #Flatten 
        x = self.classifier(x)    
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

    
#(conv-batchnorm-ReLu-conv-batchnorm-Relu-pool)xN -> LinearxM -> softmax model
class ModelArchitecture2(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        super().__init__()
        num_filters = [64,64,64]
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=7,
                stride=1,
                padding=3),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[0],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[2],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.num_output_features = num_filters[-1]*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features,64),
            #nn.Dropout(p=0.1), was better without
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features) #Flatten 
        x = self.classifier(x)    
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
#(batchnorm-relu-conv-relu-pool)xN -> batchnorm-LinearxM -> softmax model    
class ModelArchitecture3(nn.Module):
    def __init__(self,
                 image_channels,
                 num_classes):
        super().__init__()
        num_filters = [64,64,64]
        self.num_classes = num_classes
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                in_channels=image_channels,
                out_channels=num_filters[0],
                kernel_size=7,
                stride=1,
                padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.BatchNorm2d(num_filters[0]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[0],
                out_channels=num_filters[1],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[1]),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters[1],
                out_channels=num_filters[2],
                kernel_size=5,
                stride=1,
                padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(num_filters[2]),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )

        self.num_output_features = num_filters[-1]*4*4
        self.classifier = nn.Sequential(
            nn.Linear(self.num_output_features,64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
        
        #initialize weights with xavier_uniform
        #self.classifier.apply(init_weights) in combination with TanH, didnt perform well

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features) #Flatten 
        x = self.classifier(x)    
        batch_size = x.shape[0]
        out = x
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out

    
    
#Chris' modell
class ChrisModel(nn.Module):

    def __init__(self,
                 image_channels,
                 num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes

        config = {}
        config["label"] = "camel deep cnn"
        config["fc_layers"] = [32]
        config["cnn_layers"] = [32, 64, 128, 64]
        config["conv_kernel"] = [5, 3, 3, 3]
        config["pool_kernel"] = [2, 2, 2, 2]
        config["conv_padding"] = [x // 2 for x in config["conv_kernel"]]  # preserve dimension if odd
        config["conv_stride"] = [1, 1, 1, 1]
        config["pool_stride"] = [2, 2, 2, 2]
        config["pool_padding"] = [0, 0, 0, 0]

        # TODO: Implement this function (Task  2a)
        # init all weights and layers for CNN

        # feature extraction configuration
        cnn_activation_function = nn.ReLU
        image_dim = 32  # square
        conv_kernel = config["conv_kernel"]
        conv_padding = config["conv_padding"]
        conv_stride = config["conv_stride"]
        pool_kernel = config["pool_kernel"]
        pool_stride = config["pool_stride"]
        pool_padding = config["pool_padding"]
        cnn_layers = [image_channels]
        cnn_layers.extend(config["cnn_layers"])

        # convolution preserves dimensions - kernel: 5x5, padding: 2, stride: 1
        # max pool causes 50% dimensional reduction - kernel: 2x2, stride: 2
        new_dimension = lambda layer_dim, padding, filter, stride: (layer_dim + padding - filter) // stride + 1
        dimension_i = image_dim

        # Generate the pure convolutional layers
        num_conv_layers = len(cnn_layers) - 1
        cnn_components = []
        for i in range(num_conv_layers):
            conv = nn.Conv2d(
                in_channels=cnn_layers[i],
                out_channels=cnn_layers[i+1],
                kernel_size=conv_kernel[i],
                stride=conv_stride[i],
                padding=conv_padding[i],
                             )
            activation = cnn_activation_function()
            pool = nn.MaxPool2d(
                kernel_size=pool_kernel[i],
                stride=pool_stride[i],
                padding=pool_padding[i],
                                )
            dimension_i = new_dimension(dimension_i, pool_padding[i], pool_kernel[i], pool_stride[i])
            cnn_components.extend([conv, activation, pool])

        # The output of feature_extractor will be [batch_size, num_filters, map_dim_1, map_dim_2]
        n_outputs = cnn_layers[-1] * dimension_i ** 2
        self.feature_extractor = nn.Sequential(*cnn_components)

        # classifier configuration
        fc_activation_function = nn.ReLU
        self.num_output_features = n_outputs  # used in forward
        fc_layers = [n_outputs]
        fc_layers.extend(config["fc_layers"])
        fc_layers.append(num_classes)

        # generate classifier
        num_fc_layers = len(fc_layers) - 1
        fc_components = []
        # fc_components.append(nn.BatchNorm1d(n_outputs, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))
        for i in range(num_fc_layers):
            # batchnorm = nn.BatchNorm1d(fc_layers[i], eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            fc_layer = nn.Linear(fc_layers[i], fc_layers[i+1])
            activation = fc_activation_function()
            # fc_components.extend([batchnorm, fc_layer, activation])
            fc_components.extend([fc_layer, activation])

        # drop last activation function as we use softmax
        # There is no need for softmax activation function, as this is
        # included with nn.CrossEntropyLoss
        self.classifier = nn.Sequential(*fc_components[:-1])

    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image, shape: [batch_size, 3, 32, 32]
        """
        # TODO: Implement this function (Task  2a)
        batch_size = x.shape[0]
        features = self.feature_extractor(x)  # pure cnn
        features = features.view(-1, self.num_output_features)  # reshape to (batch_size, num_output_features)
        out = self.classifier(features)  # dense neural network
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        
        
def load_model_settings(architecture):
    """
            Initializes the model with the correct hyperparameters/training details
            Args:
                architecture: an int between 1-3
            Returns: 
                correct model class to be trained on
    """
    global batch_size, learning_rate
    
    if architecture == 1:
        batch_size = 32
        learning_rate = 0.001
        return ModelArchitecture1(image_channels=3, num_classes=10)
    
    elif architecture == 2:
        learning_rate = 3e-4 #5e-2 for SGD
        batch_size = 32
        return ModelArchitecture2(image_channels=3, num_classes=10)
    
    elif architecture == 3:
        learning_rate = 5e-2
        batch_size = 32
        return ModelArchitecture3(image_channels=3, num_classes=10)
    
    elif architecture == 4:
        batch_size = 64
        learning_rate = 0.001
        return ChrisModel(image_channels=3, num_classes=10)
    
    else:
        "choose either 1,2 or 3"
        return -1
    
    
def load_train_settings(trainer, architecture):
    """
            Initializes the trainer with the correct details for the architecture model
            Args:
                architecture: an int between 1-3
                trainer: a trainer for a model
    """
    
    if architecture == 1:
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),trainer.learning_rate,momentum=0.9)
         
    elif architecture == 2:
        #trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),trainer.learning_rate)
        trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(),trainer.learning_rate, weight_decay=0.05)  
 
    elif architecture == 3:
        #TODO: try Adadel, RMSprop etc on this architecture
        trainer.optimizer = torch.optim.SGD(trainer.model.parameters(),trainer.learning_rate)
        
    elif architecture == 4:
        trainer.optimizer = torch.torch.optim.AdamW(trainer.model.parameters(),
                                           trainer.learning_rate,
                                           betas=(0.9, 0.999),
                                           eps=1e-08,
                                           weight_decay=0.01,
                                           amsgrad=False)
    else:
        return
    
 
if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-2
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)

    #model for architecture 2 - one of best models
    model2 = load_model_settings(2)
    trainer2 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model2,
        dataloaders
    )
    
    load_train_settings(trainer2, 2)
    trainer2.train()
    


    #last data:
    print("training loss/accuracy:",trainer2.trainloss, trainer2.trainacc)
    print("validation loss/accuracy:",trainer2.valloss, trainer2.valacc)
    print("test loss/accuracy:",trainer2.testloss, trainer2.testacc)
    
    create_plots(trainer2, "task3_arch2_plot")
    
    #model for chris's architectures - one of best models
    dataloaders = load_cifar10(batch_size,chrismodel=True)
    model3 = load_model_settings(4)
    trainer3 = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model3,
        dataloaders
    )
    
    load_train_settings(trainer3, 4)
    trainer3.train()
    


    #last data:
    print("training loss/accuracy:",trainer3.trainloss, trainer3.trainacc)
    print("validation loss/accuracy:",trainer3.valloss, trainer3.valacc)
    print("test loss/accuracy:",trainer3.testloss, trainer3.testacc)

    
    #figure plot:
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer2.train_history["loss"], label="Training loss - model 1")
    utils.plot_loss(trainer3.train_history["loss"], label="Training loss - model 2")
    utils.plot_loss(trainer2.validation_history["loss"], label="Validation loss - model 1")
    utils.plot_loss(trainer3.validation_history["loss"], label="Validation loss - model 2")
    utils.plot_loss(trainer2.test_history["loss"], label="Test loss - model 1")
    utils.plot_loss(trainer3.test_history["loss"], label="Test loss - model 2")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer2.train_history["accuracy"], label="Training Accuracy - model 1")
    utils.plot_loss(trainer3.train_history["accuracy"], label="Training Accuracy - model 2")
    utils.plot_loss(trainer2.validation_history["accuracy"], label="Validation Accuracy - model 1")
    utils.plot_loss(trainer3.validation_history["accuracy"], label="Validation Accuracy - model 2")
    utils.plot_loss(trainer2.test_history["accuracy"], label="Test Accuracy - model 1")
    utils.plot_loss(trainer3.test_history["accuracy"], label="Test Accuracy - model 2")
    plt.legend()
    plt.savefig("task3_model1_and_2.png")
    plt.show()