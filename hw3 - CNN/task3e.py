import pathlib
import matplotlib.pyplot as plt
import utils
import torch
from torch import nn
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import create_plots




class BestModel(nn.Module):
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
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )
        
 

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(-1, self.num_output_features) #Flatten 
        x = self.classifier(x)    
        return x

 


  

    

    

if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc). 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 3e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size)
    scheduling = False

    model = BestModel(image_channels=3, num_classes=10)   
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    
    

    trainer.optimizer = torch.optim.AdamW(trainer.model.parameters(),trainer.learning_rate, weight_decay=0.05)  
    trainer.train()
   
    #last data:
    print("training loss/accuracy:",trainer.trainloss, trainer.trainacc)
    print("validation loss/accuracy:",trainer.valloss, trainer.valacc)
    print("test loss/accuracy:",trainer.testloss, trainer.testacc)
 
    
     
    #figure plot:
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    plt.ylim([0.0, 1.5])
    utils.plot_loss(trainer.train_history["loss"], label="Training loss")
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    utils.plot_loss(trainer.test_history["loss"], label="Test loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    plt.ylim([0.65, 0.9])
    utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.test_history["accuracy"], label="Test Accuracy")
    plt.legend()
    plt.savefig("task3e.png")
    plt.show()
