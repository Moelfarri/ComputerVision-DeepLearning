import pathlib
import matplotlib.pyplot as plt
import utils
import torch
import torchvision
from torch import nn
#from pytorch_model_summary import summary
from dataloaders import load_cifar10
from trainer import Trainer, compute_loss_and_accuracy
from task2 import * 


class TransferLearningModel(nn.Module):

    def __init__(self, num_classes):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        self.num_classes = num_classes
        
        self.model = torchvision.models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(512,self.num_classes) #no need to apply softmax as this is done in nn.CrossEntropyLoss
        
        #for Transfer learning:
        for param in self.model.parameters(): #freeze all parameters
            param.requires_grad = False
        for param in self.model.fc.parameters(): #unfreeze last fully connected layer
            param.requires_grad = True
        
        for param in self.model.layer4.parameters(): #Unfreeze the last 5 convolutional layers to alter spefically towards our dataset
            param.requires_grad = True
        
        
        
    def forward(self, x):
        """
        Performs a forward pass through the model
        Args:
            x: Input image
        """
        out = self.model(x) 
        batch_size = x.shape[0]
        expected_shape = (batch_size, self.num_classes)
        assert out.shape == (batch_size, self.num_classes),\
            f"Expected output of forward pass to be: {expected_shape}, but got: {out.shape}"
        return out
    
    

    
    

if __name__ == "__main__":
    # Set the random generator seed (parameters, shuffling etc).
    # You can try to change this and check if you still get the same result! 
    utils.set_seed(0)
    epochs = 10
    batch_size = 32
    learning_rate = 5e-4
    early_stop_count = 4
    dataloaders = load_cifar10(batch_size,task4a=True)
    model =  TransferLearningModel(num_classes=10)
    
    
    
    
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        model,
        dataloaders
    )
    
    trainer.optimizer = torch.optim.Adam(trainer.model.parameters(),trainer.learning_rate)
    
    
    trainer.train()
    create_plots(trainer, "task4a")
    
    #last data:
    print("training loss/accuracy:",trainer.trainloss, trainer.trainacc)
    print("validation loss/accuracy:",trainer.valloss, trainer.valacc)
    print("test loss/accuracy:",trainer.testloss, trainer.testacc)

    