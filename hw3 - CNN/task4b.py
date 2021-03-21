
import matplotlib.pyplot as plt
from PIL import Image
import torchvision
import torch
import numpy as np
import skimage.transform as skit

image = Image.open("images/zebra.jpg")
print("Image shape:", image.size)

model = torchvision.models.resnet18(pretrained=True)
print(model)
first_conv_layer = model.conv1
print("First conv layer weight shape:", first_conv_layer.weight.shape)
print("First conv layer:", first_conv_layer)

# Resize, and normalize the image with the mean and standard deviation
image_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
image = image_transform(image)[None]
print("Image shape:", image.shape)

activation = first_conv_layer(image)
print("Activation shape:", activation.shape)


def torch_image_to_numpy(image: torch.Tensor):
    """
    Function to transform a pytorch tensor to numpy image
    Args:
        image: shape=[3, height, width]
    Returns:
        iamge: shape=[height, width, 3] in the range [0, 1]
    """
    # Normalize to [0 - 1.0]
    image = image.detach().cpu() # Transform image to CPU memory (if on GPU VRAM)
    image = image - image.min()
    image = image / image.max()
    image = image.numpy()
    if len(image.shape) == 2: # Grayscale image, can just return
        return image
    assert image.shape[0] == 3, "Expected color channel to be on first axis. Got: {}".format(image.shape)
    image = np.moveaxis(image, 0, 2)
    return image


#Task 4b
indices = [14, 26, 32, 49, 52]
fig = plt.figure(figsize=(20,10))
axis = np.array(range(len(indices))) + 1

for i,ax  in zip(indices,axis):
    kernel     = model.conv1.weight[i,:,:,:]
    activation = model.conv1.forward(image)[0,i,:,:] #0th and only picture is zebra.
    
    
    fig.add_subplot(2, len(indices), ax)
    fig.add_subplot(2, len(indices), ax).set_title(str(i),fontsize=60)
    plt.imshow(torch_image_to_numpy(kernel)) #kernel image
    
    fig.add_subplot(2, len(indices), len(indices)+ax)
    plt.imshow(torch_image_to_numpy(activation),cmap="gray") #activation image

fig.tight_layout(pad=0.1,h_pad=-10)
#plt.savefig("task4b.png")
plt.show()

#Task 4c
model = torchvision.models.resnet18(pretrained=True)
model_4c = torch.nn.Sequential(*list(model.children())[:-2]) #this model has the normalized layer after the last convolutional layer. The task says thats fine.
activations = model_4c.forward(image)

fig = plt.figure(figsize=(100, 40))
for i in range(10):    
    fig.add_subplot(3, 10, i+1)
    #grayscale
    fig.add_subplot(3, 10, i+1).set_title(str(i),fontsize=100)
    plt.imshow(torch_image_to_numpy(activations[0,i,:,:]),cmap="gray") 
    
    #alpha 0.2
    fig.add_subplot(3, 10, 11+i)
    resized_activation = skit.resize(torch_image_to_numpy(activations[0,i,:,:]),  (224, 224), order=0)
    plt.imshow(resized_activation,cmap="gray") 
    plt.imshow(torch_image_to_numpy(image[0]),alpha=0.2)
    
    #alpha 0.4
    fig.add_subplot(3, 10, 21+i)
    resized_activation = skit.resize(torch_image_to_numpy(activations[0,i,:,:]),  (224, 224), order=0)
    plt.imshow(resized_activation,cmap="gray") 
    plt.imshow(torch_image_to_numpy(image[0]),alpha=0.4)

 
    
fig.tight_layout(h_pad=-80,w_pad=10)
plt.savefig("task4c.png")
plt.show()