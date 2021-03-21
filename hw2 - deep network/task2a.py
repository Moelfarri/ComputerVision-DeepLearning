import numpy as np
import utils
import typing
import mnist
np.random.seed(1)



def pre_process_images(X: np.ndarray):
    """
    Args:
        X: images of shape [batch size, 784] in the range (0, 255)
    Returns:
        X: images of shape [batch size, 785] normalized as described in task2a
    """
    # TODO implement this function (Task 2a)
    #Normalizing with mean and standard deviation of the entire training set
    #actual code:
    #X_train,_,_,_ = mnist.load() <-- this way get mean/std of 60000 training images
    #hardcoded mean and std:
    mean   = 33.3184214498299   #np.mean(X_train) 
    std    = 78.56748998339798  #np.std(X_train)  
    X = (X - mean)/std
    
    assert X.shape[1] == 784,\
        f"X.shape[1]: {X.shape[1]}, should be 784"
   

    #add bias trick
    X = np.insert(X, obj=X[0].size, values=1, axis=1)
    return X



def cross_entropy_loss(targets: np.ndarray, outputs: np.ndarray):
    """
    Args:
        targets: labels/targets of each image of shape: [batch size, num_classes]
        outputs: outputs of model of shape: [batch size, num_classes]
    Returns:
        Cross entropy error (float)
    """
    assert targets.shape == outputs.shape,\
        f"Targets shape: {targets.shape}, outputs: {outputs.shape}"
    # TODO: Implement this function (copy from last assignment)
    samples = outputs.shape[0]
    average_cost = -np.sum((targets*np.log(outputs)), axis=1) #sum all rows first
    average_cost = (1/samples)*np.sum(average_cost) #sum all costs and divide my batch size to get mean
    return average_cost



class SoftmaxModel:

    def __init__(self,
                 # Number of neurons per layer
                 neurons_per_layer: typing.List[int],
                 use_improved_sigmoid: bool,  # Task 3a hyperparameter
                 use_improved_weight_init: bool  # Task 3c hyperparameter
                 ):
        # Always reset random seed before weight init to get comparable results.
        np.random.seed(1)
        # Define number of input nodes
        self.I = 785
        self.use_improved_sigmoid = use_improved_sigmoid

        # Define number of output nodes
        # neurons_per_layer = [64, 10] indicates that we will have two layers:
        # A hidden layer with 64 neurons and a output layer with 10 neurons.
        self.neurons_per_layer = neurons_per_layer

        # Initialize the weights
        self.ws = []                                     #Format: [wji, ...., wkj]
        prev = self.I
        if not use_improved_weight_init:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                print("Initializing weight to shape:", w_shape)
                w = np.random.uniform(-1,1,w_shape) #uniformly randomized weights
                self.ws.append(w)
                prev = size
        else:
            for size in self.neurons_per_layer:
                w_shape = (prev, size)
                w_mean = 0
                w_std  = 1/np.sqrt(prev) #sqrt of fan-in/inputs to one neuron
                print("Initializing weight to shape:", w_shape)
                w = np.random.normal(w_mean,w_std,w_shape)
                self.ws.append(w)
                prev = size

        
        
        
        self.grads = [None for i in range(len(self.ws))]  #Format: [xi*dj, ...., aj*dk]
        self.activations = []                             #Format: [xi, aj, ...., y_hat]

        
    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X: images of shape [batch size, 785]
        Returns:
            y: output of model with shape [batch size, num_outputs]
        """
        # TODO implement this function (Task 2b)
        
        #hardcoded version of forward with 2 layers:
        #self.activations = [X]
        #zj = X@self.ws[0]
        #aj = 1/(1+np.exp(-zj))
        #self.activations.append(aj)
        #zk = aj@self.ws[1]
        #output = np.exp(zk) / np.array([np.sum(np.exp(zk),  axis=1)]).T
        #self.activations.append(output)
        
        #Generalized implementation of forward:
        ai = X
        self.activations = [ai]
        #iterate through the counter and weight matrixes connecting the layers:
        for i,wi in zip(range(len(self.ws)),self.ws):
            zi = ai@wi
            
            #if last iteration apply softmax activation instead of sigmoid
            if i == len(self.ws)-1:
                ai = self.softmax(zi)
                output = ai
            else:   
                ai = self.sigmoid(zi)
            self.activations.append(ai)
        return output

    def backward(self, X: np.ndarray, outputs: np.ndarray,
                 targets: np.ndarray) -> None:
        """
        Computes the gradient and saves it to the variable self.grad

        Args:
            X: images of shape [batch size, 785]
            outputs: outputs of model of shape: [batch size, num_outputs]
            targets: labels/targets of each image of shape: [batch size, num_classes]
        """
        # TODO implement this function (Task 2b)
        assert targets.shape == outputs.shape,\
            f"Output shape: {outputs.shape}, targets: {targets.shape}"
        # For example, self.grads[0] will be the gradient for the first hidden layer
        
        #####
        #hardcoded version of backward pass for 2 layers:
        #samples = outputs.shape[0]
        #zj  = X@self.ws[0]#x_i@wji xdim = 100x785, wji_dim = 785x64 multiplied gives zj = 100x64 => aj = f(zj) is 100x64
        #dk  = -(targets-outputs) #dim 100x10
        #wkj = self.ws[1]         #64x10
        #fderv = (1/(1+np.exp(-zj)))*(1-1/(1+np.exp(-zj))) #100x64
        #self.grads[1] = 1/samples*(self.activations[1].T@dk) #64x100 @ 100x10 = 64x10
        #dj  = (dk@wkj.T) *fderv # 100x64 * (100x10 @ (64x10).T) => 100x64
        #self.grads[0] = 1/samples*(X.T@dj) #(100x785).T @ 100x64   => 785x64
        #####
        
        
        #Generalized implementation of backward:
        dk = -(targets-outputs)
        deltas = []
        deltas.append(dk)
        
        #define all deltas using current and previous weights, activation functions and previous deltas
        for aj,wji,wkj,d in zip(reversed(self.activations[:len(self.activations)-2])
                                 ,reversed(self.ws[:len(self.ws)-1]),reversed(self.ws),deltas):
            zj = aj@wji
            delta4_prev_layer =   self.sigmoid_derivative(zj)*(d@wkj.T)
            deltas.append(delta4_prev_layer)
        
        #define gradients using deltas and activation functions
        samples = outputs.shape[0]
        for i,aj, delta in zip(range(len(deltas)),self.activations[:len(self.activations)-1],reversed(deltas)):
            self.grads[i] = 1/samples*(aj.T@delta) 
        
        
                                   
        for grad, w in zip(self.grads, self.ws):
            assert grad.shape == w.shape,\
                f"Expected the same shape. Grad shape: {grad.shape}, w: {w.shape}."
            
            
        
            
    def zero_grad(self) -> None:
        self.grads = [None for i in range(len(self.ws))]
        
        
        
    def sigmoid(self, z: np.array) -> np.array:
        
        #if improved sigmoid is not toggled use normal sigmoid
        if not self.use_improved_sigmoid:
            return 1/(1+np.exp(-z)) 
        else:
            return 1.7159*np.tanh((2.0/3.0)*z)
    
    
    def softmax(self, yhat: np.array) -> np.array:
        return np.exp(yhat) / np.sum(np.exp(yhat),  axis=1,keepdims=True)
        
    def sigmoid_derivative(self, z: np.array) -> np.array:
        
        #if improved sigmoid is not toggled use normal sigmoid derivative
        if not self.use_improved_sigmoid:
            return (1/(1+np.exp(-z))*(1-1/(1+np.exp(-z))))
        else:
            return 1.7159*(2.0/3.0)*(1/(np.cosh((2.0/3.0)*z)*np.cosh((2.0/3.0)*z)))
    

    
    

def one_hot_encode(Y: np.ndarray, num_classes: int):
    """
    Args:
        Y: shape [Num examples, 1]
        num_classes: Number of classes to use for one-hot encoding
    Returns:
        Y: shape [Num examples, num classes]
    """
    # TODO: Implement this function (copy from last assignment)
    return (np.arange(num_classes) == Y).astype(int)



def gradient_approximation_test(
        model: SoftmaxModel, X: np.ndarray, Y: np.ndarray):
    """
        Numerical approximation for gradients. Should not be edited. 
        Details about this test is given in the appendix in the assignment.
    """
    epsilon = 1e-3
    for layer_idx, w in enumerate(model.ws):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                orig = model.ws[layer_idx][i, j].copy()
                model.ws[layer_idx][i, j] = orig + epsilon
                logits = model.forward(X)
                cost1 = cross_entropy_loss(Y, logits)
                model.ws[layer_idx][i, j] = orig - epsilon
                logits = model.forward(X)
                cost2 = cross_entropy_loss(Y, logits)
                gradient_approximation = (cost1 - cost2) / (2 * epsilon)
                model.ws[layer_idx][i, j] = orig
                # Actual gradient
                logits = model.forward(X)
                model.backward(X, logits, Y)
                difference = gradient_approximation - \
                    model.grads[layer_idx][i, j]
                assert abs(difference) <= epsilon**2,\
                    f"Calculated gradient is incorrect. " \
                    f"Layer IDX = {layer_idx}, i={i}, j={j}.\n" \
                    f"Approximation: {gradient_approximation}, actual gradient: {model.grads[layer_idx][i, j]}\n" \
                    f"If this test fails there could be errors in your cross entropy loss function, " \
                    f"forward function or backward function"


if __name__ == "__main__":
    # Simple test on one-hot encoding
    Y = np.zeros((1, 1), dtype=int)
    Y[0, 0] = 3
    Y = one_hot_encode(Y, 10)
    assert Y[0, 3] == 1 and Y.sum() == 1, \
        f"Expected the vector to be [0,0,0,1,0,0,0,0,0,0], but got {Y}"

    X_train, Y_train, *_ = utils.load_full_mnist()
    X_train = X_train[:1]
    Y_train = Y_train[:1]


    X_train = pre_process_images(X_train)
    Y_train = one_hot_encode(Y_train, 10)
    assert X_train.shape[1] == 785,\
        f"Expected X_train to have 785 elements per image. Shape was: {X_train.shape}"

    neurons_per_layer = [64, 10]
    use_improved_sigmoid = False
    use_improved_weight_init = False
    model = SoftmaxModel(
        neurons_per_layer, use_improved_sigmoid, use_improved_weight_init)

    # Gradient approximation check for 100 images
    X_train = X_train[:100]
    Y_train = Y_train[:100]
    for layer_idx, w in enumerate(model.ws):
        model.ws[layer_idx] = np.random.uniform(-1, 1, size=w.shape)

    gradient_approximation_test(model, X_train, Y_train)