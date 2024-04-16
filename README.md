# Training LeNet-5 and Custom MLP on MNIST Dataset
In this report, I will describe the process of training LeNet-5 and a custom MLP (Multi-Layer Perceptron) on the MNIST dataset. The goal is to compare their predictive performances and explore regularization techniques to improve the LeNet-5 model.

## Author: HyeJung Moon, hyejung.moon@gmail.com, 23620026, 2024.04.16

## 1. Dataset Preparation: [dataset.py](https://github.com/MoonHyeJung/NN4DL/blob/main/dataset.py)
I implemented a custom MNIST dataset class in dataset.py. This class loads the MNIST images, applies necessary preprocessing (such as normalization), and provides data to the models during training and testing. The dataset contains handwritten digit images (28x28 pixels) along with their corresponding labels (0 to 9).
data download to C:\Users\userID\MNIST2024\data from eclass of SoulTech

## 2. Model Implementation: [model.py](https://github.com/MoonHyeJung/NN4DL/blob/main/model.py)
### LeNet-5
LeNet-5 is a classic convolutional neural network (CNN) architecture proposed by Yann LeCun et al. It consists of two convolutional layers followed by average pooling, and then three fully connected layers. The number of model parameters in LeNet-5 can be computed as follows:
![poster](./leNet-5.png)
reference: LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based learning applied to document recognition. Proceedings of the IEEE, 86(11), 2278-2324.

#### First Convolutional Layer:
- Input: 1 channel (grayscale image)
- Output: 6 feature maps
- Kernel size: 5x5
- Parameters: (5x5x1)x6 = 150
#### Second Convolutional Layer:
- Input: 6 feature maps
- Output: 16 feature maps
- Kernel size: 5x5
- Parameters: (5x5x6)x16 = 2400
#### Fully Connected Layers:
- First FC layer: 16x4x4 (flattened feature maps) -> 120 units
- Second FC layer: 120 -> 84 units
- Third FC layer: 84 -> 10 units (output classes)
- Parameters: (16x4x4)x120 + 120x84 + 84x10 = 61720

Total parameters in LeNet-5: 61720

### Custom MLP
I implemented a custom MLP with a similar number of parameters as LeNet-5. The architecture consists of three fully connected layers:

#### First Convolutional Layer:
- Input: 1 channel (grayscale image)
- Output: 6 feature maps
- Kernel size: 5x5
- Parameters: (5*5*1+1)*6 = 156
#### Second Convolutional Layer:
- Input: 6 feature maps
- Output: 16 feature maps
- Kernel size: 5x5
- Parameters: (5*5*6+1)*16 = 2416
#### Fully Connected Layers:
- First FC layer: 16x4x4 (flattened feature maps) -> 120 units
- Second FC layer: 120 -> 84 units
- Third FC layer: 84 -> 10 units (output classes)
- Parameters: (16x5x5)x120 + 120x84 + 84x10 = 123,412

Total parameters in Custom MLP: 123,412

## 3. Training Process: [main.py](https://github.com/MoonHyeJung/NN4DL/blob/main/main.py)
I wrote main.py to train both models.
The training process includes monitoring average loss values and accuracy at the end of each epoch.
I used the test dataset as a validation dataset during training.

## 4. Results
I plotted the following statistics for both models:
- Training loss and accuracy curves for each modes
- Test loss and accuracy curves for each modes
![poster](./plot0.jpg)

## 5. Performance Comparison
I compared the predictive performances of LeNet-5 and the custom MLP.
Additionally, I verified that the accuracy of my LeNet-5 implementation matches the known accuracy reported in the literature.
![poster](./table.png)
- added perforemance with new model by regularization

## 6. Regularization Techniques
### 6.1 LeNet5moon
- Increased depth: LeNet5moon has a deeper network structure. The added Convolutional Layer and Fully Connected Layer allow learning more abstract features, which improves model performance.
- Introducing Batch Normalization: LeNet5moon introduces Batch Normalization after each convolutional layer to stabilize learning and speed up convergence. This allows for more efficient learning.
- Apply Dropout: Add Dropout between Fully Connected Layers to prevent overfitting and improve the generalization ability of the model. This helps build more stable and generalized models.

### 6.2 LeNet5moon2
- Increased number of filters and feature maps: LeNet5moon2 can extract more features by using more filters in each convolutional layer. This allows for learning more diverse features and more accurate classification.
- Adjusted network size: LeNet5moon2 adjusts the network size according to the size of the input image to provide a more suitable model. This allows it to provide better generalization performance for different sizes and types of input images.
- Apply Batch Normalization: LeNet5moon2 also introduces Batch Normalization after each convolutional layer to stabilize learning and increase convergence speed. This allows for faster and more stable learning.
- Architecture of LeNet5moon
Input (1, 32, 32)

| Layer          | Operation           | Output Shape |
|----------------|---------------------|--------------|
| Conv2d         | (1, 6, 5, 5)        | (6, 28, 28)  |
| BatchNorm2d    |                     |              |
| ReLU           |                     |              |
| MaxPool2d      | (2, 2)              | (6, 14, 14)  |
| Conv2d         | (6, 16, 5, 5)       | (16, 10, 10) |
| BatchNorm2d    |                     |              |
| ReLU           |                     |              |
| MaxPool2d      | (2, 2)              | (16, 5, 5)   |
| Flatten        |                     | (400,)       |
| Linear         | (400, 120)          | (120,)       |
| Dropout        | (p=0.5)             |              |
| Linear         | (120, 84)           | (84,)        |
| Dropout        | (p=0.5)             |              |
| Linear         | (84, 10)            | (10,)        |

I development LeNet5moon2 using LeNet5moon.
- Performance and speed have been improved.
- Architecture of LeNet5moon2
Input (1, 32, 32)

| Layer          | Operation           | Output Shape |
|----------------|---------------------|--------------|
| Conv2d         | (1, 10, 5, 5)       | (10, 28, 28) |
| BatchNorm2d    |                     |              |
| ReLU           |                     |              |
| MaxPool2d      | (2, 2)              | (10, 14, 14) |
| Conv2d         | (10, 32, 5, 5)      | (32, 10, 10) |
| BatchNorm2d    |                     |              |
| ReLU           |                     |              |
| MaxPool2d      | (2, 2)              | (32, 5, 5)   |
| Flatten        |                     | (800,)       |
| Linear         | (800, 120)          | (120,)       |
| Dropout        | (p=0.5)             |              |
| Linear         | (120, 84)           | (84,)        |
| Dropout        | (p=0.5)             |              |
| Linear         | (84, 10)            | (10,)        |

Conclusion
![poster](./plot.jpg)

In summary, I successfully trained LeNet-5 and the custom MLP on the MNIST dataset, compared their performances, and explored regularization techniques.
The results are presented in the accompanying code and plots. 
