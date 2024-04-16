# Training LeNet-5 and Custom MLP on MNIST Dataset
In this report, I will describe the process of training LeNet-5 and a custom MLP (Multi-Layer Perceptron) on the MNIST dataset. The goal is to compare their predictive performances and explore regularization techniques to improve the LeNet-5 model.

## Author: HyeJung Moon, hyejung.moon@gmail.com, 23620026, 2024.04.16

## 1. Dataset Preparation
I implemented a custom MNIST dataset class in dataset.py. This class loads the MNIST images, applies necessary preprocessing (such as normalization), and provides data to the models during training and testing. The dataset contains handwritten digit images (28x28 pixels) along with their corresponding labels (0 to 9).
data download to C:\Users\userID\MNIST2024\data from eclass of SoulTech

## 2. Model Implementation
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

## 3. Training Process
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
I employed at least two regularization techniques to improve the LeNet-5 model:

- Dropout: Applied dropout layers to prevent overfitting.
- Weight Decay (L2 Regularization): Added weight decay to the optimizer.

Conclusion
![poster](./plot.jpg)

In summary, I successfully trained LeNet-5 and the custom MLP on the MNIST dataset, compared their performances, and explored regularization techniques.
The results are presented in the accompanying code and plots. 
