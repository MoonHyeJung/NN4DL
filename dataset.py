import os, io, tarfile
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image  # Image 모듈을 가져옴

class MNIST(Dataset):
    """ MNIST dataset

    Args:
        data_dir: directory path containing images

    Note:
        1) Each image should be preprocessed as follows:
            - First, all values should be in a range of [0,1]
            - Subtract mean of 0.1307, and divide by std 0.3081
            - These preprocessing steps can be implemented using torchvision.transforms
        2) Labels can be obtained from filenames: {number}_{label}.png
    """

    def __init__(self, data_dir):
        # write your codes here
        self.data_dir = data_dir
        # Implement dataset initialization here
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        self.file_open = tarfile.open(data_dir, 'r')
        # List of image file
        self.png_files = self.file_open.getnames()[1:]
        # List of corresponding labels
        self.labels = [int(os.path.basename(i)[-5]) for i in self.png_files]

    def __len__(self):
        # write your codes here
        return len(self.png_files)

    def __getitem__(self, idx):
        # write your codes here
        png_file = self.png_files[idx]
        label = self.labels[idx]

        # Read image and apply transformations
        extract_file = self.file_open.extractfile(png_file).read()
        image = Image.open(io.BytesIO(extract_file))
        image = self.transform(image)

        return image, label

if __name__ == '__main__':
    # write test codes to verify your implementations
    train_dataset = MNIST(data_dir='../data/train.tar')
    train_image, train_landmarks = train_dataset[0]
    print(f"train image shape: {train_image.shape}")
    print(f"train landmarks: {train_landmarks}")
    test_dataset = MNIST(data_dir='../data/test.tar')
    test_image, test_landmarks = test_dataset[0]
    print(f"test image shape: {test_image.shape}")
    print(f"test landmarks: {test_landmarks}")



