import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


# TODO: Conv Net
class LierDetectModelWithCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_features_ = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=16, kernel_size=6, stride=2, padding=0),

            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=6, stride=2, padding=0),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=6, stride=2, padding=0),
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=6, stride=2, padding=0),
            
        )

        self.dropout = nn.Dropout(0.5)

        self.image_linear = nn.Sequential(
            nn.Linear(128*26, 128),
            nn.ReLU(), 
            nn.Linear(128, 10),
            nn.ReLU(),
        )

        

        self.numeric_features_ = nn.Sequential(
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5, 5),
            nn.ReLU(),
        )

        self.combined_featuers_ = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )


    def forward(self, x1, x2):
        # x1 = x1.reshape(x1.shape[0], 3, -1)

        out1 = self.image_features_(x1)
        out1 = torch.flatten(out1, 1)
        out1 = self.dropout(out1)
        out1 = self.image_linear(out1)
        out2 = self.numeric_features_(x2)

        x = torch.cat((out1, out2), 1)

        out = self.combined_featuers_(x)

        return out




class LierDetectModel_v1(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_features_ = nn.Sequential(
            nn.Linear(478*3, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.Linear(400, 200),
            nn.LeakyReLU(),
            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.Linear(50, 10),
        )

        self.numeric_features_ = nn.Sequential(
            nn.Linear(10,5),
            nn.ReLU(),
            nn.Linear(5, 5),
        )
    
        self.combined_featuers_ = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.image_features_(x1)
        x2 = self.numeric_features_(x2)

        x = torch.cat((x1, x2), 1)
        x = self.combined_featuers_(x)

        return x



# Linear Model with BatchNormalization and Linear
class LierDetectModel_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_features_ = nn.Sequential(
            nn.BatchNorm1d(478*3),
            nn.Linear(478*3, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.3),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.3),
            nn.Linear(400, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),
            nn.Dropout(0.3),
            nn.Linear(400, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
        )

        ## TODO: RNN Numeric features
        self.numeric_features_ = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.Dropout(0.3),
            nn.Linear(5, 5),
        )
    
        self.combined_featuers_ = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.image_features_(x1)
        x2 = self.numeric_features_(x2)

        x = torch.cat((x1, x2), 1)
        x = self.combined_featuers_(x)

        return x




class LierDetectModel_v3(nn.Module):
    def __init__(self):
        super().__init__()

        self.image_features_ = nn.Sequential(
            nn.BatchNorm1d(478*3),
            nn.Linear(478*3, 400),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(400, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),
            nn.Dropout(0.3),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Dropout(0.3),
            nn.Linear(50, 10),
        )

        ## TODO: RNN Numeric features
        self.numeric_features_ = nn.Sequential(
            nn.BatchNorm1d(10),
            nn.Linear(10,5),
            nn.ReLU(),
            nn.BatchNorm1d(5),
            nn.Dropout(0.3),
            nn.Linear(5, 5),
        )
    
        self.combined_featuers_ = nn.Sequential(
            nn.Linear(15, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        x1 = torch.flatten(x1, start_dim=1)
        x1 = self.image_features_(x1)
        x2 = self.numeric_features_(x2)

        x = torch.cat((x1, x2), 1)
        x = self.combined_featuers_(x)

        return x





# # Linear Data Net for Sensor INPUT
# class LinearNet(nn.Module):
#     def __init__(self, input_size):
#         super(LinearNet, self).__init__()

#         self.input_size = input_size

#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(self.input_size, 10),
#             nn.ReLU(),
#             nn.Linear(10, 10),
#             nn.ReLU(),
#             nn.Linear(10, 2),

#         )

#     def forward(self, x):
#         out = self.linear_relu_stack(x)

#         return out

# # Previous Multi Data Net 
# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
        
#         self.image_features_ = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
#             nn.ReLu(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Dropout(),
#             nn.Conv2d(16, 64, kernel_size=5, padding=2),
#             nn.ReLu(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2),
#             nn.Dropout(),
#         )

#         # input 10 arduino sensor values
#         self.numeric_features_ = nn.Sequential(
#             nn.Linear(10, 64),
#             nn.ReLu(inplace=True),
#             nn.Dropout(),
#             nn.Linear(64, 64*64),
#             nn.ReLu(inplace=True),
#             nn.Dropout(),
#         )

#         self.combined_featuers_ = nn.Sequential(
#             nn.Linear(64*64*2, 64*64*2*2),
#             nn.ReLu(inplace=True),
#             nn.Dropout(),
#             nn.Linear(64*64*2*2, 64*64*2),
#             nn.ReLu(inplace=True),
#             nn.Linear(64*3*3*2, 64),
#             nn.Linear(64, 2),
#         )


#     def forward(self, x, y):
#         x = self.image_features_(x)
#         x = x.view(-1, 64*64)
#         y = self.numeric_features_(y)
#         z = torch.cat((x, y), 1)
#         z = self.combined_featuers_(z)
#         return z



