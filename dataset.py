import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import numpy as np
from skimage import io

import os

import face_alignment



class WithFaceLandmarksDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_csv = pd.read_csv(csv_file)
        self.transform = transform

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_name = os.path.join(self.root_dir,
                                self.data_csv.iloc[idx, 0])

        image = Image.open(img_name)
        landmarks = self.fa.get_landmarks(image)
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype('float')

        heart_rate = self.data_csv.iloc[idx, 1]
        lie = self.data_csv.iloc[idx, 2]

        if self.transform:
            sample = self.transform(sample)

        return landmarks, heart_rate, lie





def dataprocessing_get_landmarks(csv_file):
    data = pd.read_csv(csv_file)

    new_row = np.array([], dtype=np.float32)

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cuda")

    for idx, image in enumerate(data["img"]):
        input_ = io.imread(image)
        preds = fa.get_landmarks_from_image(input_)

        new_row.append(preds)

    temp_df = pd.DataFrame(new_row, columns=["landmarks"])

    result = pd.concat([data, temp_df], axis=1)

    result.to_csv("processed_dataset.csv")
