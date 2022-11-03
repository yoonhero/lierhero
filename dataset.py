import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import pandas as pd
from PIL import Image
import numpy as np
from skimage import io
import cv2

import os

# import face_alignment

import mediapipe as mp


def dataprocessing_get_landmarks(csv_file):
    mp_face_mesh = mp.solutions.face_mesh
    face_detector = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4)


    data = pd.read_csv(csv_file)
    new_row = np.array([], dtype=np.float32)


    for idx, image_path in enumerate(data["image"]):
        input_image = cv2.imread(image_path)
        temp_landmark = face_detector.process(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))

        try:
            np_landmark = np.array([[landmark.x, landmark.y, landmark.z] for landmark in list(temp_landmark.multi_face_landmarks[0].landmark)], dtype=np.float32)
        
        except TypeError:
            print(f"Error When Reading Face Mesh.\nSource: {image_path}")
            np_landmark = np.zeros((478, 3), dtype=torch.float32)

        # Restore =>>> a = [[float(l[0]), float(l[1]), float(l[2])] for l in [landmarks.split("-") for landmarks in landmark.split("|")]]
        landmark = "|".join(["-".join([str(v[0]), str(v[1]), str(v[2])]) for v in np_landmark])

        new_row.append(landmark)

    temp_df = pd.DataFrame(new_row, columns=["landmarks"])

    result = pd.concat([data, temp_df], axis=1)

    result.to_csv("processed_dataset.csv")


class FaceLandmarksDatasetWithMediapipe(Dataset):
    def __init__(self, csv_file):
        self.data_csv = pd.read_csv(csv_file)

        # TODO: Transform Image
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean, std)
        # ])

        mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.4)

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.to_list()

        img_name = self.data_csv["image"].values[idx]  
        image = cv2.imread(img_name)

        temp_landmark = self.face_detector.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        try:
            tensor_landmark = torch.tensor([[landmark.x, landmark.y, landmark.z] for landmark in list(temp_landmark.multi_face_landmarks[0].landmark)], dtype=torch.float32)
        
        except TypeError:
            print(f"Error When Reading Face Mesh.\nSource: {img_name}")
            tensor_landmark = torch.zeros((478, 3), dtype=torch.float32)


        rear_heart_rate = [int(v) for v in self.data_csv["heart_rate"][idx].split("|")]
        heart_rate = torch.tensor(rear_heart_rate, dtype=torch.float32)
        lie = torch.tensor([self.data_csv["lie"].values[idx]], dtype=torch.float32)

        # if self.transform:
        #     sample = self.transform(sample)

        return (tensor_landmark, heart_rate), lie


if __name__ == "__main__":
    dataset = FaceLandmarksDatasetWithMediapipe("data.csv")

    X, lie = dataset[0]
    print(X, lie)

        


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





def dataprocessing_get_landmarks_prev(csv_file):
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


# if __name__ == "__main__":
#     fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, flip_input=False, device="cpu")

#     input_ = io.imread("./test.jpeg")

#     preds = fa.get_landmarks_from_image(input_)

#     print(preds, type(preds))

