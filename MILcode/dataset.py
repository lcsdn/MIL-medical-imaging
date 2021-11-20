import os.path
from os import listdir
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

def add_age(df):
    """Add an age column to the pandas DataFrame, based on date of birth."""
    new_df = df.copy()
    ref_date = pd.to_datetime(2021, format="%Y")
    new_df["AGE"] = (ref_date - pd.to_datetime(df.DOB)).astype('timedelta64[Y]')
    return new_df

class BloodSmearDataset(Dataset):
    """This dataset includes the preprocessed blood smears of a list of subjects."""
    
    def __init__(self,
            img_dir,
            data_df,
            transform=None,
            age_stats=(68.77, 17.60),
            lymph_count_stats=(26.42, 46.64)
        ):
        """
        Args:
            img_dir: (str) path to the images directory.
            data_df: (DataFrame) list of subjects / sessions used.
            transform: Optional, transformations applied to the images.
            age_stats: (tuple) Optional, mean and std of age for normalisation.
            lymph_count_stats: (tuple) Optional, mean and std of lymphocytes count for normalisation.
        """
        if 'AGE' not in data_df.columns:
            data_df = add_age(data_df)
        self.img_dir = img_dir
        self.data_df = data_df
        self.transform = transform
        self.age_stats = age_stats
        self.lymph_count_stats = lymph_count_stats
        self.gender_code = {'F': 1, 'f': 1, 'M': -1, 'm': -1}
    
    def load_images(self, patient_id):
        patient_dir = os.path.join(self.img_dir, patient_id)
        num_images = len(os.listdir(patient_dir))
        filenames = [os.path.join(patient_dir, '%06d.jpg' % i) for i in range(num_images)]
        images = [Image.open(filename).convert('RGB') for filename in filenames]
        return images
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        """
        Args:
            idx: (int) the index of the subject/session whose data is loaded.
        Returns:
            sample: (dict) corresponding data described by the following keys:
                images: (list) list of blood smear images after applying transform
                label: (int) the diagnosis code (0 for healthy or 1 for lymphocytosis)
                patient_id: (str) ID of the patient (format P...)
                gender: (int) gender of the patient (0 for male or 1 for female)
                age: (int) age of the patient
                lymph_count: (float) absolute number of lymphocytes found in the patient
        """
        label = self.data_df.iloc[idx].LABEL
        patient_id = self.data_df.iloc[idx].ID
        gender_str = self.data_df.iloc[idx].GENDER
        gender = self.gender_code[gender_str]
        age = self.data_df.iloc[idx].AGE
        lymph_count = self.data_df.iloc[idx].LYMPH_COUNT
        images = self.load_images(patient_id)

        # Applying transforms
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        if self.age_stats is not None:
            age = (age - self.age_stats[0]) / self.age_stats[1]
        if self.lymph_count_stats is not None:
            lymph_count = (lymph_count - self.lymph_count_stats[0]) / self.lymph_count_stats[1]

        sample = {'images': images,
                  'label': label,
                  'patient_id': patient_id,
                  'gender': gender,
                  'age': age,
                  'lymph_count': lymph_count}
        return sample

    def train(self):
        """Put all the transforms of the dataset in training mode"""
        self.transform.train()

    def eval(self):
        """Put all the transforms of the dataset in evaluation mode"""
        self.transform.eval()
