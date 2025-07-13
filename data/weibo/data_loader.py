
import torch
from torch.utils.data import Dataset
import pickle
import cv2
from PIL import Image
from numpy import asarray
from transformers import ViTFeatureExtractor, BertTokenizer


def get_tokenizer(config):
    """Get BERT Chinese tokenizer"""
    return BertTokenizer.from_pretrained(config.text_tokenizer)


class WeiboDatasetLoader(Dataset):
    def __init__(self, config, dataframe, mode):
        self.config = config
        self.mode = mode
        self.image_filenames = dataframe["image"].values
        self.text = list(dataframe["text"].values)
        self.labels = dataframe["label"].values

        # Setup tokenizer
        tokenizer = get_tokenizer(config)
        self.encoded_text = tokenizer(
            self.text, 
            padding=True, 
            truncation=True, 
            max_length=config.max_length, 
            return_tensors='pt'
        )
        
        # Setup ViT image feature extractor
        self.transforms = ViTFeatureExtractor.from_pretrained(config.image_model_name)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.set_text(idx)
        item.update(self.set_img(idx))
        return item

    def set_text(self, idx):
        item = {
            key: values[idx].clone().detach()
            for key, values in self.encoded_text.items()
        }
        item['text'] = self.text[idx]
        item['label'] = self.labels[idx]
        item['id'] = idx
        return item

    def set_img(self, idx):
        # Load image based on label (1 = rumor, 0 = nonrumor)
        if self.labels[idx] == 1:
            try:
                image = cv2.imread(f"{self.config.rumor_image_path}/{self.image_filenames[idx]}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = Image.open(f"{self.config.rumor_image_path}/{self.image_filenames[idx]}").convert('RGB')
                image = asarray(image)
        else:
            try:
                image = cv2.imread(f"{self.config.nonrumor_image_path}/{self.image_filenames[idx]}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except:
                image = Image.open(f"{self.config.nonrumor_image_path}/{self.image_filenames[idx]}").convert('RGB')
                image = asarray(image)
        
        return self.set_image(image)

    def set_image(self, image):
        """Process image using ViT feature extractor"""
        image = self.transforms(images=image, return_tensors='pt')
        image = image.convert_to_tensors(tensor_type='pt')['pixel_values']
        return {'image': image.reshape((3, 224, 224))}

