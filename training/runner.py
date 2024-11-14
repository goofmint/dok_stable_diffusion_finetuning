import os
import argparse
import boto3
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Stable Diffusion model with images from S3.")
    parser.add_argument('--s3-token', type=str, required=True, help='Object Storage Access Key ID')
    parser.add_argument('--s3-secret', type=str, required=True, help='Object Storage Secret Access Key')
    parser.add_argument('--s3-bucket', type=str, required=True, help='Object Storage Bucket Name')
    parser.add_argument('--s3-endpoint', type=str, required=True, help='Object Storage Endpoint URL')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder in Object Storage bucket')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder in Object Storage bucket')
    parser.add_argument('--id', type=str, required=True, help='Sakura Dok artifact ID')
    parser.add_argument('--output', type=str, required=True, help='Output folder in Sakura Dok')
    return parser.parse_args()

def download_images_from_s3(s3, bucket_name, input_folder, local_input_dir):
    os.makedirs(local_input_dir, exist_ok=True)
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=input_folder)
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('.jpg') or key.endswith('.png'):
            local_path = os.path.join(local_input_dir, os.path.basename(key))
            s3.download_file(bucket_name, key, local_path)
            print(f'Downloaded {key} to {local_path}')

class CustomDataset(Dataset):
    def __init__(self, image_dir, tokenizer, max_length=77, image_size=(256, 256)):
        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.image_size = image_size
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]
        self.captions = [self._extract_caption(p) for p in self.image_paths]
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),  # 画像のリサイズ
            transforms.ToTensor()
        ])

    def _extract_caption(self, image_path):
        return os.path.splitext(os.path.basename(image_path))[0]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        caption = self.captions[idx]
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)  # 画像のリサイズとテンソル変換
        inputs = self.tokenizer(caption, padding='max_length', max_length=self.max_length, return_tensors='pt')
        return image, inputs.input_ids.squeeze(), inputs.attention_mask.squeeze()

def fine_tune_model(local_input_dir, local_output_dir):
    pretrained_model_name_or_path = "CompVis/stable-diffusion-v1-4"
    num_train_epochs = 3
    batch_size = 4
    learning_rate = 5e-6

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    pipeline = StableDiffusionPipeline.from_pretrained(pretrained_model_name_or_path)

    dataset = CustomDataset(local_input_dir, tokenizer)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(pipeline.unet.parameters(), lr=learning_rate)

    pipeline.unet.train()
    for epoch in range(num_train_epochs):
        for images, input_ids, attention_mask in dataloader:
            text_embeddings = text_encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            # ノイズの追加と予測、損失の計算と逆伝播の処理を追加
            optimizer.step()
            optimizer.zero_grad()

    pipeline.save_pretrained(local_output_dir)

def upload_model_to_s3(s3, bucket_name, output_folder, local_output_dir):
    for root, dirs, files in os.walk(local_output_dir):
        for file in files:
            local_path = os.path.join(root, file)
            relative_path = os.path.relpath(local_path, local_output_dir)
            s3_path = os.path.join(output_folder, relative_path)
            s3.upload_file(local_path, bucket_name, s3_path)
            print(f'Uploaded {local_path} to s3://{bucket_name}/{s3_path}')

def main():
    args = parse_args()

    s3 = boto3.client('s3',
        aws_access_key_id=args.s3_token,
        aws_secret_access_key=args.s3_secret,
        endpoint_url=args.s3_endpoint
    )

    local_input_dir = './train_images'
    local_output_dir = './fine_tuned_model'

    download_images_from_s3(s3, args.s3_bucket, args.input_folder, local_input_dir)
    fine_tune_model(local_input_dir, local_output_dir)
    upload_model_to_s3(s3, args.s3_bucket, args.output_folder, local_output_dir)

if __name__ == "__main__":
    main()
