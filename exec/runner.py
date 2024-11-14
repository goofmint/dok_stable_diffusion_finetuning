import os
import argparse
import boto3
import torch
from diffusers import StableDiffusionPipeline

def download_model_from_s3(s3_endpoint, aws_access_key_id, aws_secret_access_key, bucket_name, model_s3_path, local_model_dir):
    s3 = boto3.client('s3',
				aws_access_key_id=aws_access_key_id,
				aws_secret_access_key=aws_secret_access_key,
        endpoint_url=s3_endpoint
    )

    os.makedirs(local_model_dir, exist_ok=True)

    # S3オブジェクトのリストを取得
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_s3_path)
    for obj in response.get('Contents', []):
        key = obj['Key']
        if key.endswith('/'):
            continue  # ディレクトリはスキップ
        local_path = os.path.join(local_model_dir, os.path.relpath(key, model_s3_path))
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        s3.download_file(bucket_name, key, local_path)
        print(f'Downloaded {key} to {local_path}')

def generate_image_with_model(local_model_dir, prompt, output_image_path):
    """
    追加学習済みモデルを使用して画像を生成する関数。

    Parameters:
    - local_model_dir: ローカルに保存されたモデルのディレクトリ
    - prompt: 画像生成のためのプロンプト
    - output_image_path: 生成された画像の保存パス
    """
    # モデルのロード
    pipe = StableDiffusionPipeline.from_pretrained(local_model_dir)
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

    # 画像生成
    with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
        image = pipe(prompt).images[0]

    # 画像の保存
    image.save(output_image_path)
    print(f'Image saved to {output_image_path}')

def main():
    parser = argparse.ArgumentParser(description="オブジェクトストレージから追加学習済みモデルをダウンロードし、画像を生成するスクリプト")
    parser.add_argument('--s3-token', type=str, required=True, help='Object Storage Access Key ID')
    parser.add_argument('--s3-secret', type=str, required=True, help='Object Storage Secret Access Key')
    parser.add_argument('--s3-bucket', type=str, required=True, help='Object Storage Bucket Name')
    parser.add_argument('--s3-endpoint', type=str, required=True, help='Object Storage Endpoint URL')
    parser.add_argument('--id', type=str, required=True, help='Sakura Dok artifact ID')
    parser.add_argument('--output', type=str, required=True, help='Output folder in Sakura Dok')
    parser.add_argument('--model_path', type=str, required=True, help='Model path on Object Storage')
    parser.add_argument('--prompt', type=str, required=True, help='text prompt for generating images')

    args = parser.parse_args()

    local_model_dir = './downloaded_model'

    # モデルのダウンロード
    download_model_from_s3(
        args.s3_endpoint,
        args.s3_token,
        args.s3_secret,
        args.s3_bucket,
        args.model_path,
        local_model_dir
    )

    # 画像の生成
    generate_image_with_model(local_model_dir, args.prompt, args.output)

if __name__ == "__main__":
    main()
