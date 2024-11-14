#!/bin/bash

. ../import-env ../.env

set -ue
shopt -s nullglob

export TZ=${TZ:-Asia/Tokyo}

# アウトプット先ディレクトリ（自動付与） /opt/artifact固定です
if [ -z "${SAKURA_ARTIFACT_DIR:-}" ]; then
  echo "Environment variable SAKURA_ARTIFACT_DIR is not set" >&2
  exit 1
fi

# DOKのタスクID（自動付与）
if [ -z "${SAKURA_TASK_ID:-}" ]; then
  echo "Environment variable SAKURA_TASK_ID is not set" >&2
  exit 1
fi

if [ -z "${S3_BUCKET:-}" ]; then
  echo "Environment variable S3_BUCKET is not set" >&2
  exit 1
fi

if [ -z "${S3_ENDPOINT:-}" ]; then
  echo "Environment variable S3_ENDPOINT is not set" >&2
  exit 1
fi

if [ -z "${S3_SECRET:-}" ]; then
  echo "Environment variable S3_SECRET is not set" >&2
  exit 1
fi

if [ -z "${S3_TOKEN:-}" ]; then
  echo "Environment variable S3_TOKEN is not set" >&2
  exit 1
fi

if [ -z "${MODEL_PATH:-}" ]; then
	MODEL_PATH="models/"
fi

if [ -z "${PROMPT:-}" ]; then
  PROMPT="What is the meaning of life?"
fi

# S3_はすべてboto3用の環境変数です
  python3 runner.py \
		--id="${SAKURA_TASK_ID}" \
		--output="${SAKURA_ARTIFACT_DIR}" \
		--s3-bucket="${S3_BUCKET:-}" \
		--s3-endpoint="${S3_ENDPOINT:-}" \
		--s3-secret="${S3_SECRET:-}" \
		--s3-token="${S3_TOKEN:-}" \
    --prompt="${PROMPT}" \
    --model_path="${MODEL_PATH}"
