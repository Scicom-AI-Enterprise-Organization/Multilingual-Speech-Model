#!/bin/bash
# Tamil then Chinese Emilia tagging, sequential.
set -o pipefail

# metadata for both (small parquets)
export HF_HUB_ENABLE_HF_TRANSFER=1
hf download Scicom-intl/Malaysian-Tamil-Emilia --repo-type dataset \
  --include "audio_length_ratio_text/*.parquet" --local-dir /root/data/meta-tamil
hf download Scicom-intl/Malaysian-Chinese-Emilia --repo-type dataset \
  --include "audio_length_ratio_text/*.parquet" --local-dir /root/data/meta-chinese

echo "##### TAMIL #####"
bash /root/pipeline/scale_generic.sh \
  Scicom-intl/Malaysian-Tamil-Emilia \
  Scicom-intl/Malaysian-Tamil-Emilia-Nonverbal-Tags \
  "/root/data/meta-tamil/audio_length_ratio_text/*.parquet" \
  audio_processed_trim-0-0.zip audio_processed_trim-1-0.zip audio_processed_trim-2-0.zip \
  audio_processed_trim-3-0.zip audio_processed_trim-4-0.zip audio_processed_trim-5-0.zip \
  audio_processed_trim-6-0.zip

echo "##### CHINESE #####"
bash /root/pipeline/scale_generic.sh \
  Scicom-intl/Malaysian-Chinese-Emilia \
  Scicom-intl/Malaysian-Chinese-Emilia-Nonverbal-Tags \
  "/root/data/meta-chinese/audio_length_ratio_text/*.parquet" \
  malaysian-chinese_processed_trim-0-0.zip malaysian-chinese_processed_trim-0-1.zip \
  malaysian-chinese_processed_trim-1-0.zip malaysian-chinese_processed_trim-1-1.zip \
  malaysian-chinese_processed_trim-2-0.zip malaysian-chinese_processed_trim-2-1.zip \
  malaysian-chinese_processed_trim-3-0.zip malaysian-chinese_processed_trim-3-1.zip \
  malaysian-chinese_processed_trim-4-0.zip malaysian-chinese_processed_trim-4-1.zip \
  malaysian-chinese_processed_trim-5-0.zip malaysian-chinese_processed_trim-5-1.zip \
  malaysian-chinese_processed_trim-6-0.zip malaysian-chinese_processed_trim-6-1.zip \
  malaysian-chinese_processed_trim-7-0.zip malaysian-chinese_processed_trim-7-1.zip \
  malaysian-chinese_processed_trim-8-0.zip malaysian-chinese_processed_trim-8-1.zip \
  malaysian-chinese_processed_trim-9-0.zip malaysian-chinese_processed_trim-9-1.zip \
  malaysian-chinese_processed_trim-10-0.zip

echo "ALL DATASETS DONE"
