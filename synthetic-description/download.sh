wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/haqkiem-TTS/train-00000-of-00001.parquet -O haqkiem.parquet
unzip -o haqkiem-TTS_audio.zip
rm haqkiem-TTS_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/hindi_ai4bharat_indictts_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/hindi_ai4bharat_indictts/train-00000-of-00001.parquet -O hindi_ai4bharat_indictts.parquet
unzip -o hindi_ai4bharat_indictts_audio.zip
rm hindi_ai4bharat_indictts_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/Emilia-NV_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/Emilia-NV/train-00000-of-00001.parquet -O Emilia-NV.parquet
unzip -o Emilia-NV_audio.zip
rm Emilia-NV_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/nusantara-audiobook_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/malay-audiobook/train-00000-of-00001.parquet -O malay-audiobook.parquet
unzip -o nusantara-audiobook_audio.zip
rm nusantara-audiobook_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/gemini-flash-2.0-speech_data_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/gemini-flash-2.0-speech/train-00000-of-00001.parquet -O gemini-flash-2.0-speech.parquet
unzip -o gemini-flash-2.0-speech_data_audio.zip
rm gemini-flash-2.0-speech_data_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/expresso_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/expresso/train-00000-of-00001.parquet -O expresso.parquet
unzip -o expresso_audio.zip
rm expresso_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/singaporean_accent_district_names_continuation_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/singaporean_accent_district_names_continuation/train-00000-of-00001.parquet -O singaporean_accent_district_names_continuation.parquet
unzip -o singaporean_accent_district_names_continuation_audio.zip
rm singaporean_accent_district_names_continuation_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/DisfluencySpeech_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/DisfluencySpeech/train-00000-of-00001.parquet -O DisfluencySpeech.parquet
unzip -o DisfluencySpeech_audio.zip
rm DisfluencySpeech_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/Latin-Audio_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/Latin-Audio/train-00000-of-00001.parquet -O Latin-Audio.parquet
unzip -o Latin-Audio_audio.zip
rm Latin-Audio_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/genshin-voice_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/genshin-voice/train-00000-of-00001.parquet -O genshin-voice.parquet
unzip -o genshin-voice_audio.zip
rm genshin-voice_audio.zip
python3 sampling.py --file 'genshin-voice.parquet' --max_row 20000

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/japanese-anime-speech-v2_data_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/japanese-anime-speech-v2/train-00000-of-00001.parquet -O japanese-anime-speech-v2.parquet
unzip -o japanese-anime-speech-v2_data_audio.zip
rm japanese-anime-speech-v2_data_audio.zip
python3 sampling.py --file 'japanese-anime-speech-v2.parquet' --max_row 20000

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/indian_accent_english_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/indian_accent_english/train-00000-of-00001.parquet -O indian_accent_english.parquet
unzip -o indian_accent_english_audio.zip
rm indian_accent_english_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/multilingual-tts_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/multilingual-tts/train-00000-of-00001.parquet -O multilingual-tts.parquet
unzip -o multilingual-tts_audio.zip
rm multilingual-tts_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/hungarian-single-speaker-tts_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/hungarian-single-speaker-tts/train-00000-of-00001.parquet -O hungarian-single-speaker-tts.parquet
unzip -o hungarian-single-speaker-tts_audio.zip
rm hungarian-single-speaker-tts_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/OutteTTS-urdu-dataset_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/OutteTTS-urdu-dataset/train-00000-of-00001.parquet -O OutteTTS-urdu-dataset.parquet
unzip -o OutteTTS-urdu-dataset_audio.zip
rm OutteTTS-urdu-dataset_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/maya-audio_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/maya-audio/train-00000-of-00001.parquet -O maya.parquet
unzip -o maya-audio_audio.zip
rm maya-audio_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/AnimeVox_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/AnimeVox/train-00000-of-00001.parquet -O AnimeVox.parquet
unzip -o AnimeVox_audio.zip
rm AnimeVox_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/cml-tts/train-00000-of-00001.parquet -O cml-tts.parquet
python3 sampling_folder.py --file 'cml-tts.parquet' --max_row 20000
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_dutch_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_dutch_audio.zip
rm cml-tts_dutch_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_french_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_french_audio.zip
rm cml-tts_french_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_german_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_german_audio.zip
rm cml-tts_german_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_italian_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_italian_audio.zip
rm cml-tts_italian_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_polish_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_polish_audio.zip
rm cml-tts_polish_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_portuguese_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_portuguese_audio.zip
rm cml-tts_portuguese_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'
HF_HUB_ENABLE_HF_TRANSFER=1 hf download malaysia-ai/Multilingual-TTS cml-tts_spanish_audio.zip --repo-type=dataset --local-dir=./
unzip -o cml-tts_spanish_audio.zip
rm cml-tts_spanish_audio.zip
python3 trim_audio.py --file 'cml-tts.parquet' --kept_file 'cml-tts_kept.json'

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/elevenlabs_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/elevenlabs_ru/train-00000-of-00001.parquet -O elevenlabs_ru.parquet
unzip -o elevenlabs_audio.zip
rm elevenlabs_audio.zip

wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/ru_book_dataset_audio.zip
wget https://huggingface.co/datasets/malaysia-ai/Multilingual-TTS/resolve/main/ru_book_dataset/train-00000-of-00001.parquet -O ru_book_dataset.parquet
unzip -o ru_book_dataset_audio.zip
rm ru_book_dataset_audio.zip