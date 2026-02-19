import torch
import torch.nn.functional as F
import torch.nn as nn
import librosa
import numpy as np
from funasr import AutoModel
from vox_profile.model.fluency.whisper_fluency import WhisperWrapper as WhisperFluency
from vox_profile.model.voice_quality.whisper_voice_quality import WhisperWrapper as WhisperQuality
from vox_profile.model.accent.whisper_accent import WhisperWrapper as WhisperAccent
from vox_profile.model.age_sex.wavlm_demographics import WavLMWrapper as WavLMAge

fluency_label_list = [
    'Fluent', 
    'Disfluent'
]

disfluency_type_labels = [
    "Block", 
    "Prolongation", 
    "Sound Repetition", 
    "Word Repetition", 
    "Interjection"
]

quality_labels = [
    'shrill', 'nasal', 'deep',  # Pitch
    'silky', 'husky', 'raspy', 'guttural', 'vocal-fry', # Texture
    'booming', 'authoritative', 'loud', 'hushed', 'soft', # Volume
    'crisp', 'slurred', 'lisp', 'stammering', # Clarity
    'singsong', 'pitchy', 'flowing', 'monotone', 'staccato', 'punctuated', 'enunciated',  'hesitant', # Rhythm
]

english_accent_list = [
    'East Asia', 'English', 'Germanic', 'Irish', 
    'North America', 'Northern Irish', 'Oceania', 
    'Other', 'Romance', 'Scottish', 'Semitic', 'Slavic', 
    'South African', 'Southeast Asia', 'South Asia', 'Welsh'
]

sex_unique_labels = ["Female", "Male"]

device = torch.device('cuda')

fluency_model = WhisperFluency.from_pretrained("tiantiaf/whisper-large-v3-speech-flow").to(device).eval()
quality_model = WhisperQuality.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").to(device).eval()
accent_model = WhisperAccent.from_pretrained("tiantiaf/whisper-large-v3-narrow-accent").to(device).eval()
wavlm_model = WavLMAge.from_pretrained("tiantiaf/wavlm-large-age-sex").to(device).eval()

emotion = AutoModel(model="iic/emotion2vec_plus_large")

def predict_fluency(audio):
    utterance_fluency_list = list()
    utterance_disfluency_list = list()
    utterance_timestamps = list()

    # The way we do inference for fluency is different as the training data is 3s, so we need to do some shifting
    audio_data = torch.from_numpy(audio)[None]
    audio_segment = (audio_data.shape[1] - 3*16000) // 16000 + 1
    if audio_segment < 1: audio_segment = 1
    input_audio = list()
    input_audio_length = list()
    for idx in range(audio_segment): 
        input_audio.append(audio_data[0, 16000*idx:16000*idx+3*16000])
        input_audio_length.append(torch.tensor(len(audio_data[0, 16000*idx:16000*idx+3*16000])))
    input_audio = torch.stack(input_audio, dim=0)
    input_audio_length = torch.stack(input_audio_length, dim=0)

    with torch.no_grad():
        fluency_outputs, disfluency_outputs = fluency_model(input_audio, length=input_audio_length)
        fluency_prob   = F.softmax(fluency_outputs, dim=1).detach().cpu().numpy().astype(float).tolist()

        disfluency_prob = nn.Sigmoid()(disfluency_outputs)
        # we can set a higher threshold in practice
        disfluency_predictions = (disfluency_prob > 0.7).int().detach().cpu().numpy().tolist()
        disfluency_prob = disfluency_prob.cpu().numpy().astype(float).tolist()
        
    # Now lets gather the predictions for the utterance
    for audio_idx in range(audio_segment):
        disfluency_type = list()
        utterance_timestamps.append((audio_idx, audio_idx + 3))
        if fluency_prob[audio_idx][0] > 0.5:
            utterance_fluency_list.append("Fluent")
        else:
            # If the prediction is disfluent, then which disfluency type
            utterance_fluency_list.append("Disfluent")
            predictions = disfluency_predictions[audio_idx]
            for label_idx in range(len(predictions)):
                if predictions[label_idx] == 1: disfluency_type.append(disfluency_type_labels[label_idx])
        utterance_disfluency_list.append(disfluency_type)

    r = {
        'timestamp': utterance_timestamps,
        'label': utterance_fluency_list,
        'disfluency_label': utterance_disfluency_list,
    }
    return {'fluency': r}

def predict_quality(audio):
    with torch.no_grad():
        audio_data = torch.from_numpy(audio)[None]
        logits = quality_model(audio_data, return_feature=False)
        return {'quality': quality_labels[logits.argmax(-1)[0]]}

def predict_accent(audio):
    with torch.no_grad():
        audio_data = torch.from_numpy(audio)[None]
        logits = accent_model(audio_data, return_feature=False)
        return {'accent': english_accent_list[logits.argmax(-1)[0]]}

def predict_sex_age(audio):
    with torch.no_grad():
        audio_data = torch.from_numpy(audio)[None]
        wavlm_age_outputs, wavlm_sex_outputs = wavlm_model(audio_data.cuda())

        # Age is between 0-100
        age_pred = wavlm_age_outputs.detach().cpu().numpy() * 100
        return {'sex': sex_unique_labels[wavlm_sex_outputs.argmax(-1)[0]], 'age': int(age_pred[0])}

def predict_emotion(audio):
    with torch.no_grad():
        o = emotion.generate(audio)[0]
    label = o['labels'][np.argmax(o['scores'])]
    return {'emotion': label}