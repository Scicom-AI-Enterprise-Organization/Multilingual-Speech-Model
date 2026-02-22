import torch
import torch.nn.functional as F
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
from funasr import AutoModel
from vox_profile.model.fluency.whisper_fluency import WhisperWrapper as WhisperFluency
from vox_profile.model.voice_quality.whisper_voice_quality import WhisperWrapper as WhisperQuality
from vox_profile.model.accent.whisper_accent import WhisperWrapper as WhisperAccent
import math
from huggingface_hub import PyTorchModelHubMixin
from typing import Optional
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class ModelHead(nn.Module):
    r"""Classification head."""

    def __init__(self, config, num_labels):

        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, num_labels)

    def forward(self, features, **kwargs):

        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class AgeGenderModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.age = ModelHead(config, 1)
        self.gender = ModelHead(config, 3)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits_age = self.age(hidden_states)
        logits_gender = torch.softmax(self.gender(hidden_states), dim=1)

        return hidden_states, logits_age, logits_gender

# https://huggingface.co/spaces/JaesungHuh/voice-gender-classifier/blob/main/model.py
class SEModule(nn.Module):
    def __init__(self, channels : int , bottleneck : int = 128) -> None:
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.ReLU(),
            # nn.BatchNorm1d(bottleneck), # I remove this layer
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
            )

    def forward(self, input : torch.Tensor) -> torch.Tensor:
        x = self.se(input)
        return input * x


class Bottle2neck(nn.Module):
    def __init__(self, inplanes : int, planes : int, kernel_size : Optional[int] = None, dilation : Optional[int] = None, scale : int = 8) -> None:
        super(Bottle2neck, self).__init__()
        width       = int(math.floor(planes / scale))
        self.conv1  = nn.Conv1d(inplanes, width*scale, kernel_size=1)
        self.bn1    = nn.BatchNorm1d(width*scale)
        self.nums   = scale -1
        convs       = []
        bns         = []
        num_pad = math.floor(kernel_size/2)*dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs  = nn.ModuleList(convs)
        self.bns    = nn.ModuleList(bns)
        self.conv3  = nn.Conv1d(width*scale, planes, kernel_size=1)
        self.bn3    = nn.BatchNorm1d(planes)
        self.relu   = nn.ReLU()
        self.width  = width
        self.se     = SEModule(planes)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0:
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(sp)
          sp = self.bns[i](sp)
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]),1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        
        out = self.se(out)
        out += residual
        return out 
    
# https://huggingface.co/spaces/JaesungHuh/voice-gender-classifier/blob/main/model.py
class ECAPA_gender(nn.Module, PyTorchModelHubMixin):
    def __init__(self, C : int = 1024):
        super(ECAPA_gender, self).__init__()
        self.C = C
        self.conv1  = nn.Conv1d(80, C, kernel_size=5, stride=1, padding=2)
        self.relu   = nn.ReLU()
        self.bn1    = nn.BatchNorm1d(C)
        self.layer1 = Bottle2neck(C, C, kernel_size=3, dilation=2, scale=8)
        self.layer2 = Bottle2neck(C, C, kernel_size=3, dilation=3, scale=8)
        self.layer3 = Bottle2neck(C, C, kernel_size=3, dilation=4, scale=8)
        # I fixed the shape of the output from MFA layer, that is close to the setting from ECAPA paper.
        self.layer4 = nn.Conv1d(3*C, 1536, kernel_size=1)
        self.attention = nn.Sequential(
            nn.Conv1d(4608, 256, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Tanh(), # I add this layer
            nn.Conv1d(256, 1536, kernel_size=1),
            nn.Softmax(dim=2),
            )
        self.bn5 = nn.BatchNorm1d(3072)
        self.fc6 = nn.Linear(3072, 192)
        self.bn6 = nn.BatchNorm1d(192)
        self.fc7 = nn.Linear(192, 2)
        self.pred2gender = {0 : 'male', 1 : 'female'}
        self.flipped_filter = torch.FloatTensor([-0.97, 1.]).unsqueeze(0).unsqueeze(0)
        self.mel = None
        
    def create_melspectrogram(self, device):
        self.flipped_filter = self.flipped_filter.to(device)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=512, win_length=400, hop_length=160, \
            f_min = 20, f_max = 7600, window_fn=torch.hamming_window, n_mels=80,
        wkwargs={'device': device}).to(device)

    def logtorchfbank(self, x : torch.Tensor) -> torch.Tensor:
        # Preemphasis
        x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), 'reflect')
        x = F.conv1d(x, self.flipped_filter).squeeze(1)

        # Melspectrogram
        x = self.mel(x) + 1e-6
        
        # Log and normalize
        x = x.log()   
        x = x - torch.mean(x, dim=-1, keepdim=True)
        return x
    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x = self.logtorchfbank(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x+x1)
        x3 = self.layer3(x+x1+x2)

        x = self.layer4(torch.cat((x1,x2,x3),dim=1))
        x = self.relu(x)

        t = x.size()[-1]

        global_x = torch.cat((x,torch.mean(x,dim=2,keepdim=True).repeat(1,1,t), torch.sqrt(torch.var(x,dim=2,keepdim=True).clamp(min=1e-4)).repeat(1,1,t)), dim=1)
        
        w = self.attention(global_x)

        mu = torch.sum(x * w, dim=2)
        sg = torch.sqrt( ( torch.sum((x**2) * w, dim=2) - mu**2 ).clamp(min=1e-4) )

        x = torch.cat((mu,sg),1)
        x = self.bn5(x)
        x = self.fc6(x)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.fc7(x)
        
        return x
    
    def predict(self, audio, device):
        audio = torch.from_numpy(audio)[None].to(device)
        self.eval()
        with torch.no_grad():
            output = self.forward(audio)
            _, pred = output.max(1)
        return self.pred2gender[pred.item()]

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

voice_quality_label_list = [
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

device = torch.device('cuda')

fluency_model = WhisperFluency.from_pretrained("tiantiaf/whisper-large-v3-speech-flow").to(device).eval()
quality_model = WhisperQuality.from_pretrained("tiantiaf/whisper-large-v3-voice-quality").to(device).eval()
accent_model = WhisperAccent.from_pretrained("tiantiaf/whisper-large-v3-narrow-accent").to(device).eval()

emotion = AutoModel(model="iic/emotion2vec_plus_large")
gender_model = ECAPA_gender.from_pretrained('JaesungHuh/ecapa-gender').to(device).eval()
gender_model.create_melspectrogram(device)

model_name = 'audeering/wav2vec2-large-robust-24-ft-age-gender'
processor = Wav2Vec2Processor.from_pretrained(model_name)
agegender_model = AgeGenderModel.from_pretrained(model_name).to(device).eval()

_SEGMENT_SIZE = 3 * 16000

def predict_fluency(audio):
    utterance_fluency_list = list()
    utterance_disfluency_list = list()
    utterance_timestamps = list()

    # Build segments directly on GPU
    audio_tensor = torch.from_numpy(audio).to(device)
    total_samples = audio_tensor.shape[0]
    audio_segment = max(1, (total_samples - _SEGMENT_SIZE) // 16000 + 1)

    segments = []
    lengths = []
    for idx in range(audio_segment):
        seg = audio_tensor[16000 * idx: 16000 * idx + _SEGMENT_SIZE]
        segments.append(seg)
        lengths.append(seg.shape[0])
    input_audio = torch.stack(segments, dim=0)
    input_audio_length = torch.tensor(lengths, device=device)

    with torch.no_grad(), torch.autocast('cuda'):
        fluency_outputs, disfluency_outputs = fluency_model(input_audio, length=input_audio_length)
        fluency_prob = F.softmax(fluency_outputs.float(), dim=1).cpu().numpy().astype(float).tolist()
        disfluency_prob = torch.sigmoid(disfluency_outputs.float())
        disfluency_predictions = (disfluency_prob > 0.7).int().cpu().numpy().tolist()
        disfluency_prob = disfluency_prob.cpu().numpy().astype(float).tolist()

    for audio_idx in range(audio_segment):
        disfluency_type = list()
        utterance_timestamps.append((audio_idx, audio_idx + 3))
        if fluency_prob[audio_idx][0] > 0.5:
            utterance_fluency_list.append("Fluent")
        else:
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
    with torch.no_grad(), torch.autocast('cuda'):
        audio_data = torch.from_numpy(audio)[None].to(device)
        logits = quality_model(audio_data, return_feature=False)
        voice_quality_prob = torch.sigmoid(torch.tensor(logits, dtype=torch.float32))
        threshold = 0.7
        predictions = (voice_quality_prob > threshold).int().cpu().numpy()[0].tolist()
        voice_label = [voice_quality_label_list[i] for i, p in enumerate(predictions) if p == 1]
        return {'quality': voice_label}

def predict_accent(audio):
    with torch.no_grad(), torch.autocast('cuda'):
        audio_data = torch.from_numpy(audio)[None].to(device)
        logits = accent_model(audio_data, return_feature=False)
        return {'accent': english_accent_list[logits.argmax(-1)[0]]}

def predict_sex_age(audio):
    with torch.no_grad(), torch.autocast('cuda'):
        y = processor(y, sampling_rate=16000)
        y = y['input_values'][0]
        y = y.reshape(1, -1)
        y = torch.from_numpy(y).to(device)
        y = agegender_model(y)
        y = torch.hstack([y[1], y[2]])
        y = y.detach().cpu().numpy()
    
    labels = ['female', 'male', 'child']
    label = labels[y[0,1:].argmax(-1)]
    age = int(y[0, 0] * 100)
    return {'sex': label, 'age': age}

def predict_gender(audio):
    with torch.no_grad(), torch.autocast('cuda'):
        return {'gender': gender_model.predict(audio)}

def predict_emotion(audio):
    o = emotion.generate(audio)[0]
    label = o['labels'][np.argmax(o['scores'])]
    return {'emotion': label}