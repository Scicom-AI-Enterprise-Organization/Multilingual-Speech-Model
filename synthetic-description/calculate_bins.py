import json
from glob import glob
import os

SPEAKER_RATE_BINS = ["very slowly", "quite slowly", "slightly slowly", "moderate speed", "slightly fast", "quite fast", "very fast"]
SNR_BINS = ["very noisy", "quite noisy", "slightly noisy", "moderate ambient sound", "slightly clear", "quite clear", "very clear"]
REVERBERATION_BINS = ["very roomy sounding", "quite roomy sounding", "slightly roomy sounding", "moderate reverberation", "slightly confined sounding", "quite confined sounding", "very confined sounding"]
UTTERANCE_LEVEL_STD = ["very monotone", "quite monotone", "slightly monotone", "moderate intonation", "slightly expressive", "quite expressive", "very expressive"]
SI_SDR_BINS = ["extremely noisy", "very noisy", "noisy", "slightly noisy", "almost no noise", "very clear"]
PESQ_BINS = ["very bad speech quality", "bad speech quality", "slightly bad speech quality", "moderate speech quality", "great speech quality", "wonderful speech quality"]

# this one is supposed to be apply to speaker-level mean pitch, and relative to gender
SPEAKER_LEVEL_PITCH_BINS = ["very low pitch", "quite low pitch", "slightly low pitch", "moderate pitch", "slightly high pitch", "quite high pitch", "very high pitch"]