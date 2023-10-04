from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

sampling_rate = 48000
inputs = processor(
    audio=example["audio"]["array"],
    sampling_rate=sampling_rate,
    return_tensors="pt",
)
