from datasets import load_dataset
from scipy.io.wavfile import write
import numpy as np


def download_dataset(language: str = "en", n_samples: int = 1000) -> None:
    """
    This function downloads the dataset from the HuggingFace Datasets Hub.
    """
    # Create the loader
    dataset = (
        load_dataset(
            "mozilla-foundation/common_voice_13_0",
            language,
            split="test",
            cache_dir="data/voices/",
            streaming=True,
        )
        .shuffle()
        .filter(
            lambda x: x["age"] != ""
            and x["gender"] != ""
            and x["accent"] != ""
        )
    )

    iterator = iter(dataset)

    for _ in range(n_samples):
        element = next(iterator)
        audio_array_encoded = element["audio"]["array"]

        # Convert to mp3
        samplerate = element["audio"]["sampling_rate"]
        file_name = (
            f"{language}_"
            f'{element["age"]}_'
            f'{element["gender"]}_'
            f'{element["accent"]}_'
            f"{_}"
        )

        try:
            print(f"Writing file {language}, number {_}")
            write(
                f"data/voices/{file_name}.wav",
                samplerate,
                audio_array_encoded.astype(np.float32),
            )
        except Exception as e:
            print(f"Error writing file {language}, number {_}", e)
            continue


if __name__ == "__main__":
    languages = ["ja"]
    for language in languages:
        download_dataset(language=language, n_samples=500)
