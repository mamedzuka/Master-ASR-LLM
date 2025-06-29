from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def get_transcriber():
    import torch
    from transformers.pipelines import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # replace on -1 for CPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # replace on torch.float32 for CPU

    return pipeline(
        "automatic-speech-recognition",
        model="jonatasgrosman/wav2vec2-large-xlsr-53-russian",
        device=device,
        torch_dtype=torch_dtype,
        chunk_length_s=10,
        stride_length_s=1,
    )


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует аудиофайл и сохраняет результат в txt файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    transcriber = get_transcriber()

    result = []
    with open(file_path, "rb") as f:
        while chunk := f.read(5 * 1024 * 1024):
            result.append(transcriber(chunk)["text"])  # type: ignore

    output_path = Path(output_dir) / f"{Path(file_path).stem}.txt"
    with open(output_path, "w") as f:
        f.write(" ".join(result))

    return str(output_path)


def read_file_by_chunk():
    with open("erw", "rb") as f:
        f.read()
