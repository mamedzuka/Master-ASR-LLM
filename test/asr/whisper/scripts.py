from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def _get_transcriber():
    import torch
    from transformers.pipelines import pipeline

    device = "cuda:0" if torch.cuda.is_available() else "cpu"  # replace on -1 for CPU
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32  # replace on torch.float32 for CPU

    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-large-v3-turbo",
        device=device,
        torch_dtype=torch_dtype,
    )


def transcribe_audio(file_path: str, output_dir: str) -> str:
    """
    Транскрибирует аудиофайл и сохраняет результат в txt файл c именем из file_path.
    Возвращает путь к сохраненному файлу.
    """
    transcriber = _get_transcriber()
    result: str = transcriber(file_path, return_timestamps=True)["text"]  # type: ignore

    output_path = Path(output_dir) / f"{Path(file_path).stem}.txt"
    with open(output_path, "w") as f:
        f.write(result)

    return str(output_path)
