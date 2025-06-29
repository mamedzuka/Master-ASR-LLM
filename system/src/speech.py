from pathlib import Path
from typing import Dict
import os

import torch                             
from transformers import pipeline       
from symspellpy import SymSpell, Verbosity


def transcribe_mp3(path: str | Path) -> Dict[str, str]:
    model_name = os.getenv("WHISPER_MODEL_NAME", "openai/whisper-large-v3-turbo")
    num_beams = int(os.getenv("WHISPER_BEAM_SIZE", 5))

    device = 0 if torch.cuda.is_available() else -1
    pipe = pipeline(
        task="automatic-speech-recognition",
        model=model_name,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
        device=device,
    )
    result = pipe(
        str(path),
        generate_kwargs={"num_beams": num_beams, "language": "ru"},
    )
    return {"text": result["text"]}


def _load_symspell(dict_path: str | Path = "ru-100k.txt") -> SymSpell:
    sym = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    sym.load_frequency_dictionary(str(dict_path), term_index=0, count_index=1, separator=" ")
    return sym


def correct_text(text: str, dict_path: str | Path | None = None) -> str:
    sym = _load_symspell(dict_path or os.getenv("SYM_DICT_PATH", "ru-100k.txt"))
    out = []
    for word in text.split():
        sug = sym.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
        out.append(sug[0].term if sug else word)
    return " ".join(out)

