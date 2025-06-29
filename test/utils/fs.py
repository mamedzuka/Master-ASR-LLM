from pathlib import Path


def find_files(directory: str, extension: str):
    directory_path = Path(directory)
    if not directory_path.is_dir():
        raise ValueError(f"the provided path '{directory}' is not a valid directory.")

    return (path for path in directory_path.rglob("*") if path.is_file() and path.suffix == extension)
