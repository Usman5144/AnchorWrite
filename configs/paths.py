import os
from pathlib import Path


PROJECT_ROOT = Path(
    os.environ.get(
        "ANCHORWRITE_ROOT",
        Path(__file__).resolve().parents[1],
    )
).expanduser().resolve()


RAW_DATA_DIR = Path(
    os.environ.get(
        "ANCHORWRITE_DATA_DIR",
        PROJECT_ROOT / "data",
    )
).expanduser()


KENLM_2 = Path(
    os.environ.get(
        "ANCHORWRITE_KENLM_2",
        PROJECT_ROOT / "assets" / "umbc_2gram.binary",
    )
).expanduser()


KENLM_4 = Path(
    os.environ.get(
        "ANCHORWRITE_KENLM_4",
        PROJECT_ROOT / "assets" / "umbc_4gram.trie.binary",
    )
).expanduser()


SEMANTIC_MODEL = Path(
    os.environ.get(
        "ANCHORWRITE_SEMANTIC_MODEL",
        PROJECT_ROOT / "models_aux" / "all-MiniLM-L6-v2",
    )
).expanduser()


OUTPUT_DIR = Path(
    os.environ.get(
        "ANCHORWRITE_OUTPUT_DIR",
        PROJECT_ROOT / "outputs",
    )
).expanduser()
