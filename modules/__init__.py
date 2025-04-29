# modules/__init__.py
import os
import warnings
import logging
from transformers.utils.logging import set_verbosity_error


os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

warnings.filterwarnings("ignore", message=r"The current process just got forked")
set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)

