import os
from dataclasses import dataclass

@dataclass
class VideoDataInput:
    data_path: str | os.PathLike
    question: str
    answer: str
    question_id: str = ''

DEFAULT_POINT_TOKEN = '<point>'
