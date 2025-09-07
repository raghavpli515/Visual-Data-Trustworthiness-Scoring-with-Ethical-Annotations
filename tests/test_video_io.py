import os
from src.vdt_scoring.utils.video_io import read_frames

def test_read_frames_missing():
    import pytest
    with pytest.raises(FileNotFoundError):
        list(read_frames("nope.mp4"))
