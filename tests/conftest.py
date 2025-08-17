import os
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pytest 

@pytest.fixture(autouse=True)
def _isolate_environ(monkeypatch):
    # prevent accidental env leakage (eg- real API keys)
    for k in list(os.environ.keys()):
        if any(s in k.lower() for s in ["key","token",'secret','password','credential']):
            monkeypatch.delenv(k,raising=False)


@pytest.fixture
def temp_workdir(monkeypatch):
    """
    Isolated temp working dir per test for file side-effects.    
    """
    cwd = os.getcwd()
    d = tempfile.mkdtemp(prefix="ut_")
    os.chdir(d)
    try:
        yield Path(d)
    finally:
        os.chdir(cwd)
        shutil.rmtree(d,ignore_errors=True) 

@pytest.fixture
def sample_df():
    """
    Deterministic ,minimal DF with required columns used by your code paths.
    """
    return pd.DataFrame(
        [
            {"main_category":"phones","name":"Pixel 8 Pro","category":"Phones"},
            {"main_category":"laptops","name":"Macbook Air","category":"Laptops"},
            {"main_category":"phones","name":"iPhone 15","category":"Phones"},
            {"main_category":"audio","name":"Sony WH-1000XM5","category":"Audio"}

        ]
    )

@pytest.fixture
def agent_state_cls():
    from agents.nodes.base_agent import AgentState
    return AgentState 