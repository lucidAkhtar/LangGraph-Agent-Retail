import os
import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
import pytest 

# A fixture is a function decorated with @pytest.fixture that provides setup data or resources to your tests.
# Instead of writing the same setup code in every test, you define it once as a fixture and reuse it.
# the word fixtures comes from traditional testing.It is the fixed state needed to run before the tests.
# They are needed to avoid repetition.To make tests cleaner,modular and reusable.


# Example:

"""
Without fixture:
----------------

def test_connection():
    db = connect_to_db()
    result = db.to_add(2,3)
    assert result == 5 

With fixture:
-------------

import pytest

@pytest.fixture
def db():
    return connect_to_db()

def test_addition(db):
# db which is a pytest fixture is passed to the function and hence this fixture runs earlier and first like constructor....

    result = db.add(2,3)
    assert result == 5

"""
@pytest.fixture(autouse=True)
def _isolate_environ(monkeypatch):
    # prevent accidental env leakage (eg- real API keys)
    for k in list(os.environ.keys()):
        if any(s in k.lower() for s in ["key","token",'secret','password','credential']):
            monkeypatch.delenv(k,raising=False)

"""
- monkeypatch is pytest built-in fixture, comes from pytest itself.
- monkeypatching  = dynamically changing/overriding code at runtime.

Example:

#app.py
def get_user():
    return "real_user"

# test_app.py
import app

def test_fake_user(monkeypatch):
    monkeypatch.setattr(app,"get_user",lambda:"test_user")
    assert app.get_user == "test_user"

- `get_user` is a function, so monkeypatch.setattr replaces it with a new function (lambda: "test_user").
When called, it returns "test_user", matching the assert.

"""
@pytest.fixture
def temp_workdir(monkeypatch):
    """
    Isolated temp working directory per test for file side-effects.    
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