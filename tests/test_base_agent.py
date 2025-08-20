import pandas as pd
import pytest


def test_agent_state_defaults(agent_state_cls):
    """
    - ensures the default initialization of AgentState works as expected.
    - checks user_input is set,while other fields fall back to empty defaults
    like (empty dict,list or dataframe).prevents accidental `None` issues.
    """
    state = agent_state_cls(user_input = "find me a phone")
    assert state.user_input == "find me a phone"
    assert state.preferences == {}
    assert state.retrieved_products == []
    assert isinstance(state.filtered_products,pd.DataFrame) and state.filtered_products.empty
    assert state.compared_insights == []
    assert state.recommendations == []

def test_agent_state_dataframe_allowed(agent_state_cls):
    """
    - verifies that filtered_products can hold a pandas DataFrame and this it is preserved without modification.
    """
    df = pd.DataFrame({"a":[1,2]})
    s = agent_state_cls(user_input = "x",filtered_products = df)
    pd.testing.assert_frame_equal(s.filtered_products,df)

def test_agent_state_dump_roundtrip(agent_state_cls):
    """
    - tests model_dump() preserves all fields,including DataFrame OR checks serialization keeps fields.
    - even though, dataframe is not json-serializable,it must still appear in dump.
    """
    df = pd.DataFrame({"a":[1]})
    s = agent_state_cls(user_input="hi",filtered_products=df)
    d = s.model_dump()
    assert d["user_input"] == "hi"
    assert "filtered_products" in d #non-json serializable is acceptable;field must exist

@pytest.mark.parametrize(
    "field,value",[
        ("preferences",{"brand":"Apple"}),
        ("retrieved_products",[1,2]),
        ("compared_insights",["x"]),
        ("recommendations",["r1"])
    ]
)

def test_agent_state_mutability(agent_state_cls,field,value):
    """
    - confirms fields in AgentState can be updated dynamically.
    - prevents immutability bugs eg - accidentally freezing objects.
    """
    s = agent_state_cls(user_input = "q")
    setattr(s,field,value)
    assert getattr(s,field) == value

