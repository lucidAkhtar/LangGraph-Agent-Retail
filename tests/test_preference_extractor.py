import pytest 
# Goal is to test the extract_preferences() function without actually calling the real LLM/parser/prompt pipeline (because those are slow,external and non-deterministic).
# So instead of real LLM calls, the test fakes the whole pipeline with a dummy class - FakeChain.

### CORE CONCEPT ####

"""
a | b means a.__or__(b)
b | a means a.__ror__(b) if b.__or__ fails 
thats why both are defined, so chaining works in either direction.

"""

class FakeChain:
    
    """
    - The real chain (prompt |llm |parser) is an object pipeline built by operator overloading.Instead of functions, langchain/ollama objects overload pipe operator (|) so you can chain them. 
    
    __init__ -> store the fake result (canned_prefs dict)
    __or__ -> defined behavior of `self | other` , always return self. Ensures you can still write `prompt | llm | parser` and end up with a FakeChain.
    __ror__ -> defines behavior of `other | self`. Same logic,always returns self. Ensures order does not matter in the fake.
    invoke -> when the chain is executed, just return the canned fake result (instead of calling an LLM)
    
    """
    def __init__(self,result=None):
        self._result = result


    def __or__(self,other):
        return self
    
    def __ror__(self,other):
        return self
    
    def invoke(self, _payload):
        return self._result
    

@pytest.fixture
def canned_prefs():
    return {"brand":"Apple",
            "budget":"under 80K",
            "must have features":"great camera"
            }
        
@pytest.fixture
def patch_chain(monkeypatch,canned_prefs):
    
    import agents.nodes.preference_extractor as pe
    # Replace the three pipeline parts with a single fake chain.
    fake = FakeChain(result=canned_prefs)
    monkeypatch.setattr(pe,"prompt",fake)
    monkeypatch.setattr(pe,"llm",fake) # replaces OLLAMA instance
    monkeypatch.setattr(pe,"parser",fake) # replaces JsonOutputParser instance

    return canned_prefs

def test_extract_preferences_assigns_and_returns_same_state(agent_state_cls,patch_chain):
    
    import agents.nodes.preference_extractor as pe
    s = agent_state_cls(user_input="i want an apple phone under 80000 with great camera")
    out = pe.extract_preferences(s)
    assert out is s
    assert s.preferences == patch_chain

def test_extract_preferences_preserves_other_fields(agent_state_cls,patch_chain):
    
    import agents.nodes.preference_extractor as pe
    s = agent_state_cls(user_input="x",retrieved_products=[1,2,3],recommendations=["keep"])
    pe.extract_preferences(s)
    assert s.retrieved_products == [1,2,3]
    assert s.recommendations == ["keep"]