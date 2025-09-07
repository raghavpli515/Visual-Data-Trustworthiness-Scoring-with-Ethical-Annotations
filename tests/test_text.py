from src.vdt_scoring.explain.text import textual_reasons

def test_textual_reasons():
    reasons = textual_reasons(edge=0.05, motion=0.7, blur=0.8)
    assert any("blur" in r.lower() for r in reasons)
    assert any("temporal" in r.lower() for r in reasons)
