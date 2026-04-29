from research_copilot.agents import AGENT_SEQUENCE, AGENT_SPECS, AgentSpec


def test_agent_sequence_matches_graph_contract():
    assert AGENT_SEQUENCE == (
        "router",
        "planner",
        "tool_executor",
        "rag_tool",
        "answer_synthesizer",
        "critic",
        "finalizer",
    )


def test_agent_specs_document_inputs_and_outputs():
    assert set(AGENT_SPECS) == set(AGENT_SEQUENCE)
    for name in AGENT_SEQUENCE:
        spec = AGENT_SPECS[name]
        assert isinstance(spec, AgentSpec)
        assert spec.name == name
        assert spec.role
        assert spec.inputs
        assert spec.outputs
        assert spec.instructions
