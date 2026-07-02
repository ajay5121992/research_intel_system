"""
Quick smoke test: exercises all four intent routes end-to-end against the
bootstrapped agent. Run after `pip install -r requirements.txt` and
`python data/generate_sample_data.py` to verify the whole pipeline works.

    python smoke_test.py
"""
from src.bootstrap import bootstrap
from src.agent_graph import run_query

TEST_QUESTIONS = [
    ("What wellness topics have surged in search interest recently?", "TREND"),
    ("What are the key stories in politics this quarter?", "ENTITY_RAG_KG"),
    ("Which topics are under-covered versus market interest?", "GAP"),
    ("Summarize recent signals around business and the economy, and how well our library covers it.", "HYBRID"),
]


def main():
    print("Bootstrapping system...")
    agent, ctx, library_df, external_signals = bootstrap()
    print(f"Ready. {len(library_df)} articles, {len(external_signals)} signal topics.\n")

    for question, expected_intent in TEST_QUESTIONS:
        print("=" * 80)
        print(f"Q: {question}")
        print(f"(expected route family: {expected_intent})")
        result = run_query(agent, question)
        print(f"-> route taken: {result['intent']} (confidence={result.get('intent_confidence', 'n/a')})")
        print(f"-> answer: {result['final_answer']}")
        print(f"-> citations: {result.get('citations', [])}")
        print(f"-> trace: {[t['node'] for t in result['trace']]}")
        print()

    print("Smoke test complete.")


if __name__ == "__main__":
    main()
