from sae_hacking.gemma_cooccurrences import generate_prompts


def test_generate_prompts():
    d = generate_prompts("google/gemma-2-2b", 10, 100, "monology/pile-uncopyrighted", 2)
    next(d.iter(3))
