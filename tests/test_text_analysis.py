from pathlib import Path

from doc2query.rewards.lexical import lexical_metrics
from doc2query.text.cache import AnalysisCache
from doc2query.text.normalization import SimplePolishNormalizer


def test_simple_polish_analysis_and_metrics() -> None:
    normalizer = SimplePolishNormalizer()
    query = normalizer.analyze("Ile energii zużywa pompa 5 kW?")
    passage = normalizer.analyze("Pompa o mocy 5 kW zużywa energię elektryczną.")
    metrics = lexical_metrics(query, passage)
    assert "5" in query.numbers
    assert "kw" in query.units
    assert 0 <= metrics.content_jaccard <= 1
    assert metrics.number_preservation == 1


def test_copy_has_higher_copy_density_than_paraphrase() -> None:
    normalizer = SimplePolishNormalizer()
    passage = normalizer.analyze("Warszawa jest stolicą Polski i leży nad Wisłą")
    copied = lexical_metrics(normalizer.analyze("Warszawa jest stolicą Polski"), passage)
    paraphrased = lexical_metrics(
        normalizer.analyze("Jakie miasto pełni funkcję stołeczną?"), passage
    )
    assert copied.copy_density > paraphrased.copy_density


def test_sqlite_cache_roundtrip(tmp_path: Path) -> None:
    normalizer = SimplePolishNormalizer()
    with AnalysisCache(tmp_path / "analysis.sqlite", normalizer) as cache:
        first = cache.analyze("Zażółć gęślą jaźń")
        second = cache.analyze("Zażółć gęślą jaźń")
    assert first == second
