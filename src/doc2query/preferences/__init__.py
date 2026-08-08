"""Candidate selection and preference-data contracts."""

from doc2query.preferences.build import build_preference_dataset, select_candidate_sets
from doc2query.preferences.schemas import CandidateScore, ScoredCandidate

__all__ = [
    "CandidateScore",
    "ScoredCandidate",
    "build_preference_dataset",
    "select_candidate_sets",
]
