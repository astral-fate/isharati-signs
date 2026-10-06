"""The RoPE segmenter's label decoder: the package labels frames UNK=0, O=1, B=2, I=3
(sign_language_segmentation.utils.bio.BIO). Until October 2026 the wrapper treated 1 and 2 as signing,
so every O (resting) frame became part of a "sign"."""
import numpy as np

from isharati.pose.segmentation import (B, I, O, UNK, filter_segments, labels_to_segments,
                                        likeliest_probs_to_segments)


def spans(segs):
    return [(s["start"], s["end"]) for s in segs]


def test_label_ids_match_package():
    assert (UNK, O, B, I) == (0, 1, 2, 3)


def test_rest_is_not_a_sign():
    #        0  1  2  3  4  5  6  7  8  9
    preds = [O, O, B, I, I, O, O, B, I, O]
    assert spans(labels_to_segments(preds)) == [(2, 4), (7, 8)]


def test_b_splits_adjacent_signs():
    preds = [B, I, I, B, I, I, O]
    assert spans(labels_to_segments(preds)) == [(0, 2), (3, 5)]


def test_orphan_inside_opens_a_sign_and_unk_closes_it():
    preds = [O, I, I, UNK, I, B]
    assert spans(labels_to_segments(preds)) == [(1, 2), (4, 4), (5, 5)]


def test_all_rest_gives_nothing():
    assert labels_to_segments([O] * 20) == []
    assert labels_to_segments([]) == []


def test_argmax_of_log_probs():
    preds = [O, B, I, I, O, O]
    log_probs = np.full((len(preds), 4), -5.0)
    log_probs[np.arange(len(preds)), preds] = -0.1
    assert spans(likeliest_probs_to_segments(log_probs)) == [(1, 3)]


def test_filter_merges_then_drops_short():
    segs = [{"start": 0, "end": 3}, {"start": 5, "end": 9}, {"start": 20, "end": 21}]
    assert spans(filter_segments(segs, min_len=4, gap=1)) == [(0, 9)]
