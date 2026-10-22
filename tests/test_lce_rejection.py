"""The one rejection ledger, ported from LCE (design v6 3.3)."""

from datetime import UTC, datetime, timedelta

import pytest

from mr_mem.lce.rejection import RejectionStore

T0 = datetime(2026, 5, 1, tzinfo=UTC)
E = "A and B together suggest E."


@pytest.fixture
def store(tmp_path):
    value = RejectionStore(tmp_path / "rejections.sqlite")
    yield value
    value.close()


def test_denied_inference_is_suppressed_for_the_same_support_only(store):
    rejection = store.reject(region_id="line-1", content=E, source_refs=("A", "B"),
                             authority_ref="user-correction-1", rejected_at=T0,
                             reason="the user denied E")
    assert rejection.active
    # same content, same support (order and whitespace do not matter): suppressed
    assert store.active_match(region_id="line-1", content=" A and B  together suggest E. ",
                              source_refs=("B", "A")) == rejection
    # new evidence C changes the support: E may be evaluated again
    assert store.active_match(region_id="line-1", content=E, source_refs=("A", "B", "C")) is None
    # rejecting again is idempotent
    assert store.reject(region_id="line-1", content=E, source_refs=("A", "B"),
                        authority_ref="again", rejected_at=T0 + timedelta(days=1)) == rejection


def test_reopen_lifts_suppression_and_keeps_the_record(store):
    rejection = store.reject(region_id="line-1", content=E, source_refs=("A", "B"),
                             authority_ref="user", rejected_at=T0)
    reopened = store.reopen(rejection.rejection_id, authority_ref="user-reopened",
                            reopened_at=T0 + timedelta(days=3))
    assert not reopened.active
    assert store.active_match(region_id="line-1", content=E, source_refs=("A", "B")) is None
    assert store.list() == (reopened,)


def test_relevant_records_reach_the_compiler_by_region_or_shared_source(store):
    """Audit 1: a model can rephrase E, so the compiler must see the denial itself."""
    on_line = store.reject(region_id="line-1", content=E, source_refs=("A", "B"),
                           authority_ref="user", rejected_at=T0)
    elsewhere = store.reject(region_id="line-9", content="X implies Y.", source_refs=("X",),
                             authority_ref="user", rejected_at=T0)
    assert store.relevant(region_ids=("line-1",)) == (on_line,)
    assert store.relevant(source_refs=("B", "Q")) == (on_line,)
    assert store.relevant(region_ids=("line-2",), source_refs=("Q",)) == ()
    store.reopen(elsewhere.rejection_id, authority_ref="user", reopened_at=T0)
    assert store.relevant(source_refs=("X",)) == ()


def test_identifiers_and_times_are_deterministic(tmp_path):
    def make(path):
        store = RejectionStore(path)
        item = store.reject(region_id="r", content=E, source_refs=("A",), authority_ref="u",
                            rejected_at=T0)
        store.close()
        return item.rejection_id

    assert make(tmp_path / "one.sqlite") == make(tmp_path / "two.sqlite")
    with pytest.raises(ValueError):
        RejectionStore(tmp_path / "x.sqlite").reject(
            region_id="r", content=E, source_refs=("A",), authority_ref="u",
            rejected_at=datetime(2026, 5, 1))
