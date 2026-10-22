from datetime import UTC, datetime, timedelta

from mr_mem.integrations.path_a_lines import LineCard, select_known_lines, stated_times

NOW = datetime(2024, 5, 6, 16, 0, tzinfo=UTC)


def card(q, summary=None, *, settled=False, hours_ago=0, text=None):
    return LineCard(q, summary, settled, NOW - timedelta(hours=hours_ago),
                    text or f"{q} {summary or ''}")


def test_stated_times_reads_the_compiler_format():
    assert stated_times("flight on 2024-05-06 15:00, back 2024-05-09") == (
        datetime(2024, 5, 6, 15, 0, tzinfo=UTC), datetime(2024, 5, 9, tzinfo=UTC))
    assert stated_times("no date here, 2024-13-40 invalid") == ()


def test_recent_lines_come_first_and_old_lines_sleep():
    cards = [card(f"line {i}", hours_ago=i) for i in range(30)]
    sel = select_known_lines(cards, batch_text="", now=NOW, recent=5, cued=5, due=5)
    assert [c.question for c in sel.lines] == [f"line {i}" for i in range(5)]
    assert (sel.recent, sel.cued, sel.due) == (5, 0, 0)


def test_new_content_wakes_a_dormant_line():
    portugal = card("Trip to Portugal next year", "plans to fly to Lisbon via Paris",
                    hours_ago=24 * 300)
    cards = [card(f"busy line {i}", hours_ago=i) for i in range(20)] + [portugal]
    sel = select_known_lines(cards, batch_text="I booked my Lisbon flight for March",
                             now=NOW, recent=10, cued=3, due=0)
    assert portugal in sel.lines
    assert sel.cued >= 1


def test_due_time_bound_plan_is_woken_even_when_quiet():
    flight = card("Catch the afternoon flight", "flight to Tokyo at 2024-05-06 15:00",
                  hours_ago=48)
    settled = card("Decided not to buy the laptop", "decided on 2024-05-01", settled=True,
                   hours_ago=200)
    cards = [card(f"chat {i}", hours_ago=i) for i in range(20)] + [flight, settled]
    sel = select_known_lines(cards, batch_text="unrelated small talk", now=NOW,
                             recent=5, cued=0, due=5)
    assert sel.lines[0] == flight  # due lines lead
    assert settled not in sel.lines  # past-tense lines are not "due"


def test_future_plan_is_not_due_yet():
    trip = card("Trip next year", "leaving 2025-03-01", hours_ago=5)
    sel = select_known_lines([trip], batch_text="", now=NOW, recent=0, cued=0, due=5)
    assert sel.lines == ()


def test_dense_cue_bridges_words_that_do_not_overlap():
    portugal = card("Portugal trip", "go next year", hours_ago=999)
    other = card("Laptop purchase", "comparing models", hours_ago=999)
    vectors = {"Portugal trip go next year": (1.0, 0.0),
               "Laptop purchase comparing models": (0.0, 1.0),
               "booked Lisbon": (0.9, 0.1)}

    def embed(text):
        return vectors[text.strip()]

    sel = select_known_lines([other, portugal], batch_text="booked Lisbon", now=NOW,
                             recent=0, cued=1, due=0, embed=embed)
    assert sel.lines == (portugal,)
