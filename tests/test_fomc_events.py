from datetime import date

import pandas as pd

from text_signals.sources.fomc.events import zlb_flags

# DFEDTARU around the two entries into and exits from the zero lower bound (FRED values).
UPPER = pd.Series(
    {
        date(2008, 12, 16): 0.25,
        date(2015, 12, 15): 0.25,
        date(2015, 12, 16): 0.50,
        date(2020, 3, 3): 1.25,
        date(2020, 3, 14): 1.25,
        date(2020, 3, 16): 0.25,
        date(2022, 3, 16): 0.25,
        date(2022, 3, 17): 0.50,
    }
)


def test_zlb_uses_the_range_in_force_before_the_release() -> None:
    releases = [
        date(2008, 10, 29),  # before the first range: a single target, not the ZLB
        date(2008, 12, 16),  # the move to the ZLB is not itself at the ZLB
        date(2009, 3, 18),  # at the ZLB
        date(2015, 12, 16),  # liftoff, decided while at the ZLB
        date(2016, 1, 27),  # after liftoff
        date(2020, 3, 15),  # the move to the ZLB
        date(2020, 3, 23),  # at the ZLB
        date(2022, 3, 16),  # liftoff
        date(2022, 5, 4),  # after liftoff
    ]
    assert zlb_flags(releases, UPPER) == [False, False, True, True, False, False, True, True, False]
