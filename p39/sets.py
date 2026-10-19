"""Shared definitions of the evaluation samples."""
# WSE: an overpass enters the main comparison if every reach-level estimator and the
# published node height at the gauge node exist. The raw node height is a reference
# that exists only when the gauge node itself was observed, so it is evaluated on its
# own subset and does not define the sample.
WSE_REQUIRED = ["official", "official_s2", "official_sm", "mean", "wmean", "median",
                "ols", "wls", "theilsen", "huber"]
