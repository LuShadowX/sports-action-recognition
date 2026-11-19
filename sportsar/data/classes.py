"""Mapping from UCF101 classes to the four sports actions of the assignment.

UCF101 has no clips literally named "serving" or "volleying", so each coarse action is
represented by the UCF101 sports classes whose dominant motion matches it:

* serving   - overarm delivery that starts a point / ball:  TennisSwing, CricketBowling, BaseballPitch
* running   - sprint run-up dominates the clip:              LongJump, PoleVault, HighJump
* smashing  - explosive overhead strike / slam:              VolleyballSpiking, BasketballDunk
* volleying - striking the ball out of the air, rallying:    TableTennisShot, SoccerJuggling

Use ``label_mode: fine`` to train on the 10 underlying UCF101 classes instead.
"""

COARSE_ACTIONS: dict[str, list[str]] = {
    "serving": ["TennisSwing", "CricketBowling", "BaseballPitch"],
    "running": ["LongJump", "PoleVault", "HighJump"],
    "smashing": ["VolleyballSpiking", "BasketballDunk"],
    "volleying": ["TableTennisShot", "SoccerJuggling"],
}

FINE_TO_COARSE: dict[str, str] = {
    fine: coarse for coarse, fines in COARSE_ACTIONS.items() for fine in fines
}

FINE_CLASSES: list[str] = sorted(FINE_TO_COARSE)


def class_names(label_mode: str) -> list[str]:
    if label_mode == "coarse":
        return list(COARSE_ACTIONS)
    if label_mode == "fine":
        return list(FINE_CLASSES)
    raise ValueError(f"label_mode must be 'coarse' or 'fine', got {label_mode!r}")


def label_key(label_mode: str) -> str:
    return "coarse_action" if label_mode == "coarse" else "fine_class"
