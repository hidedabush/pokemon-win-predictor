"""OPTIONAL: how much of the result is luck?

Plan: with poke-env + a local Showdown server, give a slightly smarter bot (pick the
highest-damage move) two fixed teams and play the same matchup ~100 times. The spread of
win rates shows how much randomness alone changes the outcome, i.e. an upper bound on how
accurate any team-only predictor can be.

Needs a local Pokémon Showdown server (see poke-env docs). Skip if short on time.
"""


def run_matchup(team_a: str, team_b: str, n_battles: int = 100) -> float:
    raise NotImplementedError
