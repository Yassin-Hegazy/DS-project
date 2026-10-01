from src.data import compute_team_seed


def test_assignment_seed_example():
    assert compute_team_seed(["34521", "40218", "41190"]) == 41698
