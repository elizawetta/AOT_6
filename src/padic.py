from collections import defaultdict
from fractions import Fraction

from .schemas import Claim, ClaimMetadata


def validate_path(path) -> tuple[int, int, int]:
    value = tuple(path)
    if len(value) != 3 or any(type(x) is not int or not 0 <= x < 3 for x in value):
        raise ValueError('A path must contain exactly three integers from 0 to 2')
    return value


def encode(path) -> int:
    a, b, c = validate_path(path)
    return a + 3 * b + 9 * c


def decode(code: int) -> tuple[int, int, int]:
    if type(code) is not int or not 0 <= code <= 26:
        raise ValueError('code must be an integer between 0 and 26')
    return code % 3, (code // 3) % 3, code // 9


def distance(left, right) -> Fraction:
    left, right = validate_path(left), validate_path(right)
    for depth, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return Fraction(1, 3 ** depth)
    return Fraction(0)


def group_claims(claims: list[Claim], metadata: dict[str, ClaimMetadata]):
    groups = defaultdict(list)
    for claim in claims:
        groups[metadata[claim.claim_id].path].append(claim)
    def time_key(claim):
        stamp = metadata[claim.claim_id].occurred_at
        return stamp is None, stamp.isoformat() if stamp else '', claim.claim_id
    return [(path, sorted(groups[path], key=time_key)) for path in sorted(groups)]
