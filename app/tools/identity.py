import re


def is_valid_tckn(tckn: str) -> bool:
    """Validate Turkish National ID (T.C. Kimlik Numarası) with checksum.

    Rules:
    - 11 digits
    - First digit cannot be 0
    - 10th digit checksum: ((sum of 1st,3rd,5th,7th,9th)*7 - (sum of 2nd,4th,6th,8th)) % 10
    - 11th digit checksum: sum of first 10 digits % 10
    """
    if not re.fullmatch(r"\d{11}", tckn):
        return False
    if tckn[0] == "0":
        return False

    digits = [int(d) for d in tckn]
    odd_sum = sum(digits[0:9:2])
    even_sum = sum(digits[1:8:2])
    d10 = ((odd_sum * 7) - even_sum) % 10
    if d10 != digits[9]:
        return False
    d11 = (sum(digits[:10])) % 10
    return d11 == digits[10]


__all__ = ["is_valid_tckn"]
