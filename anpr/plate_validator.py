import re
from collections import defaultdict, Counter
from typing import List, Tuple, Dict, Any


# Common Indian State and Union Territory Codes
STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR",
    "HP", "JH", "KA", "KL", "MP", "MH", "MN", "ML",
    "MZ", "NL", "OD", "PB", "RJ", "SK", "TN", "TS",
    "TR", "UP", "UK", "WB", "DL", "JK", "LA", "CH",
    "LD", "PY", "AN", "DD", "DN", "DH", "BH"
}

# Character confusion maps for OCR normalization
CHAR_TO_DIGIT = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6"
}

DIGIT_TO_CHAR = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "5": "S",
    "8": "B",
    "6": "G"
}


def clean_plate_text(text: str) -> str:
    """
    Clean OCR output:
    - Convert to uppercase
    - Remove spaces and special characters
    - Strip leading HSRP 'IND' badge text if present
    - Strip single-character leading/trailing frame border artifacts when they create invalid length
    - Keep only A-Z and 0-9
    """
    if not text:
        return ""

    text = text.upper()
    text = re.sub(r"[^A-Z0-9]", "", text)

    # 1. Strip High Security Registration Plate (HSRP) 'IND' badge prefix
    if text.startswith("IND") and len(text) > 10:
        text = text[3:]

    # 2. Strip single leading noise character if text is 11 chars and chars 1:3 form a valid state code
    if len(text) == 11 and text[1:3] in STATE_CODES and text[0] not in ("B", "H"):
        text = text[1:]

    # 3. Strip single trailing border noise digit if string ends with 5 consecutive digits (e.g. 83874 -> 8387)
    if len(text) == 10 and text[-5:].isdigit() and text[:2] in STATE_CODES:
        # If dropping the last digit forms a valid Indian plate structure (e.g. DL3CC8387)
        trimmed = text[:-1]
        if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}", trimmed):
            text = trimmed

    return text


def normalize_ocr_plate(cleaned: str) -> str:
    """
    Apply positional heuristics to correct typical OCR character confusions
    without over-normalizing valid arbitrary series letters.
    - State code (first 2 chars): typically letters unless BH series.
    - Last 4 characters: typically digits.
    """
    if not cleaned or len(cleaned) < 7 or len(cleaned) > 10:
        return cleaned

    # Check if BH series: e.g. 22BH1234AA
    if len(cleaned) >= 9 and cleaned[2:4] == "BH" and cleaned[:2].isdigit():
        chars = list(cleaned)
        for i in range(4, 8):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]
        return "".join(chars)

    chars = list(cleaned)

    # 1. State code correction (first 2 characters should be uppercase letters in STATE_CODES)
    if chars[0] + chars[1] not in STATE_CODES:
        for i in (0, 1):
            if chars[i] in DIGIT_TO_CHAR:
                chars[i] = DIGIT_TO_CHAR[chars[i]]

    # 2. Last 4 characters correction (must be digits in standard Indian plates)
    n = len(chars)
    if n >= 8:
        for i in range(n - 4, n):
            if chars[i] in CHAR_TO_DIGIT:
                chars[i] = CHAR_TO_DIGIT[chars[i]]

    return "".join(chars)


def validate_indian_plate(text: str) -> Tuple[bool, str, str]:
    """
    Validate an Indian vehicle registration number.

    Expected formats:
    - Standard 2-digit RTO: SS RR LL NNNN (e.g., HR51BC3493, MH12CD5678, TS09EF4321)
    - 1-digit RTO (Delhi/UTs): SS R LL NNNN (e.g., DL3CC8387, DL3CBA6729, DL1Y1234)
    - Vintage/Commercial: SS RR NNNN or SS R NNNN (e.g., DL011234, DL31234)
    - Bharat Series: YY BH NNNN XX (e.g., 22BH1234AA)

    Returns:
        Tuple: (valid: bool, cleaned_text: str, reason: str)
    """
    raw_cleaned = clean_plate_text(text)

    if not raw_cleaned:
        return False, "", "Empty plate text"

    # Length check (Indian plates are between 7 and 10 characters)
    if len(raw_cleaned) < 7 or len(raw_cleaned) > 10:
        return False, raw_cleaned, "Invalid length"

    cleaned = normalize_ocr_plate(raw_cleaned)

    # 1. Bharat Series format (YY BH NNNN XX)
    if len(cleaned) >= 9 and cleaned[2:4] == "BH":
        if re.fullmatch(r"[0-9]{2}BH[0-9]{4}[A-Z]{1,2}", cleaned):
            return True, cleaned, "Valid Indian plate (BH Series)"
        return False, cleaned, "Invalid BH series format"

    # 2. State Code validation
    state_code = cleaned[:2]
    if state_code not in STATE_CODES:
        return False, cleaned, "Invalid state code"

    # 3. Standard 2-digit or 1-digit RTO with Series and Number (e.g. HR51BC3493, DL3CC8387)
    if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{4}", cleaned):
        return True, cleaned, "Valid Indian plate"

    # 4. Standard 2-digit or 1-digit RTO with 4-digit number (no series letters, e.g. DL011234, DL31234)
    if re.fullmatch(r"[A-Z]{2}[0-9]{1,2}[0-9]{4}", cleaned):
        return True, cleaned, "Valid Indian plate"

    # Diagnostic reason for invalid formats
    if not (cleaned[2:4].isdigit() or cleaned[2:3].isdigit()):
        return False, cleaned, "Invalid RTO code"

    return False, cleaned, "Invalid series/number format"


def vote_plate_consensus(observations: List[Tuple[str, float, bool]]) -> Tuple[str, float, bool]:
    """
    Multi-frame voting mechanism to resolve OCR ambiguities across multiple frames.

    Example:
    AP05AB1234 (conf: 0.90)
    AP05A81234 (conf: 0.70)
    AP05AB1234 (conf: 0.95)
    AP05AB1234 (conf: 0.88)
    resolves to AP05AB1234 with consensus confidence.

    Args:
        observations: List of (cleaned_text, ocr_confidence, is_valid) tuples.

    Returns:
        Tuple: (consensus_plate: str, consensus_confidence: float, is_valid: bool)
    """
    if not observations:
        return "", 0.0, False

    valid_observations = [(clean_plate_text(t), c, v) for t, c, v in observations if t and clean_plate_text(t)]
    if not valid_observations:
        return "", 0.0, False

    if len(valid_observations) == 1:
        text, conf, _ = valid_observations[0]
        is_val, clean_t, _ = validate_indian_plate(text)
        return clean_t, conf, is_val

    # 1. String-level weighted scoring (valid plates get 1.5x weight)
    string_weights = defaultdict(float)
    for text, conf, valid in valid_observations:
        norm_t = normalize_ocr_plate(text)
        weight = max(0.1, conf) * (1.5 if valid else 1.0)
        string_weights[norm_t] += weight

    best_string = max(string_weights.items(), key=lambda x: x[1])[0]

    # 2. Positional character majority voting among dominant string length
    lengths = Counter(len(t) for t, c, v in valid_observations)
    dominant_len = lengths.most_common(1)[0][0]

    filtered_obs = [(normalize_ocr_plate(t), c) for t, c, v in valid_observations if len(normalize_ocr_plate(t)) == dominant_len]

    if filtered_obs:
        consensus_chars = []
        for pos in range(dominant_len):
            char_weights = defaultdict(float)
            for text, conf in filtered_obs:
                char_weights[text[pos]] += max(0.1, conf)
            best_char = max(char_weights.items(), key=lambda x: x[1])[0]
            consensus_chars.append(best_char)

        char_voted_string = "".join(consensus_chars)
    else:
        char_voted_string = best_string

    candidate = char_voted_string if char_voted_string else best_string
    is_valid, final_plate, _ = validate_indian_plate(candidate)

    # Highest observed confidence for consensus plate
    matching_confs = [c for t, c, v in valid_observations if normalize_ocr_plate(t) in (candidate, final_plate)]
    best_conf = max(matching_confs) if matching_confs else max(c for t, c, v in valid_observations)

    return final_plate or candidate, round(best_conf, 4), is_valid


# Test examples
if __name__ == "__main__":
    test_obs = [
        ("AP05AB1234", 0.90, True),
        ("AP05A81234", 0.70, False),
        ("AP05AB1234", 0.95, True),
        ("AP05AB1234", 0.88, True)
    ]

    consensus, conf, valid = vote_plate_consensus(test_obs)
    print(f"Multi-frame voting consensus: {consensus}, Confidence: {conf:.2f}, Valid: {valid}")