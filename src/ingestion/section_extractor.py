def extract_section_titles(text: str):
    """Simple heuristic: uppercase or colon-ended lines considered section headers."""
    lines = text.split("\n")
    section_map = {}
    current = "Introduction"

    for i, line in enumerate(lines):
        s = line.strip()
        if len(s) > 3 and (s.isupper() or s.endswith(":")):
            current = s.rstrip(":")
        section_map[i] = current

    return section_map
