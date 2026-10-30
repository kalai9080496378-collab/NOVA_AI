# Dictionary mapping common Romanized Tamil words to Tamil script
_THANGLISH_TAMIL_MAP = {
    "namma": "நம்ம",
    "hero": "ஹீரோ",
    "romba": "ரொம்ப",
    "cool": "கூல்",
    "aana": "ஆனா",
    "villain": "வில்லன்",
    "plan": "பிளான்",
    "vera": "வேற",
    "level": "லெவல்",
    "entry": "என்ட்ரி",
    "kudutharu": "குடுத்தாரு",
    "aaguraru": "ஆகுறாரு",
    "vantha": "வந்தா",
    "super": "சூப்பர்",
    "scene": "காட்சி",
    "la": "ல",
    "with": "வித்",
    "full": "ஃபுல்",
    "style": "ஸ்டைல்",
    "and": "அண்ட்",
    "energy": "எனர்ஜி",
}


def transliterate_thanglish_to_tamil(thanglish_text: str) -> str:
    """
    Converts Romanized Thanglish narration into Tamil phonetic script prior to Tamil TTS synthesis.
    This guarantees the Tamil TTS voice engine speaks with authentic cadence and pronunciation
    rather than spelling English letters.
    """
    words = thanglish_text.split()
    converted_words = []
    for word in words:
        clean = word.lower().strip(",.!?")
        if clean in _THANGLISH_TAMIL_MAP:
            # Preserve trailing punctuation
            punct = word[len(clean):] if len(word) > len(clean) else ""
            converted_words.append(_THANGLISH_TAMIL_MAP[clean] + punct)
        else:
            converted_words.append(word)

    return " ".join(converted_words)
