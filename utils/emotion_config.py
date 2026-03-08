
"""
utils/emotion_config.py
Shared emotion label definitions, mappings, and colour palettes.
These constants are used by both backend and frontend layers.
No values have been changed from the original app.py.
"""

# ── Canonical Label Order ────────────────────────────────────────────────────
EMOTION_ORDER = [
    "Happy / Joy",
    "Sad",
    "Angry",
    "Fear",
    "Disgust",
    "Surprise",
    "Neutral",
    "Love",
    "Excited",
    "Frustrated",
]

# ── Emoji + Colour Map ───────────────────────────────────────────────────────
EMOTION_MAP = {
    "Happy / Joy": {"emoji": "😄", "color": "#22c55e"},
    "Sad":         {"emoji": "😢", "color": "#3b82f6"},
    "Angry":       {"emoji": "😠", "color": "#ef4444"},
    "Fear":        {"emoji": "😨", "color": "#8b5cf6"},
    "Disgust":     {"emoji": "🤢", "color": "#065f46"},
    "Surprise":    {"emoji": "😲", "color": "#facc15"},
    "Neutral":     {"emoji": "😐", "color": "#9ca3af"},
    "Love":        {"emoji": "❤️", "color": "#ec4899"},
    "Excited":     {"emoji": "🤩", "color": "#f97316"},
    "Frustrated":  {"emoji": "😤", "color": "#c2410c"},
}

# ── Helper Accessors ─────────────────────────────────────────────────────────
def emotion_label_with_emoji(name: str) -> str:
    ui = EMOTION_MAP.get(name)
    return f"{ui['emoji']} {name}" if ui else name


def emotion_color(name: str) -> str:
    ui = EMOTION_MAP.get(name)
    return ui['color'] if ui else "#888"


def emotion_style_class(name: str) -> str:
    if name in {"Happy / Joy", "Love"}:
        return "emotion-happy"
    if name == "Sad":
        return "emotion-sad"
    if name in {"Angry", "Frustrated"}:
        return "emotion-angry"
    if name == "Fear":
        return "emotion-fear"
    if name == "Excited":
        return "emotion-happy"
    if name == "Surprise":
        return "emotion-surprise"
    if name == "Disgust":
        return "emotion-disgust"
    return "emotion-neutral"
