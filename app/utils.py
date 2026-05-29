def format_duration(seconds):
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"
