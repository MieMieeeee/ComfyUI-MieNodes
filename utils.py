import datetime

LOGO_SUFFIX = "|Mie"
LOGO_EMOJI = "🐑"


def mie_log(message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {LOGO_EMOJI}: {message}")


def add_suffix(source):
    return source + LOGO_SUFFIX


def add_emoji(source):
    return source + " " + LOGO_EMOJI
