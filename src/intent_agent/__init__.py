from intent_agent.config import get_settings


def main() -> None:
    if get_settings().debug:
        print("Hello from intent-agent!")
