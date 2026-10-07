"""Local Socket Mode entry point."""
if __package__:
    from .config import RuntimeConfig, load_local_environment
    from .app import get_app, shutdown_local_services, start_local_services
else:
    # Direct script execution puts src/, not its parent, on the import path.
    # The shared application uses absolute src.* imports, so expose that
    # package root only for the required `python src/local.py` invocation.
    from pathlib import Path
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.config import RuntimeConfig, load_local_environment
    from src.app import get_app, shutdown_local_services, start_local_services

from slack_bolt.adapter.socket_mode import SocketModeHandler


def main():
    load_local_environment()
    settings = RuntimeConfig.from_environment()
    settings.validate(socket_mode=True)
    bolt_app = get_app()
    scheduler = None
    try:
        scheduler = start_local_services()
        SocketModeHandler(bolt_app, settings.app_token).start()
    except KeyboardInterrupt:
        return
    finally:
        shutdown_local_services(scheduler)


if __name__ == "__main__":
    main()
