"""Manual integration probe; never executed during test collection."""

if __name__ == "__main__":
    from main import process

    print(process("show me my first 3 tasks", "U123", "C123"))
