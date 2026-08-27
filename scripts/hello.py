# scripts/hello.py
# A quick sanity check: confirms Python, VS Code, and our venv are all wired up correctly.

def greet(name: str) -> str:
    """Return a greeting for the given name."""
    return f"Hello, {name}! Your environment is working."


if __name__ == "__main__":
    message = greet("engineer")
    print(message)