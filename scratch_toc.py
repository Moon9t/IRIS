with open("docs/BOOK.md", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if line.startswith("#"):
            print(f"{idx+1}: {line.strip()}")
