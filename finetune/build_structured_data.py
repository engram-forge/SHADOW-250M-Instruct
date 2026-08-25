"""Build deterministic, original structured SFT examples for anti-loop ablations."""
import argparse
import json
import pathlib

CODE = [
    ("remove duplicates while preserving order", "def unique(items):\n    seen = set()\n    result = []\n    for item in items:\n        if item not in seen:\n            seen.add(item)\n            result.append(item)\n    return result", "The function visits each item once. With hashable items, its expected time complexity is O(n) and its space complexity is O(n)."),
    ("count the frequency of each word case-insensitively", "from collections import Counter\n\ndef word_counts(text):\n    words = text.lower().split()\n    return dict(Counter(words))", "Lowercasing makes differently capitalized forms equivalent. Counter performs one pass over the words."),
    ("return the first even number or None", "def first_even(numbers):\n    for number in numbers:\n        if number % 2 == 0:\n            return number\n    return None", "The function stops at the first match and returns None only when no even value exists."),
    ("group strings by their first letter", "def group_by_initial(words):\n    groups = {}\n    for word in words:\n        if word:\n            groups.setdefault(word[0].lower(), []).append(word)\n    return groups", "Empty strings are skipped, and setdefault creates each group once."),
    ("check whether parentheses are balanced", "def balanced(text):\n    depth = 0\n    for char in text:\n        if char == '(':\n            depth += 1\n        elif char == ')':\n            depth -= 1\n            if depth < 0:\n                return False\n    return depth == 0", "A negative depth detects a closing parenthesis without a matching opening one."),
]

TABLES = [
    ("relational and document databases", [("Data model", "Tables and rows", "Documents and fields"), ("Schema", "Usually predefined", "Often flexible"), ("Relationships", "Joins and foreign keys", "Embedding or references")]),
    ("solar and wind power", [("Resource", "Sunlight", "Moving air"), ("Output pattern", "Daylight dependent", "Weather dependent"), ("Typical site", "Roofs or open land", "Open, windy areas")]),
    ("lists and tuples in Python", [("Mutability", "Mutable", "Immutable"), ("Syntax", "Square brackets", "Parentheses"), ("Common use", "Changing collections", "Fixed records")]),
    ("HTTP GET and POST", [("Purpose", "Retrieve a resource", "Submit data"), ("Request body", "Usually absent", "Commonly present"), ("Idempotence", "Expected", "Not guaranteed")]),
]

PROCEDURES = [
    ("a computer that cannot connect to Wi-Fi", ["Confirm Wi-Fi is enabled and airplane mode is off.", "Check whether another device can use the same network.", "Restart the computer and router.", "Forget the network, reconnect, and enter the password again.", "Run the operating system's network troubleshooter.", "Update the wireless adapter driver or contact the network administrator."]),
    ("a printer that is not printing", ["Check power, paper, ink, and visible error lights.", "Confirm the correct printer is selected.", "Inspect the cable or Wi-Fi connection.", "Clear stalled jobs from the print queue.", "Restart the printer and computer.", "Reinstall the driver if the test page still fails."]),
    ("a website returning a server error", ["Record the status code, time, and affected URL.", "Check service health and recent deployments.", "Inspect application and proxy logs.", "Verify database and dependency connectivity.", "Roll back the latest risky change if evidence supports it.", "Confirm recovery and document the cause."]),
    ("preparing for a job interview", ["Read the role description and identify its core skills.", "Research the organization and its products.", "Prepare concise examples of relevant achievements.", "Practice answers aloud without memorizing a script.", "Prepare thoughtful questions for the interviewer.", "Test the route or video setup before the appointment."]),
]

LETTERS = [
    ("community garden", "new volunteer", "Please arrive on time, use shared tools carefully, and ask before harvesting from assigned plots. Wear sturdy shoes and bring water. Your ideas and questions are welcome, and an experienced volunteer will help you during your first shift."),
    ("neighborhood library", "new reading-group member", "Meetings begin at seven on the first Thursday of each month. Read as much as you can, listen respectfully, and feel free to disagree with the book rather than another member. We look forward to hearing your perspective."),
    ("animal shelter", "new weekend helper", "Sign in at reception, follow staff instructions, and never open an enclosure without approval. Comfortable clothing is recommended. Your patience will help the animals feel safe while they wait for homes."),
    ("open-source project", "first-time contributor", "Read the contribution guide, discuss large changes before coding, and include tests with each fix. Small questions are encouraged. A maintainer will review your first pull request and explain any requested changes."),
]

def conversation(user, assistant, kind, recovery=False):
    return {"messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}],
            "metadata": {"kind": kind, "recovery": recovery}}

def recovery_examples(copies=10):
    bfs = """```text
BFS(graph, start):
    queue = a new FIFO queue containing start
    visited = a set containing start
    while queue is not empty:
        node = queue.remove_front()
        process node
        for each neighbor of node:
            if neighbor is not in visited:
                add neighbor to visited
                queue.add_back(neighbor)
```
In an unweighted graph, BFS finds a shortest path by exploring nodes one distance level at a time."""
    recovery = [
        ("Continue this welcome letter without repeating its opening: Welcome to the Community Garden. Welcome to the Community Garden.", "We are delighted that you have joined us. Please use shared tools carefully, ask before harvesting, and contact the coordinator whenever you need guidance. Your time and ideas will help the garden thrive.", "letter"),
        ("Repair and finish this table after the repeated empty rows: | Aspect | SQL | Document |\n|---|---|---|\n| | | |\n| | | |", "| Data model | Tables and rows | Documents and fields |\n| Schema | Usually predefined | Often flexible |\n| Relationships | Joins and foreign keys | Embedding or references |", "table"),
        ("Continue with correct BFS after this stuck prefix: name_address = []\naddress = []\nname_address = []\naddress = []", bfs, "code"),
        ("Replace this repeated Wi-Fi advice with a useful numbered procedure: Restart the router. Restart the router. Restart the router.", "1. Confirm Wi-Fi is enabled.\n2. Test the network with another device.\n3. Restart the computer and router.\n4. Forget and reconnect to the network.\n5. Run the network troubleshooter.\n6. Update the adapter driver or seek support.", "procedure"),
    ]
    return [conversation(user, assistant, kind, True) for user, assistant, kind in recovery for _ in range(copies)]

def examples(include_recovery=False, copies=1, recovery_copies=10):
    rows = []
    for task, code, explanation in CODE:
        answer = f"```python\n{code}\n```\n\n{explanation}"
        for wording in (f"Write a Python function to {task}. Explain it briefly.", f"Provide clear Python code that can {task}."):
            rows.append(conversation(wording, answer, "code"))
    bfs = """```text
BFS(graph, start):
    queue = a new FIFO queue containing start
    visited = a set containing start
    while queue is not empty:
        node = queue.remove_front()
        process node
        for each neighbor of node:
            if neighbor is not in visited:
                add neighbor to visited
                queue.add_back(neighbor)
```
In an unweighted graph, BFS finds a shortest path measured by number of edges because it explores nodes one distance level at a time."""
    rows.extend(conversation(p, bfs, "code") for p in ("Write pseudocode for breadth-first search and state when it finds a shortest path.", "Show a correct BFS algorithm and explain its shortest-path property."))
    for topic, values in TABLES:
        body = "| Aspect | First | Second |\n|---|---|---|\n" + "\n".join(f"| {a} | {b} | {c} |" for a,b,c in values)
        for wording in (f"Make a three-row comparison table for {topic}.", f"Compare {topic} in a concise Markdown table."):
            rows.append(conversation(wording, body, "table"))
    for topic, steps in PROCEDURES:
        body = "\n".join(f"{i}. {step}" for i,step in enumerate(steps,1))
        for wording in (f"Give a numbered troubleshooting procedure for {topic}.", f"Provide six ordered steps for {topic}."):
            rows.append(conversation(wording, body, "procedure"))
    for organization, person, body in LETTERS:
        answer = f"Dear {person.title()},\n\nWelcome to the {organization}. {body}\n\nWe are glad you are joining us and hope your first experience is rewarding.\n\nWarm regards,\nThe {organization.title()} Team"
        for wording in (f"Draft a friendly welcome letter from a {organization} to a {person}.", f"Write a concise, encouraging letter welcoming a {person} to the {organization}."):
            rows.append(conversation(wording, answer, "letter"))
    rows = rows * copies
    if include_recovery: rows.extend(recovery_examples(recovery_copies))
    return rows

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--base", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--include-recovery", action="store_true")
    parser.add_argument("--recovery-only", action="store_true")
    parser.add_argument("--recovery-copies", type=int, default=10)
    parser.add_argument("--copies", type=int, default=1); args = parser.parse_args()
    if args.copies < 1: raise SystemExit("--copies must be positive")
    output = pathlib.Path(args.out); output.parent.mkdir(parents=True, exist_ok=True)
    base = [json.loads(line) for line in open(args.base, encoding="utf-8") if line.strip()]
    additions = recovery_examples(args.recovery_copies) if args.recovery_only else examples(args.include_recovery, args.copies, args.recovery_copies)
    rows = base + additions
    output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(f"wrote {len(rows)} conversations ({len(rows)-len(base)} structured) to {output}")

if __name__ == "__main__": main()
