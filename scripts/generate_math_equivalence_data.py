#!/usr/bin/env python3
"""Generate training data that shows arithmetic is ONE invariant relationship.

The model knows:
- Counting sequences: 1, 2, 3, 4, 5 (100%)
- Letter sequences: A, B, C, D (100%)

The model doesn't connect:
- Symbolic: 3+1=4 (0%)

This dataset shows they're THE SAME:
- "Counting: 1, 2, 3, 4. The next number is 5. So 4+1=5. Four plus one is five."
- Multiple forms of the same invariant relationship
- Explicit equivalence statements

The goal: teach the model that + means "next in sequence"
"""

import json
import random
from pathlib import Path

# Number words
NUM_WORDS = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine",
    10: "ten", 11: "eleven", 12: "twelve", 13: "thirteen",
    14: "fourteen", 15: "fifteen", 16: "sixteen", 17: "seventeen",
    18: "eighteen", 19: "nineteen", 20: "twenty",
}

def num_to_word(n):
    if n in NUM_WORDS:
        return NUM_WORDS[n]
    return str(n)


def generate_succession_equivalences():
    """Generate data showing succession is the same across forms."""
    data = []

    for n in range(1, 20):
        next_n = n + 1

        # Form 1: Counting sequence leads to symbolic
        data.append({
            "instruction": "Complete the pattern and equation.",
            "input": f"Counting: {', '.join(str(i) for i in range(1, n+1))}...",
            "output": f"The next number is {next_n}. This means {n}+1={next_n}."
        })

        # Form 2: Symbolic explained via counting
        data.append({
            "instruction": "Explain this equation using counting.",
            "input": f"What is {n}+1?",
            "output": f"When we count {n}, {next_n}, we see that {n}+1={next_n}. The number after {n} is {next_n}."
        })

        # Form 3: Natural language to symbolic bridge
        data.append({
            "instruction": "Express in multiple ways.",
            "input": f"What comes after {n}?",
            "output": f"{next_n}. In counting: ...{n-1 if n > 1 else ''}, {n}, {next_n}. As addition: {n}+1={next_n}. In words: {num_to_word(n)} plus one equals {num_to_word(next_n)}."
        })

        # Form 4: Explicit equivalence statement
        data.append({
            "instruction": "These are all the same:",
            "input": "",
            "output": f"• Counting: {n}, {next_n}\n• Addition: {n}+1={next_n}\n• Words: {num_to_word(n)} plus one is {num_to_word(next_n)}\n• After {n} comes {next_n}\nThese all express the same relationship."
        })

        # Form 5: Direct equation with sequence context
        data.append({
            "instruction": "Solve using the number sequence.",
            "input": f"{n}+1=",
            "output": f"{next_n}\n\nBecause in the sequence 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20, the number after {n} is {next_n}."
        })

    return data


def generate_addition_equivalences():
    """Generate data showing addition via counting."""
    data = []

    for a in range(1, 11):
        for b in range(1, 11):
            result = a + b
            if result > 20:
                continue

            # Count up from a by b steps
            sequence = [a + i for i in range(b + 1)]
            seq_str = ", ".join(str(x) for x in sequence)

            # Form 1: Addition as counting forward
            data.append({
                "instruction": "Solve by counting forward.",
                "input": f"What is {a}+{b}?",
                "output": f"Start at {a}, count {b} more: {seq_str}. So {a}+{b}={result}."
            })

            # Form 2: Multiple representations
            if a <= 10 and b <= 10 and result <= 20:
                data.append({
                    "instruction": "Express this addition in multiple ways.",
                    "input": f"{a}+{b}=?",
                    "output": f"{a}+{b}={result}\n{num_to_word(a)} plus {num_to_word(b)} equals {num_to_word(result)}\nIf you have {a} and add {b} more, you get {result}."
                })

            # Form 3: Direct answer with brief explanation
            data.append({
                "instruction": "Calculate.",
                "input": f"{a}+{b}=",
                "output": f"{result}"
            })

    return data


def generate_subtraction_equivalences():
    """Generate data showing subtraction via counting backward."""
    data = []

    for a in range(2, 21):
        for b in range(1, min(a, 11)):
            result = a - b

            # Count down from a by b steps
            sequence = [a - i for i in range(b + 1)]
            seq_str = ", ".join(str(x) for x in sequence)

            # Form 1: Subtraction as counting backward
            data.append({
                "instruction": "Solve by counting backward.",
                "input": f"What is {a}-{b}?",
                "output": f"Start at {a}, count back {b}: {seq_str}. So {a}-{b}={result}."
            })

            # Form 2: Direct answer
            data.append({
                "instruction": "Calculate.",
                "input": f"{a}-{b}=",
                "output": f"{result}"
            })

    return data


def generate_multiplication_as_repeated_addition():
    """Show multiplication is repeated addition."""
    data = []

    for a in range(1, 11):
        for b in range(1, 11):
            result = a * b

            # Show as repeated addition
            additions = "+".join([str(a)] * b)

            data.append({
                "instruction": "Multiplication is repeated addition.",
                "input": f"What is {a}×{b}?",
                "output": f"{a}×{b} means {a} added {b} times: {additions}={result}. So {a}×{b}={result}."
            })

            # Direct answer
            data.append({
                "instruction": "Calculate.",
                "input": f"{a}×{b}=",
                "output": f"{result}"
            })

    return data


def generate_invariance_statements():
    """Generate explicit statements about arithmetic invariance."""
    data = []

    # Core invariance principles
    data.append({
        "instruction": "Fundamental principle of arithmetic:",
        "input": "",
        "output": "Addition (+) means 'count forward'. When we write 5+3, we start at 5 and count 3 more: 5, 6, 7, 8. So 5+3=8. This is the same as saying 'five plus three equals eight' or 'if you have five and add three more you get eight'. These are all the same relationship expressed differently."
    })

    data.append({
        "instruction": "The meaning of +1:",
        "input": "",
        "output": "+1 means 'the next number'. In counting 1,2,3,4,5,6,7,8,9,10, each number is +1 from the previous. So 7+1=8 because 8 comes after 7. This is always true: n+1 is the number after n."
    })

    data.append({
        "instruction": "The meaning of -1:",
        "input": "",
        "output": "-1 means 'the previous number'. In counting ...5,6,7,8,9..., going backward each number is -1 from the next. So 7-1=6 because 6 comes before 7. This is always true: n-1 is the number before n."
    })

    # Specific equivalence chains
    for n in range(1, 15):
        data.append({
            "instruction": "These all mean the same thing:",
            "input": f"The number after {n}",
            "output": f"= {n+1}\n= {n}+1\n= {num_to_word(n)} plus one\n= counting: {n}, {n+1}\nAll of these equal {n+1}."
        })

    return data


def generate_qa_pairs():
    """Generate simple Q&A format for fine-tuning."""
    data = []

    # Simple addition facts
    for a in range(1, 13):
        for b in range(1, 13):
            result = a + b
            data.append({
                "instruction": "Answer the math question.",
                "input": f"What is {a}+{b}?",
                "output": f"{result}"
            })
            data.append({
                "instruction": "",
                "input": f"{a}+{b}=",
                "output": f"{result}"
            })

    # Simple subtraction facts
    for a in range(1, 21):
        for b in range(1, min(a+1, 13)):
            result = a - b
            data.append({
                "instruction": "Answer the math question.",
                "input": f"What is {a}-{b}?",
                "output": f"{result}"
            })
            data.append({
                "instruction": "",
                "input": f"{a}-{b}=",
                "output": f"{result}"
            })

    # Simple multiplication facts
    for a in range(1, 13):
        for b in range(1, 13):
            result = a * b
            data.append({
                "instruction": "Answer the math question.",
                "input": f"What is {a}×{b}?",
                "output": f"{result}"
            })

    return data


def main():
    all_data = []

    # Generate each type
    print("Generating succession equivalences...")
    all_data.extend(generate_succession_equivalences())

    print("Generating addition equivalences...")
    all_data.extend(generate_addition_equivalences())

    print("Generating subtraction equivalences...")
    all_data.extend(generate_subtraction_equivalences())

    print("Generating multiplication equivalences...")
    all_data.extend(generate_multiplication_as_repeated_addition())

    print("Generating invariance statements...")
    all_data.extend(generate_invariance_statements())

    print("Generating QA pairs...")
    all_data.extend(generate_qa_pairs())

    # Shuffle
    random.seed(42)
    random.shuffle(all_data)

    # Save
    output_dir = Path("data/training")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "math_equivalence_training.json"
    with open(output_path, "w") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nGenerated {len(all_data)} training examples")
    print(f"Saved to {output_path}")

    # Also save in JSONL format for easier training
    jsonl_path = output_dir / "math_equivalence_training.jsonl"
    with open(jsonl_path, "w") as f:
        for item in all_data:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {jsonl_path}")

    # Print some examples
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLES:")
    print("="*60)
    for i, example in enumerate(all_data[:10]):
        print(f"\n--- Example {i+1} ---")
        if example["instruction"]:
            print(f"Instruction: {example['instruction']}")
        if example["input"]:
            print(f"Input: {example['input']}")
        print(f"Output: {example['output'][:200]}...")


if __name__ == "__main__":
    main()
