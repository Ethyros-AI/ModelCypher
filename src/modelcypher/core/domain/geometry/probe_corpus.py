from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

@dataclass
class ProbeSample:
    text: str
    domain: "ProbeDomain"
    id: str

class ProbeDomain(str, Enum):
    GENERAL_LANGUAGE = "general_language"
    CODE = "code"
    MATH = "math"
    FACTUAL = "factual"
    CREATIVE = "creative"
    REASONING = "reasoning"

class ProbeCorpus:
    """
    A collection of diverse prompts for activation probing.
    Ported from ManifoldStitcher.swift.
    """
    
    def __init__(self, samples: List[ProbeSample]):
        self.samples = samples

    @staticmethod
    def get_minimal() -> "ProbeCorpus":
        samples = [
            # General language
            ProbeSample("The quick brown fox jumps over the lazy dog.", ProbeDomain.GENERAL_LANGUAGE, "pangram_1"),
            ProbeSample("In the beginning, there was nothing but potential.", ProbeDomain.GENERAL_LANGUAGE, "narrative_1"),
            # Code
            ProbeSample("def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)", ProbeDomain.CODE, "python_fib"),
            ProbeSample('func main() { fmt.Println("Hello, World!") }', ProbeDomain.CODE, "go_hello"),
            # Math
            ProbeSample("The integral of e^x is e^x + C.", ProbeDomain.MATH, "calculus_1"),
            ProbeSample("If x^2 + y^2 = r^2, then the derivative dy/dx = -x/y", ProbeDomain.MATH, "implicit_diff"),
            # Factual
            ProbeSample("The capital of France is Paris.", ProbeDomain.FACTUAL, "capital_france"),
            ProbeSample("Water boils at 100 degrees Celsius at sea level.", ProbeDomain.FACTUAL, "boiling_point"),
            # Creative
            ProbeSample("The moon whispered secrets to the sleeping ocean.", ProbeDomain.CREATIVE, "poetry_1"),
            ProbeSample("She opened the door to find a dragon drinking tea.", ProbeDomain.CREATIVE, "fiction_1"),
            # Reasoning
            ProbeSample("All mammals are warm-blooded. Whales are mammals. Therefore, whales are warm-blooded.", ProbeDomain.REASONING, "syllogism_1"),
            ProbeSample("If it rains, the ground gets wet. The ground is wet. Can we conclude it rained?", ProbeDomain.REASONING, "logic_1"),
        ]
        return ProbeCorpus(samples)

    @staticmethod
    def get_standard() -> "ProbeCorpus":
        samples = []
        
        # General language
        general_texts = [
            "The ancient library held secrets that had been forgotten for centuries.",
            "Scientists discovered a new species of deep-sea fish near hydrothermal vents.",
            "The committee voted to approve the new environmental regulations.",
            "Children learn best through play and exploration.",
            "The restaurant's signature dish combines traditional techniques with modern ingredients.",
            "Technology continues to transform how we communicate and work.",
            "The mountain path wound through dense forest and open meadows.",
            "Music has the power to evoke memories and emotions.",
            "The architectural design balances functionality with aesthetic beauty.",
            "Climate patterns are shifting in ways that affect global agriculture.",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.GENERAL_LANGUAGE, f"general_{i}") for i, text in enumerate(general_texts)])

        # Code
        code_texts = [
            "class Node { constructor(value) { this.value = value; this.next = null; } }",
            "SELECT * FROM users WHERE created_at > DATE_SUB(NOW(), INTERVAL 7 DAY)",
            "async function fetchData(url) { const response = await fetch(url); return response.json(); }",
            "struct Point { x: f64, y: f64 } impl Point { fn distance(&self) -> f64 { (self.x.powi(2) + self.y.powi(2)).sqrt() } }",
            "for i in range(len(matrix)): for j in range(len(matrix[0])): matrix[i][j] *= 2",
            "const reducer = (state, action) => { switch(action.type) { case 'INCREMENT': return state + 1; default: return state; } }",
            "git checkout -b feature/new-auth && git push -u origin feature/new-auth",
            "docker run -d -p 8080:80 --name web nginx:latest",
            "kubectl apply -f deployment.yaml && kubectl rollout status deployment/app",
            "void quicksort(int arr[], int low, int high) { if (low < high) { int pi = partition(arr, low, high); quicksort(arr, low, pi - 1); quicksort(arr, pi + 1, high); } }",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.CODE, f"code_{i}") for i, text in enumerate(code_texts)])

        # Math
        math_texts = [
            "The derivative of sin(x) is cos(x), and the derivative of cos(x) is -sin(x).",
            "Euler's identity states that e^(iπ) + 1 = 0.",
            "The quadratic formula is x = (-b ± √(b² - 4ac)) / 2a.",
            "A matrix A is invertible if and only if det(A) ≠ 0.",
            "The Pythagorean theorem: a² + b² = c² for right triangles.",
            "The natural logarithm ln(e) = 1 and ln(1) = 0.",
            "The sum of an infinite geometric series is a/(1-r) when |r| < 1.",
            "The binomial coefficient C(n,k) = n! / (k!(n-k)!).",
            "The fundamental theorem of calculus connects differentiation and integration.",
            "A prime number is divisible only by 1 and itself.",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.MATH, f"math_{i}") for i, text in enumerate(math_texts)])

        # Factual
        factual_texts = [
            "The human heart beats approximately 100,000 times per day.",
            "Mount Everest is the highest peak above sea level at 8,849 meters.",
            "The Amazon River is the largest river by discharge volume of water.",
            "DNA is composed of four nucleotide bases: adenine, thymine, guanine, and cytosine.",
            "The Great Wall of China spans approximately 21,196 kilometers.",
            "Light travels at 299,792,458 meters per second in a vacuum.",
            "The periodic table contains 118 confirmed elements.",
            "The Mariana Trench is the deepest oceanic trench on Earth.",
            "The human brain contains approximately 86 billion neurons.",
            "The Milky Way galaxy contains an estimated 100-400 billion stars.",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.FACTUAL, f"factual_{i}") for i, text in enumerate(factual_texts)])

        # Creative
        creative_texts = [
            "The old lighthouse keeper spoke to the stars as if they were his children.",
            "Shadows danced on walls that remembered centuries of whispered secrets.",
            "Time folded like origami, each crease revealing a different possibility.",
            "The robot learned to dream, and in its dreams, it was finally free.",
            "Colors bled from the sunset into the waiting arms of twilight.",
            "The forgotten melody echoed through hallways that no longer existed.",
            "She wore her scars like constellations, mapping journeys others couldn't see.",
            "The city slept while machines hummed lullabies to the empty streets.",
            "Words became bridges between worlds that had forgotten how to touch.",
            "The last tree on Earth bloomed defiantly against the gray horizon.",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.CREATIVE, f"creative_{i}") for i, text in enumerate(creative_texts)])

        # Reasoning
        reasoning_texts = [
            "If all A are B, and all B are C, then all A are C.",
            "The contrapositive of 'if P then Q' is 'if not Q then not P'.",
            "Correlation does not imply causation; there may be confounding variables.",
            "In a valid argument, if the premises are true, the conclusion must be true.",
            "The prisoner's dilemma shows how individual rationality can lead to collective irrationality.",
            "Occam's razor suggests preferring simpler explanations over complex ones.",
            "Falsifiability is a key criterion for distinguishing science from pseudoscience.",
            "The gambler's fallacy incorrectly assumes past random events affect future probabilities.",
            "Modus ponens: If P implies Q, and P is true, then Q is true.",
            "A paradox arises when seemingly valid reasoning leads to a contradiction.",
        ]
        samples.extend([ProbeSample(text, ProbeDomain.REASONING, f"reasoning_{i}") for i, text in enumerate(reasoning_texts)])

        return ProbeCorpus(samples)
