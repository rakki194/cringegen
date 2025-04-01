#!/usr/bin/env python3
"""
Test entity detection functionality in the NLP module
"""

import sys
import os
from pprint import pprint

# Add the parent directory to the path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cringegen.prompt_generation.nlp import extract_entities


def test_entity_detection():
    """Test entity detection with various inputs"""
    print("==== ENTITY DETECTION TEST ====\n")

    test_cases = [
        "John Smith visited New York City and met with Google executives in August.",
        "The red fox jumped over the brown dog in the forest near London.",
        "Apple released a new iPhone at their headquarters in California.",
        "Neil Armstrong was the first person to walk on the Moon in July 1969.",
        "The Eiffel Tower in Paris, France is one of the most famous landmarks in Europe.",
        "Leonardo da Vinci painted the Mona Lisa, which is displayed at the Louvre Museum.",
        "Mount Everest, located in the Himalayas, is the tallest mountain on Earth.",
        "Amazon, Microsoft, and Google are some of the largest technology companies.",
        "The character Harry Potter was created by J.K. Rowling in her popular book series.",
        "During the COVID-19 pandemic, the World Health Organization issued health guidelines.",
    ]

    for i, text in enumerate(test_cases):
        print(f"Test case #{i+1}:")
        print(f"Text: {text}")

        entities = extract_entities(text)
        print("Detected entities:")

        if not entities or all(len(entity_list) == 0 for entity_list in entities.values()):
            print("  No entities detected! This might indicate a problem.")
        else:
            for entity_type, entity_list in entities.items():
                if entity_list:
                    print(f"  {entity_type.capitalize()}: {', '.join(entity_list)}")

        print("\nExpected entities (manual annotation):")
        if i == 0:
            print("  People: John Smith")
            print("  Places: New York City")
            print("  Organizations: Google")
            print("  Time: August")
        elif i == 1:
            print("  Animals: fox, dog")
            print("  Places: London, forest")
        # Add more expected results for other test cases

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    test_entity_detection()
