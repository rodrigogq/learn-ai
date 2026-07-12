"""Chapter 1 - the same problem solved twice: hand-written rules vs learning.

Classifies fruit (apple or orange) from two features: weight in grams and
surface smoothness (0 = very rough, 1 = very smooth). Version 1 uses rules a
programmer guessed. Version 2 LEARNS its numbers from labeled examples using
a nearest-centroid classifier, then both are judged on fruits neither has seen.

Run from the repository root:
    .venv/bin/python chapters/01-what-is-ai/python/rules_vs_learning.py
"""

import math

# Each training example is (weight_in_grams, surface_smoothness, true_label).
# Deliberately tiny so you can check every computation by hand.
TRAINING_FRUITS = [
    (120.0, 0.80, "apple"),
    (130.0, 0.90, "apple"),
    (140.0, 0.85, "apple"),
    (130.0, 0.85, "apple"),
    (160.0, 0.40, "orange"),
    (170.0, 0.45, "orange"),
    (180.0, 0.40, "orange"),
    (190.0, 0.43, "orange"),
]

# Fruits neither program sees during training - this is the honest test.
# The 160 g smooth apple exists to break the hand-written weight rule.
TEST_FRUITS = [
    (118.0, 0.90, "apple"),
    (160.0, 0.80, "apple"),
    (135.0, 0.88, "apple"),
    (155.0, 0.45, "orange"),
    (185.0, 0.38, "orange"),
    (165.0, 0.50, "orange"),
]


def classify_with_hand_written_rules(weight_in_grams, surface_smoothness):
    """Classify a fruit using rules a programmer guessed by eye.

    Arguments:
        weight_in_grams: the fruit's weight.
        surface_smoothness: 0 (very rough) to 1 (very smooth).

    Returns "apple" or "orange".
    """
    # The threshold 150 is the programmer's guess, and the rule ignores
    # smoothness entirely - the whole point of this chapter is that nobody
    # checked this rule against data. The unused argument stays in the
    # signature so both classifiers are called the same way.
    del surface_smoothness
    if weight_in_grams > 150.0:
        return "orange"
    return "apple"


def learn_class_averages(training_fruits):
    """THE LEARNING STEP: compute the average fruit of each class.

    Arguments:
        training_fruits: list of (weight_in_grams, surface_smoothness, label).

    Returns a dict mapping each label to its centroid, e.g.
    {"apple": (130.0, 0.85), "orange": (175.0, 0.42)}. These two points are
    the model's PARAMETERS - numbers that came from data, not from a person.
    """
    class_averages = {}
    for label in ("apple", "orange"):
        fruits_of_this_class = [fruit for fruit in training_fruits if fruit[2] == label]
        average_weight = sum(fruit[0] for fruit in fruits_of_this_class) / len(fruits_of_this_class)
        average_smoothness = sum(fruit[1] for fruit in fruits_of_this_class) / len(fruits_of_this_class)
        class_averages[label] = (average_weight, average_smoothness)
    return class_averages


def classify_with_learned_model(weight_in_grams, surface_smoothness, class_averages):
    """Classify a fruit as the class whose average fruit is nearest to it.

    Arguments:
        weight_in_grams: the fruit's weight.
        surface_smoothness: 0 (very rough) to 1 (very smooth).
        class_averages: the parameters produced by learn_class_averages().

    Returns "apple" or "orange".
    """
    # Smoothness lives in [0, 1] while weight lives near 150, so raw distance
    # would be decided by weight alone. Dividing weight by 100 keeps both
    # features on comparable scales (Chapter 12 treats this properly as
    # "normalization").
    def distance_to_average(average_point):
        weight_difference = (weight_in_grams - average_point[0]) / 100.0
        smoothness_difference = surface_smoothness - average_point[1]
        return math.sqrt(weight_difference ** 2 + smoothness_difference ** 2)

    best_label = None
    best_distance = float("inf")
    for label, average_point in class_averages.items():
        if distance_to_average(average_point) < best_distance:
            best_distance = distance_to_average(average_point)
            best_label = label
    return best_label


def main():
    print(f"Training data: {len(TRAINING_FRUITS)} fruits with known labels.")
    print()

    class_averages = learn_class_averages(TRAINING_FRUITS)
    print("LEARNING STEP (computing the parameters from data):")
    for label, (average_weight, average_smoothness) in class_averages.items():
        print(f"  average {label:<6} -> weight {average_weight:.1f} g, smoothness {average_smoothness:.2f}")
    print()

    print(f"Judging {len(TEST_FRUITS)} new fruits both programs have never seen:")
    print()
    print("  fruit (g, smooth)   true      rules say   learned model says")

    rule_based_correct_count = 0
    learned_model_correct_count = 0
    for weight_in_grams, surface_smoothness, true_label in TEST_FRUITS:
        rule_based_prediction = classify_with_hand_written_rules(weight_in_grams, surface_smoothness)
        learned_model_prediction = classify_with_learned_model(weight_in_grams, surface_smoothness, class_averages)
        rule_based_correct_count += rule_based_prediction == true_label
        learned_model_correct_count += learned_model_prediction == true_label
        print(f"  ({weight_in_grams:>3.0f}, {surface_smoothness:.2f})         {true_label:<9} {rule_based_prediction:<11} {learned_model_prediction}")

    print()
    print(f"Hand-written rules:  {rule_based_correct_count}/{len(TEST_FRUITS)} correct")
    print(f"Learned model:       {learned_model_correct_count}/{len(TEST_FRUITS)} correct")


if __name__ == "__main__":
    main()
