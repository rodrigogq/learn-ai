"""Chapter 4 - probability verified by brute force: dice, expectation, cross-entropy.

Three demonstrations:
  1. Simulate a million two-dice rolls; the empirical distribution matches the
     exact counting argument (law of large numbers).
  2. Sample averages converge to the expected values (3.5 for one die).
  3. Score two weather forecasters with cross-entropy, reproducing the
     chapter's hand-computed table.

Run from the repository root:
    .venv/bin/python chapters/04-probability-basics/python/dice_and_distributions.py
"""

import math
import random


def simulate_two_dice_distribution(number_of_rolls, random_generator):
    """Roll two dice many times and count how often each sum appears.

    Arguments:
        number_of_rolls: how many times to roll the pair of dice.
        random_generator: a random.Random instance. Passing it in (instead of
            using the global one) keeps every run reproducible from one seed.

    Returns a dict mapping each sum (2..12) to the fraction of rolls that
    produced it.
    """
    sum_counts = {possible_sum: 0 for possible_sum in range(2, 13)}
    for _ in range(number_of_rolls):
        first_die = random_generator.randint(1, 6)
        second_die = random_generator.randint(1, 6)
        sum_counts[first_die + second_die] += 1
    return {dice_sum: count / number_of_rolls for dice_sum, count in sum_counts.items()}


def exact_two_dice_probability(dice_sum):
    """Return the exact probability of a given two-dice sum by counting pairs.

    Arguments:
        dice_sum: the sum in question, 2 through 12.

    Out of 36 equally likely pairs, the number summing to s is 6 - |7 - s|
    (1 pair for 2 or 12, rising to 6 pairs for 7).
    """
    number_of_pairs = 6 - abs(7 - dice_sum)
    return number_of_pairs / 36.0


def estimate_expected_value_by_sampling(number_of_samples, random_generator):
    """Estimate E[one die] and E[sum of two dice] by averaging samples.

    Arguments:
        number_of_samples: how many rolls to average.
        random_generator: a random.Random instance.

    Returns the pair (average of one die, average of two-dice sums).
    """
    single_die_total = 0
    two_dice_total = 0
    for _ in range(number_of_samples):
        single_die_total += random_generator.randint(1, 6)
        two_dice_total += random_generator.randint(1, 6) + random_generator.randint(1, 6)
    return single_die_total / number_of_samples, two_dice_total / number_of_samples


def compute_cross_entropy_for_rain_forecasts(rain_probabilities_given, it_actually_rained):
    """Compute a forecaster's cross-entropy: their average surprise at reality.

    Arguments:
        rain_probabilities_given: list of probabilities the forecaster gave to
            "it will rain", one per day.
        it_actually_rained: list of booleans, one per day, True where it rained.

    Returns the average of -log(probability given to what actually happened).
    On a day without rain, the probability given to the actual outcome is
    1 - (rain probability), which is why both branches appear below.
    """
    total_surprise = 0.0
    for rain_probability, rained in zip(rain_probabilities_given, it_actually_rained):
        probability_given_to_actual_outcome = rain_probability if rained else 1.0 - rain_probability
        total_surprise += -math.log(probability_given_to_actual_outcome)
    return total_surprise / len(rain_probabilities_given)


def main():
    random_generator = random.Random(42)

    number_of_rolls = 1_000_000
    print(f"1. Two-dice distribution: {number_of_rolls:,} simulated rolls vs exact counting")
    empirical_distribution = simulate_two_dice_distribution(number_of_rolls, random_generator)
    print("  sum   simulated   exact")
    for dice_sum in range(2, 13):
        print(f"  {dice_sum:>3}   {empirical_distribution[dice_sum]:.5f}     {exact_two_dice_probability(dice_sum):.5f}")

    print()
    print("2. Expected values estimated by sampling (exact: 3.5 and 7.0)")
    for number_of_samples in (100, 10_000, 1_000_000):
        single_die_average, two_dice_average = estimate_expected_value_by_sampling(
            number_of_samples, random_generator
        )
        print(f"  {number_of_samples:>9,} samples: one die = {single_die_average:.4f}, two dice = {two_dice_average:.4f}")

    print()
    print("3. Cross-entropy: grading two weather forecasters (it rained on days 1, 2, 5)")
    it_actually_rained = [True, True, False, False, True]
    forecaster_a_probabilities = [0.8, 0.9, 0.2, 0.1, 0.6]
    forecaster_b_probabilities = [0.5, 0.5, 0.5, 0.5, 0.5]
    cross_entropy_a = compute_cross_entropy_for_rain_forecasts(forecaster_a_probabilities, it_actually_rained)
    cross_entropy_b = compute_cross_entropy_for_rain_forecasts(forecaster_b_probabilities, it_actually_rained)
    print(f"  forecaster A (confident and mostly right): {cross_entropy_a:.2f}")
    print(f"  forecaster B (always says 50/50):          {cross_entropy_b:.2f}")
    print("  Lower is better: cross-entropy rewards confident correctness.")


if __name__ == "__main__":
    main()
