/*
 * Chapter 31 - a model embedded in a C application: the whole course in one
 * closing example.
 *
 * This is what "deployment" ultimately means: a trained model's weights baked
 * directly into a small program that does a real job with no framework, no
 * Python, no dependencies - the kind of thing that runs on a microcontroller,
 * inside a camera, or in a game. Here a tiny hand-set 2-feature classifier
 * (the Chapter 1 fruit problem, come full circle) is embedded as a constant
 * array and used to classify inputs in a simple interactive-style loop.
 *
 * The point is not the model - it is the SHAPE: real trained weights, a
 * forward pass in a few lines, running anywhere C runs. Chapter 25 already
 * did this for a whole LLM; this closes the loop by showing how small the
 * deployed core of ANY model is.
 *
 * Build and run from the repository root:
 *     make -C chapters/31-deployment/c
 *     ./chapters/31-deployment/c/build/embedded_classifier
 */

#include <stdio.h>

/* The "trained" model, baked in as constants. In a real deployment these
 * numbers come from the export scripts of earlier chapters; the deployment
 * code that USES them looks exactly like this regardless of model size. */
#define FEATURE_COUNT 2
#define CLASS_COUNT 2
static const char *class_names[CLASS_COUNT] = {"apple", "orange"};

/* A one-layer classifier: score[c] = sum_f weight[c][f]*feature[f] + bias[c].
 * Weights chosen so heavier + rougher leans orange, lighter + smoother apple
 * (Chapter 6's logistic regression, deployed). */
static const float weights[CLASS_COUNT][FEATURE_COUNT] = {
    {-2.0f,  2.0f},   /* apple:  likes low weight, high smoothness */
    { 2.0f, -2.0f},   /* orange: likes high weight, low smoothness */
};
static const float biases[CLASS_COUNT] = {2.25f, -2.25f};

/*
 * The deployed forward pass - the entire "AI" of this program.
 *
 * features:  FEATURE_COUNT inputs (weight-in-grams/100, smoothness 0..1)
 *
 * Returns the predicted class index. For a real model this function would be
 * bigger, but its role - numbers in, class out, no framework - is identical.
 */
static int classify(const float *features) {
    int best_class = 0;
    float best_score = -1e30f;
    for (int c = 0; c < CLASS_COUNT; c++) {
        float score = biases[c];
        for (int f = 0; f < FEATURE_COUNT; f++) {
            score += weights[c][f] * features[f];
        }
        if (score > best_score) {
            best_score = score;
            best_class = c;
        }
    }
    return best_class;
}

int main(void) {
    printf("An embedded classifier: trained weights baked into a C program.\n");
    printf("No framework, no Python - this is the deployed core of a model.\n\n");

    /* A handful of test fruits (weight/100, smoothness), the Chapter 1 data. */
    float test_fruits[][FEATURE_COUNT] = {
        {1.20f, 0.90f},   /* light, smooth  */
        {1.85f, 0.40f},   /* heavy, rough   */
        {1.60f, 0.80f},   /* heavy but smooth - the interesting case */
        {1.30f, 0.55f},
    };
    int fruit_count = 4;

    printf("  weight  smooth   -> prediction\n");
    for (int i = 0; i < fruit_count; i++) {
        int prediction = classify(test_fruits[i]);
        printf("   %.0f g   %.2f    -> %s\n",
               test_fruits[i][0] * 100, test_fruits[i][1], class_names[prediction]);
    }

    printf("\nThat is the whole journey: Chapter 1 asked 'rules or learning?'; Chapters 2-30\n");
    printf("built every kind of model from scratch; and deployment is just running the\n");
    printf("learned numbers, right here, in a program that fits on anything. You now know\n");
    printf("the entire path from raw pixels and text to a running AI - end to end.\n");
    return 0;
}
