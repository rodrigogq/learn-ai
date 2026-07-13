/*
 * Chapter 19 - the greedy CTC decoder in pure C.
 *
 * The decoder is the deployment half of CTC: given per-frame scores from a
 * model (letters + blank), produce text by taking the best symbol per frame,
 * merging consecutive repeats, and dropping blanks. This program runs the
 * rule on a hand-built emission matrix and then demonstrates the classic
 * collapse cases - including why the blank symbol is what makes double
 * letters spellable.
 *
 * Build and run from the repository root:
 *     make -C chapters/19-speech-recognition/c
 *     ./chapters/19-speech-recognition/c/build/greedy_ctc_decoder
 */

#include <stdio.h>
#include <string.h>

#define ALPHABET_SIZE 5           /* A..E */
#define CLASS_COUNT (ALPHABET_SIZE + 1)
#define BLANK_INDEX ALPHABET_SIZE
#define MAX_FRAMES 64

static const char alphabet[ALPHABET_SIZE + 1] = "ABCDE";

/*
 * Greedy CTC decoding: argmax per frame, merge repeats, drop blanks.
 *
 * emission_scores:  frame_count rows of CLASS_COUNT scores (higher = more
 *                   likely); raw scores are fine - argmax ignores scaling
 * frame_count:      number of frames
 * decoded_text_out: receives the resulting string (caller provides space)
 *
 * The order of operations is the whole trick: merging repeats FIRST means
 * "AAAA" collapses to "A", and the blank BETWEEN two A's is what keeps a
 * genuine double letter apart.
 */
static void greedy_ctc_decode(const double emission_scores[][CLASS_COUNT], int frame_count,
                              char *decoded_text_out) {
    int output_length = 0;
    int previous_best = -1;
    for (int frame = 0; frame < frame_count; frame++) {
        int best_class = 0;
        for (int class_index = 1; class_index < CLASS_COUNT; class_index++) {
            if (emission_scores[frame][class_index] > emission_scores[frame][best_class]) {
                best_class = class_index;
            }
        }
        if (best_class != previous_best && best_class != BLANK_INDEX) {
            decoded_text_out[output_length++] = alphabet[best_class];
        }
        previous_best = best_class;
    }
    decoded_text_out[output_length] = '\0';
}

/*
 * Helper for the demos: fill an emission matrix so that each frame's argmax
 * follows a symbol string ('-' means blank), with mild scores elsewhere.
 *
 * frame_symbols:    e.g. "AA--BB-B" - one character per frame
 * emission_scores:  receives the matrix
 *
 * Returns the frame count.
 */
static int build_emissions_from_string(const char *frame_symbols,
                                       double emission_scores[][CLASS_COUNT]) {
    int frame_count = (int)strlen(frame_symbols);
    for (int frame = 0; frame < frame_count; frame++) {
        for (int class_index = 0; class_index < CLASS_COUNT; class_index++) {
            emission_scores[frame][class_index] = 0.1;
        }
        int winner = frame_symbols[frame] == '-' ? BLANK_INDEX
                                                 : (int)(strchr(alphabet, frame_symbols[frame]) - alphabet);
        emission_scores[frame][winner] = 0.9;
    }
    return frame_count;
}

int main(void) {
    static double emission_scores[MAX_FRAMES][CLASS_COUNT];
    char decoded_text[MAX_FRAMES + 1];

    printf("Greedy CTC decoding: argmax per frame -> merge repeats -> drop blanks\n\n");

    const char *demo_cases[][2] = {
        /* frames                     what it demonstrates */
        {"AAAA----BBBB",              "held notes collapse: many frames, two letters"},
        {"AA--A--BB---",              "blank between A's = a REAL double letter"},
        {"AAABBB",                    "no blank between A and B needed - they differ"},
        {"--A-A---",                  "blanks at will; repeats merge only when adjacent"},
        {"------------",              "all blank = empty transcription"},
    };
    const int demo_count = 5;

    for (int demo = 0; demo < demo_count; demo++) {
        int frame_count = build_emissions_from_string(demo_cases[demo][0], emission_scores);
        greedy_ctc_decode(emission_scores, frame_count, decoded_text);
        printf("  frames %-14s -> \"%s\"   (%s)\n",
               demo_cases[demo][0], decoded_text, demo_cases[demo][1]);
    }

    printf("\nWhy the blank exists: without it, 'AA' from holding a note and 'AA' the\n");
    printf("double letter would be indistinguishable. The blank is CTC's space bar\n");
    printf("between repeats - and during training, the network learns to emit it\n");
    printf("exactly where needed. This same decoder ships in production speech systems\n");
    printf("(with beam search on top when accuracy matters more than speed).\n");
    return 0;
}
