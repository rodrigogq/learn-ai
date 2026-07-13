/*
 * Chapter 18 - a WAV file and an FFT, both from scratch in pure C.
 *
 * Three parts:
 *   1. synthesize an A-major chord and WRITE it as a real .wav file (the
 *      44-byte header is spelled out field by field - play the file with
 *      any audio player to hear your arithmetic),
 *   2. read the file back and verify the samples survived,
 *   3. run a from-scratch radix-2 FFT (the actual Cooley-Tukey algorithm,
 *      not the naive DFT) and report the three loudest frequencies - the
 *      chord's notes come back out.
 *
 * Build and run from the repository root:
 *     make -C chapters/18-sound-and-spectrograms/c
 *     ./chapters/18-sound-and-spectrograms/c/build/wav_and_fft
 *     (then listen: open chord.wav)
 */

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SAMPLE_RATE 8000
#define CLIP_SAMPLES 8192            /* ~1 second; a power of two so the FFT can split it evenly */
#define PI 3.14159265358979323846

/*
 * Write a mono 16-bit WAV file. The header is three little chunks of plain
 * bookkeeping; every field is written explicitly below so nothing is hidden.
 *
 * file_path:     where to write
 * samples:       sample_count values in -1..1
 * sample_count:  how many samples
 */
static void write_wav_file(const char *file_path, const double *samples, int sample_count) {
    FILE *file = fopen(file_path, "wb");
    if (file == NULL) {
        fprintf(stderr, "cannot write %s\n", file_path);
        exit(1);
    }

    int32_t data_bytes = sample_count * 2;           /* 16-bit mono: 2 bytes per sample */
    int32_t riff_size = 36 + data_bytes;
    int16_t audio_format = 1;                        /* 1 = uncompressed PCM */
    int16_t channel_count = 1;
    int32_t sample_rate = SAMPLE_RATE;
    int32_t byte_rate = SAMPLE_RATE * 2;
    int16_t block_align = 2;
    int16_t bits_per_sample = 16;

    fwrite("RIFF", 1, 4, file);
    fwrite(&riff_size, 4, 1, file);
    fwrite("WAVE", 1, 4, file);
    fwrite("fmt ", 1, 4, file);
    int32_t format_chunk_size = 16;
    fwrite(&format_chunk_size, 4, 1, file);
    fwrite(&audio_format, 2, 1, file);
    fwrite(&channel_count, 2, 1, file);
    fwrite(&sample_rate, 4, 1, file);
    fwrite(&byte_rate, 4, 1, file);
    fwrite(&block_align, 2, 1, file);
    fwrite(&bits_per_sample, 2, 1, file);
    fwrite("data", 1, 4, file);
    fwrite(&data_bytes, 4, 1, file);

    for (int i = 0; i < sample_count; i++) {
        /* Scale -1..1 into the 16-bit integer range the format stores. */
        int16_t quantized = (int16_t)(samples[i] * 32000.0);
        fwrite(&quantized, 2, 1, file);
    }
    fclose(file);
}

/*
 * Read the samples back out of a mono 16-bit WAV file written by the
 * function above (skips the 44-byte header it wrote).
 */
static int read_wav_file(const char *file_path, double *samples, int maximum_samples) {
    FILE *file = fopen(file_path, "rb");
    if (file == NULL) {
        fprintf(stderr, "cannot read %s\n", file_path);
        exit(1);
    }
    fseek(file, 44, SEEK_SET);
    int sample_count = 0;
    int16_t quantized;
    while (sample_count < maximum_samples && fread(&quantized, 2, 1, file) == 1) {
        samples[sample_count++] = quantized / 32000.0;
    }
    fclose(file);
    return sample_count;
}

/*
 * The radix-2 Cooley-Tukey FFT, iterative version, from scratch.
 *
 * real, imaginary:  arrays of length sample_count holding the signal
 *                   (imaginary all zero on input); overwritten with the
 *                   transform. sample_count must be a power of two.
 *
 * Two stages:
 *   1. bit-reversal permutation - reorder the input so that the merging
 *      stages below can work in place;
 *   2. log2(N) rounds of "butterflies" - each round merges pairs of smaller
 *      transforms into one twice as large, using the twiddle factors
 *      (cos, -sin of evenly spaced angles).
 * Total work: N log2(N) instead of the naive N^2 - for 8192 samples that is
 * ~106 thousand operations instead of ~67 million.
 */
static void fast_fourier_transform(double *real, double *imaginary, int sample_count) {
    /* Stage 1: bit-reversal permutation. */
    for (int i = 1, reversed = 0; i < sample_count; i++) {
        int bit = sample_count >> 1;
        for (; reversed & bit; bit >>= 1) {
            reversed ^= bit;
        }
        reversed ^= bit;
        if (i < reversed) {
            double swap_value = real[i]; real[i] = real[reversed]; real[reversed] = swap_value;
            swap_value = imaginary[i]; imaginary[i] = imaginary[reversed]; imaginary[reversed] = swap_value;
        }
    }

    /* Stage 2: butterfly rounds, doubling the transform length each time. */
    for (int transform_length = 2; transform_length <= sample_count; transform_length <<= 1) {
        double angle_step = -2.0 * PI / transform_length;
        for (int block_start = 0; block_start < sample_count; block_start += transform_length) {
            for (int pair_index = 0; pair_index < transform_length / 2; pair_index++) {
                double angle = angle_step * pair_index;
                double twiddle_real = cos(angle);
                double twiddle_imaginary = sin(angle);

                int even_index = block_start + pair_index;
                int odd_index = even_index + transform_length / 2;

                double odd_rotated_real = real[odd_index] * twiddle_real - imaginary[odd_index] * twiddle_imaginary;
                double odd_rotated_imaginary = real[odd_index] * twiddle_imaginary + imaginary[odd_index] * twiddle_real;

                real[odd_index] = real[even_index] - odd_rotated_real;
                imaginary[odd_index] = imaginary[even_index] - odd_rotated_imaginary;
                real[even_index] += odd_rotated_real;
                imaginary[even_index] += odd_rotated_imaginary;
            }
        }
    }
}

int main(void) {
    static double samples[CLIP_SAMPLES];
    static double real[CLIP_SAMPLES];
    static double imaginary[CLIP_SAMPLES];

    /* 1. Synthesize the A-major chord: 440.0, 554.4, 659.3 Hz. */
    printf("1. Synthesizing an A-major chord (440.0 + 554.4 + 659.3 Hz) and writing chord.wav\n");
    for (int i = 0; i < CLIP_SAMPLES; i++) {
        double time_seconds = (double)i / SAMPLE_RATE;
        samples[i] = (sin(2 * PI * 440.0 * time_seconds)
                    + sin(2 * PI * 554.4 * time_seconds)
                    + sin(2 * PI * 659.3 * time_seconds)) / 3.0;
    }
    write_wav_file("chord.wav", samples, CLIP_SAMPLES);
    printf("   wrote chord.wav (%d samples) - open it with any audio player.\n\n", CLIP_SAMPLES);

    /* 2. Read it back. */
    int read_count = read_wav_file("chord.wav", real, CLIP_SAMPLES);
    printf("2. Read back %d samples; first sample %.4f (matches %.4f up to 16-bit rounding)\n\n",
           read_count, real[0], samples[0]);

    /* 3. FFT and peak report. */
    memset(imaginary, 0, sizeof(imaginary));
    fast_fourier_transform(real, imaginary, CLIP_SAMPLES);

    printf("3. From-scratch FFT: the three loudest frequencies\n");
    double frequency_resolution = (double)SAMPLE_RATE / CLIP_SAMPLES;
    for (int peak = 0; peak < 3; peak++) {
        int best_index = 0;
        double best_magnitude = 0.0;
        for (int i = 1; i < CLIP_SAMPLES / 2; i++) {
            double magnitude = sqrt(real[i] * real[i] + imaginary[i] * imaginary[i]);
            if (magnitude > best_magnitude) {
                best_magnitude = magnitude;
                best_index = i;
            }
        }
        printf("   peak %d: %.1f Hz\n", peak + 1, best_index * frequency_resolution);
        /* Zero out a small neighborhood so the next pass finds the next note
         * rather than this peak's shoulder. */
        for (int i = best_index - 3; i <= best_index + 3; i++) {
            if (i >= 0 && i < CLIP_SAMPLES) {
                real[i] = imaginary[i] = 0.0;
            }
        }
    }
    printf("   The chord came back out of the arithmetic. That is the Fourier transform.\n");
    return 0;
}
