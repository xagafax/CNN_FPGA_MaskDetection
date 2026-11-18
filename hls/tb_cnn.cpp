#include <stdio.h>
#include "test_samples.h"

void cnn_mask_hls(const int32_t input_img[32][32], int32_t scores[2], int *pred_class);

int main() {
    int correct = 0;
    printf("=== SIMULASI INFERENSI C++ HLS (MULTIPLIER-LESS PoT) ===\n");
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        int32_t scores[2];
        int pred = 0;
        cnn_mask_hls(TEST_IMAGES[i], scores, &pred);
        if (pred == TEST_LABELS[i]) correct++;
        printf("Sample #%d | Label: %d | Pred: %d | Scores: [%d, %d]\n",
               i, TEST_LABELS[i], pred, scores[0], scores[1]);
    }
    printf("Akurasi Uji HLS (10 Sampel): %d / %d\n", correct, NUM_TEST_SAMPLES);
    return 0;
}
