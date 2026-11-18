#include "weights.h"
#include <stdint.h>

// Multiplier-less MAC menggunakan bit-shift (tanpa blok DSP)
static inline int32_t pot_mac(int32_t acc, int32_t act, int8_t pot_code) {
    if (pot_code == 0) return acc;
    int8_t mag = (pot_code > 0) ? pot_code : -pot_code;
    int32_t shifted = act >> (8 - mag);
    return (pot_code > 0) ? (acc + shifted) : (acc - shifted);
}

// Top-Level Fungsi Akselerator CNN untuk High-Level Synthesis (HLS)
void cnn_mask_hls(
    const int32_t input_img[32][32],
    int32_t scores[2],
    int *pred_class
) {
    int32_t conv_buf[4][28][28];
    int32_t pool_buf[4][7][7];

    // 1. Layer Conv1 (5x5) + ReLU
    for (int f = 0; f < 4; f++) {
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                int32_t acc = 0;
                for (int kr = 0; kr < 5; kr++) {
                    for (int kc = 0; kc < 5; kc++) {
                        acc = pot_mac(acc, input_img[r + kr][c + kc], CONV1_W[f][kr][kc]);
                    }
                }
                conv_buf[f][r][c] = (acc > 0) ? acc : 0; // Aktivasi ReLU
            }
        }
    }

    // 2. Layer MaxPool (4x4, stride 4)
    for (int f = 0; f < 4; f++) {
        for (int pr = 0; pr < 7; pr++) {
            for (int pc = 0; pc < 7; pc++) {
                int32_t max_val = 0;
                for (int mr = 0; mr < 4; mr++) {
                    for (int mc = 0; mc < 4; mc++) {
                        int32_t v = conv_buf[f][pr * 4 + mr][pc * 4 + mc];
                        if (v > max_val) max_val = v;
                    }
                }
                pool_buf[f][pr][pc] = max_val;
            }
        }
    }

    // 3. Layer Fully Connected (196 -> 2)
    for (int cls = 0; cls < 2; cls++) {
        int32_t acc = 0;
        int idx = 0;
        for (int f = 0; f < 4; f++) {
            for (int pr = 0; pr < 7; pr++) {
                for (int pc = 0; pc < 7; pc++) {
                    acc = pot_mac(acc, pool_buf[f][pr][pc], FC_W[cls][idx++]);
                }
            }
        }
        scores[cls] = acc;
    }

    // 4. Komparator Akhir (0: WithMask, 1: WithoutMask)
    *pred_class = (scores[1] > scores[0]) ? 1 : 0;
}
