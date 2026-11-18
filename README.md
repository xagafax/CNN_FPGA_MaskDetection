# Klasifikasi Penggunaan Masker Wajah Menggunakan Akselerator CNN Berbasis FPGA

**Nama:** Muhammad Gazali  
**NIM:** 101022330164  
**Mata Kuliah:** Desain SoC dan FPGA  

Repositori ini berisi implementasi *end-to-end* perancangan akselerator perangkat keras **Convolutional Neural Network (CNN)** berdaya rendah untuk klasifikasi penggunaan masker wajah (*With Mask* vs *Without Mask*) menggunakan **Quantization-Aware Training (QAT) Power-of-Two (±2^k)** dan **High-Level Synthesis (HLS)**.

## 1. Dataset & Pra-pemrosesan
- **Dataset:** Face Mask 12K Images Dataset (`ashishjangra27/face-mask-12k-images-dataset` — Kaggle)
- **Total Sampel:** 11.792 citra wajah (**5.883** *With Mask* dan **5.909** *Without Mask*), resolusi **32×32 piksel (1-channel Grayscale)**
- **Pembagian Data:** 80% Training (9.433 citra) dan 20% Validation/Test (2.359 citra)

## 2. Arsitektur Jaringan (`TinyMaskCNN_32x32`)
| Layer | Konfigurasi | Dimensi Output | Jumlah Parameter | Memori Bobot (PoT 8-bit) |
| :--- | :--- | :---: | :---: | :---: |
| **Input** | Grayscale 1-Channel | 1 × 32 × 32 | — | — |
| **Conv1 + ReLU** | 4 Filter, Kernel 5×5, `s=1, p=0` | 4 × 28 × 28 | 100 | 100 B |
| **MaxPool1** | Window 4×4, `s=4` | 4 × 7 × 7 | — | — |
| **FC1 (Output)** | `Linear(196 → 2)` | 2 | 392 | 392 B |
| **Total** | — | — | **492** | **492 B (~0,48 KB)** |

## 3. Optimasi Perangkat Keras (QAT PoT & HLS)
1. **Multiplier-less MAC (`pot_mac`):** Bobot dikuantisasi menjadi nilai pangkat dua (`±2^k`) sehingga perkalian di `hls/cnn_hls.cpp` digantikan sepenuhnya oleh operasi *bit-shift* (`>>`) tanpa blok DSP.
2. **Aritmetika Fixed-Point Q4.12:** Aktivasi piksel direpresentasikan menggunakan `int32_t` dengan faktor skala `2^12 = 4096`.

## 4. Hasil Pengujian & Evaluasi
| Tahap Pengujian | Presisi Bobot | Akurasi Validasi (2.359 Citra) | Keterangan |
| :--- | :---: | :---: | :--- |
| **Tahap 1: Standard Training** | `Float32` (32-bit) | **96,69%** | *Baseline* sebelum kuantisasi |
| **Tahap 2: QAT Fine-Tuning** | Power-of-Two (±2^k) | **96,35%** | Siap sintesis HLS tanpa *multiplier* |

## 5. Cara Menjalankan
```bash
cd hls && make run
```