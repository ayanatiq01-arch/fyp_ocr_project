# Evaluation

`DeepSeek_OCR_accuracy_test.ipynb` runs [DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR) on the test pages. It prints its character error rate (CER) next to the current pipeline (UTRNet + PaddleOCR, average 1.48 %).

DeepSeek-OCR needs an NVIDIA GPU with CUDA and about 6.7 GB for the model, so it cannot run on the development PC (Intel HD 520, 8 GB RAM).

To run it:

1. Open the notebook in Google Colab.
2. Choose the **T4 GPU** runtime.
3. Run all cells.
4. When asked, upload the test images: `real1.png`, `real2.png`, `real5.jpg`, `page_photo.jpg`, `page_photo2.jpg`, `page_clean.png`.

CER counts letters only. Harakat, punctuation and spaces are ignored, and Urdu/Arabic letter variants are unified, the same as in the project's own evaluation.
