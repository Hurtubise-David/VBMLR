# Real-Time Eye-Gaze Estimation on a Computer Screen

![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/opencv-4.x-red)

This repository contains the official implementation of the research paper **"Real time eye-gaze estimation on a computer screen"**.

It presents a low-cost eye-tracking solution that operates on standard 2D webcams (like integrated laptop cameras) without requiring training with massive amount of data, specialized hardware (IR) or complex calibration.

## Abstract

Eye-gaze estimation is a critical component in Human-Computer Interaction (HCI), cognitive science, and driver monitoring. Traditional methods often rely on expensive infrared hardware or require restrictive user calibration.

This project introduces a passive computer vision approach based on Variational Bayesian Multinomial Logistic Regression (VBMLR). The system estimates gaze direction by analyzing:
1.  **Depth-from-Defocus (Blur)**: Estimating the head's Z-position by analyzing the blur level on the user's nose (optical defocus).
2.  **Geometric Features**: Extracting the iris position relative to the head pose using a radial derivative integral.
3.  **Probabilistic Mapping**: Using VBMLR to map these features to screen zones (1-9) with high accuracy even on small training datasets.


## Project Structure
