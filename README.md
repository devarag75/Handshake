# Handshake AI 🤝

## Basic Details

### Team Name
Simplist

### Team Members

- Team Lead: Abhinav s menon - Carmel College of Engineering and Technology
- Member 2: Devarag PD - Carmel College of Engineering and Technology

---

## Project Description

Handshake AI is an AI-powered system that detects whether two people are actually performing a handshake using a live camera. It uses computer vision and a trained machine-learning model to analyze the movement and trajectory of two hands.

Instead of simply detecting hands, the system learns the movement pattern of a handshake and distinguishes it from other movements such as waving, touching hands, or high-fiving.

---

## The Problem (that doesn't exist)

Have you ever wondered:

> "Was that actually a handshake, or did we just awkwardly touch hands?" 🤝

In a world full of awkward greetings, nobody knows whether a handshake was officially a handshake.

This completely unnecessary problem needed a completely unnecessary AI solution.

---

## The Solution (that nobody asked for)

We built Handshake AI.

Point the camera at two people, shake hands, and let AI decide whether the handshake was real.

The system works through the following process:

LIVE CAMERA  
↓  
HAND DETECTION  
↓  
TWO HANDS TRACKED  
↓  
HAND MOVEMENT / TRAJECTORY  
↓  
FEATURE EXTRACTION  
↓  
TRAINED AI MODEL  
↓  
HANDSHAKE / NO HANDSHAKE

Because apparently, humans can no longer be trusted to identify a handshake themselves. 🤖🤝

---

## Technical Details

### Technologies/Components Used

### Software

- Python
- OpenCV
- MediaPipe
- NumPy
- Pandas
- Scikit-learn
- Random Forest Classifier
- PySide6
- Pickle
- VS Code
- IP Webcam

### Machine Learning

The project uses a trained Random Forest Classifier.

The model analyzes movement-based features extracted from the tracked hands, including:

- Hand trajectory
- Position changes
- Velocity
- Speed
- Direction changes
- Path length
- Displacement
- Acceleration
- Movement ratio

The trained model classifies the movement into:

- HANDSHAKE
- NO_HANDSHAKE

### Hardware

- Smartphone camera
- Computer/Laptop
- USB connection / USB tethering
- Wi-Fi/IP camera connection

---

## Implementation

### Software Architecture

```text
Smartphone Camera
       ↓
IP Camera Stream
       ↓
OpenCV
       ↓
MediaPipe Hands
       ↓
Two-Hand Detection
       ↓
Palm Center Extraction
       ↓
Movement Trajectory
       ↓
Feature Extraction
       ↓
Random Forest AI
       ↓
HANDSHAKE / NO_HANDSHAKE
Project Demo
Screenshots

Video

https://drive.google.com/file/d/1Mi8NFgcSteP66sMh8qKXMuKNaMNNRIZN/view?usp=drivesdk

The demo shows two people performing a handshake while the AI tracks their hands and determines whether the movement represents a handshake.
