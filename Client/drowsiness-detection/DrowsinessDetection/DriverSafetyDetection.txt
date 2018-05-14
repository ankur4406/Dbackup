@echo off
python %~dp0\scripts\detect_drowsiness.py --shape-predictor %~dp0\scripts\shape_predictor_68_face_landmarks.dat %*
pause