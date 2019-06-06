# Grab Challenge - Traffic Management
(https://www.aiforsea.com/traffic-management)

Problem statement: 
Predicting demand for Grab bookings, based on time-series data

Architecture:
GCN + LSTM + Linear, similar idea to https://arxiv.org/abs/1812.04206
Takes too long to run though (~ 45 mins/epoch on Google Colab)