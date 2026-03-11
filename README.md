1. Problem Formulation
   The system is designed as an Early Warning System rather than a standard anomaly classifier.
   The model does not detect failures within the currently analyzed window; instead, it predicts their occurrence within a specified future time horizon.
   The problem is formulated using a sliding-window approach:
   Window Size (W = 100) The model analyzes the last 100 time steps
   Horizon Size (H = 20): If the algorithm recognizes a pattern indicating an impending failure, it raises an alert that a threshold breach (incident) will occur within the next 20 time steps.

2. Modeling Choices
  Architecture (LSTM): A Long Short-Term Memory network was chosen for its ability to maintain an internal memory state.
  Loss Function (BCEWithLogitsLoss): The Sigmoid activation layer was removed from the network's output in favor of computing the loss directly on raw logits. This ensures significantly higher numerical stability during backpropagation.
  Imbalance Handling: Incident datasets are inherently highly imbalanced (anomalies constitute a tiny fraction of the background). The loss function dynamically calculates the penalty weight (pos_weight) based on the training set distribution, forcing the model to prioritize the rare positive class.
3. Evaluation Setup & Metrics
   The standard Accuracy metric is useless for highly imbalanced problems. Therefore, the evaluation is based on Precision-Recall curve analysis.
   F1-Score Optimization: The system does not rely on a rigid, arbitrary cutoff threshold (e.g., 0.5). After training, a Precision-Recall curve is generated on the validation set, from which the algorithm automatically selects the threshold that maximizes the F1-Score. This guarantees a mathematically optimal trade-off between false alarms and missed incidents.
   The model is evaluated using a Confusion Matrix at the optimized threshold and the Average Precision Score.

   
