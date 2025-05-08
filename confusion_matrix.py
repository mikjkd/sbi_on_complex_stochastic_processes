import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_true = [2, 0, 2, 2, 0, 1] #DA SOSTITUIRE CON LE CORRETTE TRUE LABEL
y_pred = [0, 0, 2, 2, 0, 2] #DA SOSTITUIRE CON LE CORRETTE PREDICTED LABEL

custom_cmap = plt.cm.RdYlGn.reversed()
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
fig, ax = plt.subplots(figsize=(6, 6)) #Altri esempi: 'viridis', 'Greens', 'Oranges'
disp.plot(cmap='Blues', values_format='d', colorbar=True, ax=ax) #disp.plot(cmap=custom_cmap, values_format='d', colorbar=True, ax=ax)

plt.xlabel("Predicted", fontsize=12)
plt.ylabel("Real", fontsize=12)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.grid(False)
plt.tight_layout()
plt.show()
