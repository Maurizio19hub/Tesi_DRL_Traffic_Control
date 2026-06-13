import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3.common.monitor import load_results
from stable_baselines3.common.results_plotter import rolling_window

# Carica i risultati
df = load_results("../models/correct_time/t20energy-pedwaiting-collisions_f2")

# Grafico con media mobile a 10 episodi
plt.figure(figsize=(12, 5))
plt.plot(df['r'], alpha=0.3, label='reward per episodio')
plt.plot(df['r'].rolling(10).mean(), label='media mobile (10 ep)')
plt.xlabel('Episodio')
plt.ylabel('Reward')
plt.title('Training reward')
plt.legend()
plt.grid(True)
plt.savefig("../models/correct_time/t20energy-pedwaiting-collisions_f2/training_curve.png", dpi=150)
plt.show()

print(df.describe())