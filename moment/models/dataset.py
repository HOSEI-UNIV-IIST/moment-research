import pandas as pd
import matplotlib.pyplot as plt

# CSV 로드
df = pd.read_csv("/home/jinseopalang/TSFM/moment-research/TimeseriesDatasets/forecasting/autoformer/ETTh1.csv")

# OT 열만 추출
ot = df["OT"].values  # shape: (17000,)

# 시각화
plt.figure(figsize=(15, 4))
plt.plot(ot, color='blue')
plt.title("OT Timeseries (Length: {})".format(len(ot)))
plt.xlabel("Time Step")
plt.ylabel("OT Value")
plt.grid(True)
plt.tight_layout()
plt.show()

#python moment/models/dataset.py