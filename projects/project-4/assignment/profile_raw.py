import pandas as pd, json

a = pd.read_csv("src/data/sensor_A.csv", dtype=str, keep_default_na=False)
print("=== sensor_A ===")
print(a.columns.tolist())
print(a)

b = json.load(open("src/data/sensor_B.json"))
print("\n=== sensor_B ===")
print(b.keys())
print(b["readings"][0])

c = pd.read_csv("src/sensor_C.csv", dtype=str, keep_default_na=False)
print("\n=== sensor_C ===")
print(c.columns.tolist())
print(c)


