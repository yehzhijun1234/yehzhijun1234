import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor, plot_importance
from sklearn.metrics import mean_squared_error, r2_score
import joblib  # 儲存模型用

# 讀取資料
df = pd.read_csv("Taipei_house.csv")

# 顯示資料基本資訊
print(df.head())
print(df.columns)

# 去除缺失值
df = df.dropna()

# 特徵與目標變數
X = df.drop("price", axis=1)
y = df["price"]

# 類別型變數處理
X = pd.get_dummies(X)

# 特徵標準化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割資料集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 建立並訓練模型
model = XGBRegressor()
model.fit(X_train, y_train)

# 預測
y_pred = model.predict(X_test)

# 模型評估
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse}")
print(f"R^2 Score: {r2}")

# ✅ 特徵重要性視覺化
plt.figure(figsize=(10, 6))
plot_importance(model, max_num_features=10, importance_type='gain')
plt.title("特徵重要性 (前10名)")
plt.tight_layout()
plt.show()

# ✅ 預測結果 vs 實際值圖表
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("實際價格")
plt.ylabel("預測價格")
plt.title("預測 vs 實際")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--')  # 對角線
plt.tight_layout()
plt.show()

# ✅ 儲存模型與標準化器
joblib.dump(model, "xgb_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("模型與標準化器已儲存。")

# ✅ 額外：交叉驗證（10折）
cv_scores = cross_val_score(model, X_scaled, y, scoring='r2', cv=10)
print(f"10折交叉驗證 R^2 分數：{cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

