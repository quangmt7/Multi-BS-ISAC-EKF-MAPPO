import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def saturation_model(x, a, b, c):
    """Mô hình bão hòa: y = c + a * exp(-b*x)."""
    return c + a * np.exp(-b * x)


def build_models():
    return {
        "Linear": lambda: make_pipeline(
            StandardScaler(),
            LinearRegression()
        ),
        "Polynomial degree 2": lambda: make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression()
        ),
        "Polynomial degree 3": lambda: make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression()
        ),
    }


def fit_saturation(x, y):
    """Fit mô hình bão hòa bằng nonlinear least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    # Giá trị khởi tạo
    c0 = float(np.percentile(y, 10))
    a0 = float(max(np.max(y) - c0, 1e-4))

    # Dùng miền x >= 0 cho AoI chuẩn hóa
    p0 = [a0, 1.0, c0]

    params, _ = curve_fit(
        saturation_model,
        x,
        y,
        p0=p0,
        bounds=(
            [-np.inf, 1e-8, -np.inf],
            [np.inf, np.inf, np.inf]
        ),
        maxfev=20000
    )

    return params


def predict_saturation(x_train, y_train, x_test):
    params = fit_saturation(x_train, y_train)
    return saturation_model(x_test, *params)


def evaluate_models(x, y, n_splits=5):
    """
    Đánh giá mô hình bằng TimeSeriesSplit.
    Trả về bảng kết quả và mô hình tốt nhất theo CV-RMSE.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    if len(x) < 30:
        raise ValueError("Cần tối thiểu 30 mẫu hợp lệ.")

    # Sắp xếp theo thứ tự thời gian ban đầu trước khi gọi hàm này.
    # Không sort theo AoI ở đây nếu dữ liệu là chuỗi thời gian.
    splitter = TimeSeriesSplit(n_splits=n_splits)
    results = []

    model_factories = build_models()

    for name, factory in model_factories.items():
        errors = []

        for train_idx, test_idx in splitter.split(x):
            model = factory()
            model.fit(x[train_idx].reshape(-1, 1), y[train_idx])

            pred = model.predict(x[test_idx].reshape(-1, 1))
            errors.extend((y[test_idx] - pred).tolist())

        cv_rmse = float(np.sqrt(np.mean(np.square(errors))))

        # Fit toàn bộ dữ liệu để tính R² và AIC tham khảo
        full_model = factory()
        full_model.fit(x.reshape(-1, 1), y)
        full_pred = full_model.predict(x.reshape(-1, 1))

        rss = float(np.sum((y - full_pred) ** 2))
        n = len(y)
        k = 2 if name == "Linear" else (
            3 if name == "Polynomial degree 2" else 4
        )
        rss_safe = max(rss, 1e-12)
        aic = n * np.log(rss_safe / n) + 2 * k

        results.append({
            "Model": name,
            "CV_RMSE": cv_rmse,
            "Train_RMSE": float(np.sqrt(mean_squared_error(y, full_pred))),
            "Train_R2": float(r2_score(y, full_pred)),
            "AIC_approx": float(aic),
            "_model": full_model
        })

    # Đánh giá mô hình bão hòa
    try:
        errors = []

        for train_idx, test_idx in splitter.split(x):
            pred = predict_saturation(
                x[train_idx], y[train_idx], x[test_idx]
            )
            errors.extend((y[test_idx] - pred).tolist())

        params = fit_saturation(x, y)
        full_pred = saturation_model(x, *params)
        rss = float(np.sum((y - full_pred) ** 2))
        n = len(y)
        k = 3
        aic = n * np.log(max(rss, 1e-12) / n) + 2 * k

        results.append({
            "Model": "Exponential saturation",
            "CV_RMSE": float(np.sqrt(np.mean(np.square(errors)))),
            "Train_RMSE": float(np.sqrt(mean_squared_error(y, full_pred))),
            "Train_R2": float(r2_score(y, full_pred)),
            "AIC_approx": float(aic),
            "_model": params
        })

    except (RuntimeError, ValueError, FloatingPointError) as exc:
        print(f"Bỏ qua mô hình bão hòa do không fit được: {exc}")

    results.sort(key=lambda row: row["CV_RMSE"])
    best_name = results[0]["Model"]
    best_model = results[0]["_model"]

    table = pd.DataFrame([
        {k: v for k, v in row.items() if k != "_model"}
        for row in results
    ])

    return table, best_name, best_model


def predict_model(model_name, model, x_new):
    x_new = np.asarray(x_new, dtype=float).reshape(-1)

    if model_name == "Exponential saturation":
        return saturation_model(x_new, *model)

    return model.predict(x_new.reshape(-1, 1))


def analyze_relationship(x, y, label, save_dir="results/fit_analysis"):
    os.makedirs(save_dir, exist_ok=True)

    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]

    # Cross-validation phải giữ đúng thứ tự thời gian ban đầu.
    table, best_name, best_model = evaluate_models(x, y)

    print(f"\n===== {label} =====")
    print(table.to_string(index=False))
    print(f"\nMô hình có CV-RMSE thấp nhất: {best_name}")

    # Vẽ đường fit theo trục AoI tăng dần
    x_plot = np.linspace(np.min(x), np.max(x), 300)
    y_plot = predict_model(best_name, best_model, x_plot)

    # Vẽ scatter theo AoI để quan sát quan hệ giữa hai đại lượng
    order = np.argsort(x)

    plt.figure(figsize=(9, 5))
    plt.scatter(x, y, s=10, alpha=0.18, label="Observed data")
    plt.plot(
        x_plot,
        y_plot,
        linewidth=2.5,
        label=f"Best model: {best_name}"
    )
    plt.xlabel("Normalized AoI")
    plt.ylabel(label)
    plt.title(f"{label} vs. Normalized AoI")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    figure_path = os.path.join(
        save_dir, f"{label.lower().replace(' ', '_')}_fit.png"
    )
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close()

    table_path = os.path.join(
        save_dir, f"{label.lower().replace(' ', '_')}_model_comparison.csv"
    )
    table.to_csv(table_path, index=False)

    print(f"Đã lưu biểu đồ: {os.path.abspath(figure_path)}")
    print(f"Đã lưu bảng so sánh: {os.path.abspath(table_path)}")

    return table, best_name, best_model


def run_analysis(aoi_norm, prb_sensing_ratio, power_sensing_ratio):
    prb_results = analyze_relationship(
        aoi_norm,
        prb_sensing_ratio,
        label="Sensing PRB ratio"
    )

    power_results = analyze_relationship(
        aoi_norm,
        power_sensing_ratio,
        label="Sensing power ratio"
    )

    return prb_results, power_results


# Ví dụ sử dụng:
# prb_results, power_results = run_analysis(
#     self.aoi_norm,
#     self.prb,
#     self.power
# )