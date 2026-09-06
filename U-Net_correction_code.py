import os
import numpy as np
import xarray as xr
import tensorflow as tf
import pandas as pd
from tensorflow.keras import layers, models, Input, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# =====================================================
# GPU 配置与优化
# =====================================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
        print(f"✅ 使用 {len(gpus)} 个GPU")
    except RuntimeError as e:
        print(f"GPU配置错误: {e}")
else:
    print("⚠️ 未检测到GPU，将使用CPU运行")

# =====================================================
# 随机种子
# =====================================================
np.random.seed(42)
tf.random.set_seed(42)

# =====================================================
# 年份设置
# =====================================================
TRAIN_YEARS = list(range(1980, 2015))
VAL_YEARS   = list(range(2015, 2025))

# =====================================================
# 模型参数
# =====================================================
DROPOUT_RATE = 0.2
L2_REG = 1e-4
FILTERS = [32, 64, 128, 256, 512, 1024]
DOWNSAMPLE_STEPS = 5

LEARNING_RATE = 1e-4
BATCH_SIZE = 4
EPOCHS = 100

# =====================================================
# 路径
# =====================================================
ERA5_DIR = r'D:\1980-2014\era5'
CMIP6_HIST_DIR = r'D:\1980-2014\historical'
CMIP6_SSP_DIR  = r'D:\1980-2014\ssp370'
OUTPUT_DIR = r'D:\1980-2014\correct'
# =====================================================
# 工具：检查路径和文件
# =====================================================
def validate_paths():
    """验证所有路径是否存在"""
    paths = {
        "ERA5": ERA5_DIR,
        "CMIP6历史": CMIP6_HIST_DIR,
        "CMIP6未来": CMIP6_SSP_DIR,
        "输出": OUTPUT_DIR
    }

    all_valid = True
    for name, path in paths.items():
        if not os.path.exists(path):
            print(f"❌ {name} 路径不存在: {path}")
            all_valid = False
        else:
            print(f"✅ {name} 路径有效: {path}")

    if not all_valid:
        raise FileNotFoundError("请检查上述路径是否存在")

    return True

def find_cmip6_file(year):
    """查找CMIP6文件 - 适配新命名格式"""
    base_dir = CMIP6_HIST_DIR if year <= 2014 else CMIP6_SSP_DIR
    mode = "historical" if year <= 2014 else "ssp370"

    if not os.path.exists(base_dir):
        return None

    # 构建文件名模式
    filename_pattern = f"rsds_Amon_EC-Earth3_{mode}_r1i1p1f1_gr_{year}01-{year}12.nc"
    full_path = os.path.join(base_dir, filename_pattern)

    if os.path.exists(full_path):
        print(f"✅ 找到CMIP6文件 ({year}): {filename_pattern}")
        return full_path

    # 备用方案：部分匹配查找
    for f in os.listdir(base_dir):
        if f.endswith(".nc") and str(year) in f and mode in f:
            full_path = os.path.join(base_dir, f)
            print(f"✅ 找到CMIP6文件 ({year}): {f}")
            return full_path

    print(f"⚠️ 未找到 {year} 年CMIP6文件，期望文件名: {filename_pattern}")
    return None

def find_era5_file(year):
    """查找ERA5文件 - 适配新命名格式"""
    filename = f"ERA5_SSRD_monthly_{year}.nc"
    filepath = os.path.join(ERA5_DIR, filename)

    if os.path.exists(filepath):
        print(f"✅ 找到ERA5文件 ({year}): {filename}")
        return filepath

    # 备用方案：部分匹配查找
    for f in os.listdir(ERA5_DIR):
        if f.endswith(".nc") and str(year) in f and "SSRD" in f.upper():
            full_path = os.path.join(ERA5_DIR, f)
            print(f"✅ 找到ERA5文件 ({year}): {f}")
            return full_path

    print(f"⚠️ 未找到 {year} 年ERA5文件，期望文件名: {filename}")
    return None

# =====================================================
# 数据加载（训练）- 优化版
# =====================================================
def load_training_data():
    print("\n=== 开始加载训练数据 ===")

    # 验证路径
    for name, path in [("ERA5", ERA5_DIR), ("CMIP6", CMIP6_HIST_DIR), ("输出", OUTPUT_DIR)]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name}路径不存在: {path}")
        print(f"✅ {name}路径有效: {path}")

    X_list, y_list = [], []
    lat, lon = None, None
    loaded_years = []

    for year in TRAIN_YEARS:
        era5_file = find_era5_file(year)
        cmip6_file = find_cmip6_file(year)

        if era5_file is None or cmip6_file is None:
            print(f"⚠️ 跳过 {year} 年（文件缺失）")
            continue

        try:
            # 加载数据
            era5_ds = xr.open_dataset(era5_file)
            cmip_ds = xr.open_dataset(cmip6_file)

            # 查找变量名
            era5_var = None
            for var in ['ssrd_regridded', 'ssrd', 'SSRD']:
                if var in era5_ds:
                    era5_var = var
                    break

            cmip_var = None
            for var in ['ssrd_regridded', 'rsds', 'ssrd']:
                if var in cmip_ds:
                    cmip_var = var
                    break

            if era5_var is None or cmip_var is None:
                print(f"❌ {year} 年找不到合适的变量")
                print(f"   ERA5可用变量: {list(era5_ds.data_vars)}")
                print(f"   CMIP6可用变量: {list(cmip_ds.data_vars)}")
                continue

            # 获取数据
            era5 = era5_ds[era5_var]
            cmip = cmip_ds[cmip_var]

            # 打印单位信息（调试用）
            era5_units = era5.attrs.get('units', 'unknown')
            cmip_units = cmip.attrs.get('units', 'unknown')
            print(f"📊 {year}年单位 - ERA5({era5_var}): {era5_units}, CMIP6({cmip_var}): {cmip_units}")

            # =====================================================
            # ERA5单位转换 - 修正版
            # =====================================================
            # 无论单位字符串是什么，只要变量名是ssrd或ssrd_regridded，都进行转换
            # 这是因为ERA5的SSRD数据通常以J/m²为单位，需要转换为W/m²
            if era5_var.lower() in ['ssrd', 'ssrd_regridded']:
                era5 = era5 / 86400.0
                print(f"   ✅ ERA5已转换为W/m² (原单位: {era5_units})")
                #转换后更新单位信息（可选）
                era5.attrs['units'] = 'W m-2'
            else:
                print(f"   ⚠️ ERA5变量名不是ssrd，未转换 (变量: {era5_var})")

            # CMIP6数据单位检查
            if 'w/m' not in cmip_units.lower():
                print(f"⚠️ CMIP6单位不是W/m²: {cmip_units}，可能需要调整")
            else:
                print(f"   ✅ CMIP6单位正确: {cmip_units}")

            # 检查维度并统一
            if era5.shape != cmip.shape:
                print(f"⚠️ {year} 年维度不匹配: ERA5{era5.shape} vs CMIP6{cmip.shape}")
                # 尝试对齐维度（如果时间维度不同）
                if era5.shape[0] != cmip.shape[0]:
                    min_time = min(era5.shape[0], cmip.shape[0])
                    era5 = era5[:min_time]
                    cmip = cmip[:min_time]
                    print(f"   裁剪后维度: ERA5{era5.shape} vs CMIP6{cmip.shape}")
                else:
                    continue

            if lat is None:
                lat = cmip.lat.values
                lon = cmip.lon.values
                print(f"📊 数据维度: {cmip.shape} (time, lat, lon)")

            # 打印数据范围（调试用）
            print(f"   ERA5范围: [{era5.min().values:.3f}, {era5.max().values:.3f}]")
            print(f"   CMIP6范围: [{cmip.min().values:.3f}, {cmip.max().values:.3f}]")

            X_list.append(cmip.values)
            y_list.append(era5.values)
            loaded_years.append(year)
            print(f"✅ 成功加载 {year} 年数据")

        except Exception as e:
            print(f"❌ 加载 {year} 年数据失败: {e}")
            continue

    if not X_list:
        raise ValueError("❌ 未加载到任何训练数据！")

    print(f"\n📊 成功加载年份: {loaded_years}")

    # 合并数据
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)

    # 增加通道维度
    X = X[..., np.newaxis]
    y = y[..., np.newaxis]

    # 处理NaN值
    X = np.nan_to_num(X)
    y = np.nan_to_num(y)

    print(f"✅ 最终数据形状: X={X.shape}, y={y.shape}")
    print(f"   X范围: [{X.min():.3f}, {X.max():.3f}]")
    print(f"   y范围: [{y.min():.3f}, {y.max():.3f}]")

    return X, y, lat, lon

# =====================================================
# 标准化（修正版）
# =====================================================
def scale_data(X, y):
    """标准化数据，返回两个scaler"""
    ns, nx, ny, nc = X.shape
    sx = StandardScaler()
    sy = StandardScaler()

    # 分别标准化
    X_reshaped = X.reshape(ns, -1)
    y_reshaped = y.reshape(ns, -1)

    Xs = sx.fit_transform(X_reshaped).reshape(ns, nx, ny, nc)
    ys = sy.fit_transform(y_reshaped).reshape(ns, nx, ny, nc)

    print(f"\n📊 标准化后:")
    print(f"   Xs范围: [{Xs.min():.3f}, {Xs.max():.3f}]")
    print(f"   ys范围: [{ys.min():.3f}, {ys.max():.3f}]")

    return Xs, ys, sx, sy

# =====================================================
# 新U-Net模型 - 替换原build_unet函数（已修正）
# =====================================================
def build_unet(input_shape):
    """使用新架构构建U-Net模型"""

    def double_conv_block(x, n_filters):
        """双卷积块"""
        x = layers.Conv2D(n_filters, 3, padding="same", activation='tanh', kernel_initializer="he_normal")(x)
        x = layers.Conv2D(n_filters, 3, padding="same", activation='tanh', kernel_initializer="he_normal")(x)
        return x

    def downsample_block(x, n_filters):
        """下采样块"""
        f = double_conv_block(x, n_filters)
        p = layers.MaxPool2D(2)(f)
        p = layers.Dropout(DROPOUT_RATE)(p)  # 使用全局dropout率
        return f, p

    def upsample_block(x, conv_features, n_filters):
        """上采样块"""
        x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
        x = layers.concatenate([x, conv_features])
        x = layers.Dropout(DROPOUT_RATE)(x)  # ❌ 修正：使用 x 而不是 p
        x = double_conv_block(x, n_filters)
        return x

    inputs = layers.Input(shape=input_shape)

    # 编码器（下采样路径）
    f0, p0 = downsample_block(inputs, 32)
    f1, p1 = downsample_block(p0, 64)
    f2, p2 = downsample_block(p1, 128)
    f3, p3 = downsample_block(p2, 256)
    f4, p4 = downsample_block(p3, 512)

    # 瓶颈层
    bottleneck = double_conv_block(p4, 1024)

    # 解码器（上采样路径）
    u6 = upsample_block(bottleneck, f4, 512)
    u7 = upsample_block(u6, f3, 256)
    u8 = upsample_block(u7, f2, 128)
    u9 = upsample_block(u8, f1, 64)
    u10 = upsample_block(u9, f0, 32)

    # 输出层 - 确保输出为float32以匹配混合精度要求
    outputs = layers.Conv2D(1, 1, padding="same", dtype='float32')(u10)

    return models.Model(inputs, outputs, name="U-Net")

# =====================================================
# 训练
# =====================================================
def train_model(X, y):
    print("\n=== 开始训练模型（使用新U-Net架构）===")

    # 1. 优化器：固定学习率1e-4 + 梯度裁剪
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=1e-4,
        clipnorm=1.0  # 防止梯度爆炸
    )

    # 2. 构建模型
    model = build_unet(X.shape[1:])

    # 3. 编译模型
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # 4. 数据管道
    train_dataset = tf.data.Dataset.from_tensor_slices((X, y))
    train_dataset = train_dataset.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 5. 验证集（20%）
    val_size = int(0.2 * len(X))
    val_dataset = train_dataset.take(val_size)
    train_dataset = train_dataset.skip(val_size)

    # 6. 回调函数
    callbacks = [
        # 早停：8轮无改善就停止
        tf.keras.callbacks.EarlyStopping(
            patience=8,
            restore_best_weights=True,
            monitor='val_loss'
        ),
        # 学习率调整：4轮无改善就减半
        tf.keras.callbacks.ReduceLROnPlateau(
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            monitor='val_loss'
        ),
        # 保存最佳模型
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(OUTPUT_DIR, 'best_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        )
    ]

    # 7. 训练
    print(f"训练参数: batch={BATCH_SIZE}, epochs={EPOCHS}, LR=1e-4")
    print(f"数据形状: X={X.shape}, y={y.shape}")
    print(f"模型架构: 新U-Net（双卷积块、5层下采样/上采样）")

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 8. 清理
    tf.keras.backend.clear_session()
    return model

# =====================================================
# 订正并保存（修正版）
# =====================================================
def correct_and_save(model, scaler_x, scaler_y):
    print("\n=== 开始订正并保存结果 ===")

    for year in VAL_YEARS:
        cmip6_file = find_cmip6_file(year)
        if cmip6_file is None:
            print(f"❌ 未找到 {year} 年文件，跳过")
            continue

        ds = xr.open_dataset(cmip6_file)

        # 查找变量名
        var_name = None
        for var in ['ssrd_regridded', 'rsds', 'ssrd']:
            if var in ds:
                var_name = var
                break

        if var_name is None:
            print(f"❌ {year} 年找不到变量，可用: {list(ds.data_vars)}")
            continue

        # 加载数据
        X = ds[var_name].values[..., np.newaxis]
        X = np.nan_to_num(X)

        print(f"\n📊 {year}年处理:")
        print(f"   原始CMIP6范围: [{X.min():.3f}, {X.max():.3f}]")

        # 标准化输入（使用训练时的scaler）
        ns = X.shape[0]
        X_scaled = scaler_x.transform(X.reshape(ns, -1)).reshape(X.shape)
        print(f"   标准化后范围: [{X_scaled.min():.3f}, {X_scaled.max():.3f}]")

        # 预测
        dataset = tf.data.Dataset.from_tensor_slices(X_scaled).batch(8).prefetch(tf.data.AUTOTUNE)
        pred_scaled = model.predict(dataset, verbose=0).squeeze()
        print(f"   预测结果范围(标准化): [{pred_scaled.min():.3f}, {pred_scaled.max():.3f}]")

        # 反标准化
        pred = scaler_y.inverse_transform(pred_scaled.reshape(ns, -1)).reshape(pred_scaled.shape)
        print(f"   最终结果范围: [{pred.min():.3f}, {pred.max():.3f}]")

        # 保存
        out_dir = os.path.join(OUTPUT_DIR, f"rsds_{year}")
        os.makedirs(out_dir, exist_ok=True)

        xr.Dataset(
            {'rsds': (['time','lat','lon'], pred)},
            coords={'time': ds.time, 'lat': ds.lat, 'lon': ds.lon}
        ).to_netcdf(os.path.join(out_dir, f"rsds_{year}.nc"))

        print(f"✅ 保存 {year} 年订正结果: {out_dir}")

# =====================================================
# 验证指标
# =====================================================
def rmse(p, o):
    if hasattr(p, 'values'): p = p.values
    if hasattr(o, 'values'): o = o.values
    return np.sqrt(np.nanmean((p-o)**2))

def mae(p, o):
    if hasattr(p, 'values'): p = p.values
    if hasattr(o, 'values'): o = o.values
    return np.nanmean(np.abs(p-o))

def bias(p, o):
    if hasattr(p, 'values'): p = p.values
    if hasattr(o, 'values'): o = o.values
    return np.nanmean(p-o)

def corrcoef(p, o):
    if hasattr(p, 'values'): p = p.values
    if hasattr(o, 'values'): o = o.values
    p, o = p.flatten(), o.flatten()
    m = np.isfinite(p) & np.isfinite(o)
    if not np.any(m):
        return np.nan
    return np.corrcoef(p[m], o[m])[0,1]

# =====================================================
# 独立验证 - 输出到TXT
# =====================================================
def validate():
    print("\n=== 开始独立验证 ===")
    rows = []

    for year in VAL_YEARS:
        era5_file = find_era5_file(year)
        cmip6_file = find_cmip6_file(year)

        if era5_file is None or cmip6_file is None:
            continue

        corr_path = os.path.join(OUTPUT_DIR, f"rsds_{year}", f"rsds_{year}.nc")
        if not os.path.exists(corr_path):
            print(f"❌ {year} 年订正结果不存在")
            continue

        # 加载变量
        era5_ds = xr.open_dataset(era5_file)
        cmip_ds = xr.open_dataset(cmip6_file)
        corr_ds = xr.open_dataset(corr_path)

        # 查找变量名
        era5_var = next((v for v in ['ssrd_regridded', 'ssrd', 'SSRD'] if v in era5_ds), None)
        cmip_var = next((v for v in ['ssrd_regridded', 'rsds', 'ssrd'] if v in cmip_ds), None)

        if not (era5_var and cmip_var):
            continue

        era5 = era5_ds[era5_var]
        # ERA5单位转换 - 与训练部分保持一致
        if era5_var.lower() in ['ssrd', 'ssrd_regridded']:
            era5 = era5 / 86400.0

        cmip = cmip_ds[cmip_var]
        corr = corr_ds['rsds']

        rows.append({
            "year": year,
            "RMSE_raw": rmse(cmip, era5),
            "RMSE_corr": rmse(corr, era5),
            "MAE_raw": mae(cmip, era5),
            "MAE_corr": mae(corr, era5),
            "Bias_raw": bias(cmip, era5),
            "Bias_corr": bias(corr, era5),
            "R_raw": corrcoef(cmip, era5),
            "R_corr": corrcoef(corr, era5),
        })

    if rows:
        df = pd.DataFrame(rows)

        # 计算平均值
        df_mean = df.mean(numeric_only=True)

        # 生成验证结果文本
        validation_text = []
        validation_text.append("=" * 60)
        validation_text.append("CNN订正验证结果（新U-Net架构）")
        validation_text.append("=" * 60)
        validation_text.append(f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        validation_text.append(f"训练年份: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}")
        validation_text.append(f"验证年份: {VAL_YEARS[0]}-{VAL_YEARS[-1]}")
        validation_text.append(f"模型参数: Dropout={DROPOUT_RATE}, L2={L2_REG}, Filters={FILTERS}")
        validation_text.append(f"训练参数: Batch={BATCH_SIZE}, Epochs={EPOCHS}, LR={LEARNING_RATE}")
        validation_text.append("")

        validation_text.append("年度验证结果:")
        validation_text.append("-" * 60)
        validation_text.append(df.to_string(index=False, float_format='%.3f'))
        validation_text.append("")

        validation_text.append("平均验证结果:")
        validation_text.append("-" * 60)
        for metric, value in df_mean.items():
            validation_text.append(f"{metric:15s}: {value:.3f}")
        validation_text.append("")

        validation_text.append("性能改善分析:")
        validation_text.append("-" * 60)
        improvement = {}
        for metric in ['RMSE', 'MAE', 'Bias', 'R']:
            raw_col = f"{metric}_raw"
            corr_col = f"{metric}_corr"
            if raw_col in df.columns and corr_col in df.columns:
                raw_val = df_mean[raw_col]
                corr_val = df_mean[corr_col]
                if metric == 'R':
                    improvement_str = f"相关系数从 {raw_val:.3f} 提升到 {corr_val:.3f}"
                elif metric == 'Bias':
                    improvement_str = f"偏差从 {raw_val:.3f} 降低到 {corr_val:.3f}"
                else:
                    improvement_str = f"{metric}从 {raw_val:.3f} 降低到 {corr_val:.3f}"
                validation_text.append(improvement_str)

        # 保存到文件
        output_file = os.path.join(OUTPUT_DIR, "validation_results.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(validation_text))

        # 同时打印到控制台
        print('\n'.join(validation_text))
        print(f"\n✅ 验证结果已保存到: {output_file}")

    else:
        print("⚠️ 无有效数据用于验证")
        # 创建空的验证文件
        output_file = os.path.join(OUTPUT_DIR, "validation_results.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("无有效数据用于验证\n")
        print(f"⚠️ 已创建空验证文件: {output_file}")

# =====================================================
# 主程序
# =====================================================
def main():
    try:
        if gpus:
            tf.config.experimental.set_memory_growth(gpus[0], True)

        # 1. 加载数据
        X, y, lat, lon = load_training_data()

        # 2. 标准化（返回两个scaler）
        Xs, ys, scaler_x, scaler_y = scale_data(X, y)

        # 3. 训练模型（使用新U-Net架构）
        model = train_model(Xs, ys)

        # 4. 订正并保存（传入两个scaler）
        correct_and_save(model, scaler_x, scaler_y)

        # 5. 验证
        validate()

        print("\n✅ 订正 + 验证全部完成 (GPU加速)")
        print("\n💡 提示：如果数据仍然异常，请检查:")
        print("   1. ERA5和CMIP6的单位是否一致（都是W/m²）")
        print("   2. 是否需要移除标准化步骤")
        print("   3. 查看上面打印的数据范围信息，判断问题所在")
        print("\n🔧 新U-Net架构特点:")
        print("   - 双卷积块（Double Conv Block）")
        print("   - 5层下采样/上采样")
        print("   - 逐层滤波器：32→64→128→256→512→1024")
        print("   - Dropout层防止过拟合")

    except Exception as e:
        print(f"\n❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
