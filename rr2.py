import streamlit as st
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import minimize

# --- 页面设置 ---
st.set_page_config(page_title="岭回归 3D 几何解释 (优化版)", layout="wide")

st.title("岭回归 (Ridge Regression) 几何解释 - 3D 交互优化版")

# --- 侧边栏参数 ---
st.sidebar.header("参数控制")
correlation = st.sidebar.slider("特征相关性 (改变碗的形状)", -0.95, 0.95, 0.80, 0.05)
radius = st.sidebar.slider("L2 约束半径 (t)", 0.1, 3.5, 1.0, 0.1)

# --- 数学计算部分 ---

# 1. 定义 RSS 曲面矩阵 (模拟数据)
def get_rss_params(corr):
    # OLS 最优解中心
    beta_ols = np.array([2.5, 2.0]) 
    # 协方差矩阵 A
    A = np.array([[1.0, corr], [corr, 1.0]])
    # 稍微放大一点矩阵特征值，让碗稍微陡峭一点，视觉效果更好
    A = A * 1.5
    return A, beta_ols

A, beta_ols = get_rss_params(correlation)

def rss_func(beta):
    diff = beta - beta_ols
    return diff.T @ A @ diff

# 2. 计算岭回归解 (约束优化)
cons = ({'type': 'ineq', 'fun': lambda x: radius**2 - (x[0]**2 + x[1]**2)})
res = minimize(rss_func, [0, 0], constraints=cons)
ridge_beta = res.x
ridge_rss = rss_func(ridge_beta)

# --- 绘图数据生成 ---

# 生成网格
limit = 5.0
x_range = np.linspace(-limit, limit, 80) # 降低一点分辨率提高性能
y_range = np.linspace(-limit, limit, 80)
X_grid, Y_grid = np.meshgrid(x_range, y_range)
Z_rss = np.zeros_like(X_grid)

# 填充 Z 值
for i in range(X_grid.shape[0]):
    for j in range(X_grid.shape[1]):
        Z_rss[i, j] = rss_func(np.array([X_grid[i, j], Y_grid[i, j]]))

# 为了视觉效果，我们定义一个“地板高度” (Z_floor)
# 所有的投影和底部的圆圈都画在这个高度，避免和碗底 (Z=0) 重叠
max_z = np.max(Z_rss)
z_floor = -0.2 * max_z # 地板设在 Z 轴负方向

# 生成圆形约束的数据 (用于画线)
theta = np.linspace(0, 2*np.pi, 100)
cyl_x = radius * np.cos(theta)
cyl_y = radius * np.sin(theta)
cyl_z_floor = np.full_like(theta, z_floor) # 地板上的圆
cyl_z_ridge = np.full_like(theta, ridge_rss) # 切点高度的圆

# --- Plotly 绘图 ---
fig = go.Figure()

# 1. RSS 曲面 (带底部投影)
fig.add_trace(go.Surface(
    z=Z_rss, x=X_grid, y=Y_grid,
    colorscale='Viridis', 
    opacity=0.6, # 半透明，方便看里面的结构
    name='RSS 曲面',
    contours_z=dict(
        show=True, 
        usecolormap=True, 
        highlightcolor="limegreen", 
        project=dict(z=True) # 【关键修改】开启 Z 轴投影，这会自动画到底部平面
    ),
    showscale=False
))

# 2. 绘制底部的 L2 约束圆圈 (投影层)
# 我们需要手动把这个圆圈画在 Plotly 自动生成的投影平面上
# Plotly 的 project=True 通常会投影到 Z 轴显示的最小值处
fig.add_trace(go.Scatter3d(
    x=cyl_x, y=cyl_y, z=cyl_z_floor, # 画在地板高度
    mode='lines',
    line=dict(color='red', width=4),
    name='L2 约束范围 (投影)'
))

# 3. 绘制岭回归截面圆环 (实际切点高度)
fig.add_trace(go.Scatter3d(
    x=cyl_x, y=cyl_y, z=cyl_z_ridge,
    mode='lines',
    line=dict(color='red', width=3, dash='dash'),
    name='切平面圆环'
))

# 4. 绘制圆柱体的“墙” (增加立体感)
# 用 Mesh3d 画一个淡淡的圆柱体
# 构造圆柱体网格点
z_mesh = np.linspace(z_floor, max_z, 2)
theta_mesh = np.linspace(0, 2*np.pi, 40)
z_grid, theta_grid = np.meshgrid(z_mesh, theta_mesh)
x_cyl_mesh = radius * np.cos(theta_grid)
y_cyl_mesh = radius * np.sin(theta_grid)

fig.add_trace(go.Surface(
    x=x_cyl_mesh, y=y_cyl_mesh, z=z_grid,
    colorscale=[[0, 'red'], [1, 'red']],
    opacity=0.1, # 非常透明
    showscale=False,
    name='约束墙'
))

# 5. 关键点 (缩小了点的大小)
# OLS 解
fig.add_trace(go.Scatter3d(
    x=[beta_ols[0]], y=[beta_ols[1]], z=[0],
    mode='markers',
    marker=dict(size=4, color='black'), # 【修改】size 改小到 4
    name='OLS 解 (碗底)'
))

# Ridge 解
fig.add_trace(go.Scatter3d(
    x=[ridge_beta[0]], y=[ridge_beta[1]], z=[ridge_rss],
    mode='markers',
    marker=dict(size=4, color='red'), # 【修改】size 改小到 4
    name='Ridge 解 (切点)'
))

# 6. 辅助线 (原点 -> Ridge -> OLS)
fig.add_trace(go.Scatter3d(
    x=[0, ridge_beta[0], beta_ols[0]],
    y=[0, ridge_beta[1], beta_ols[1]],
    z=[z_floor, ridge_rss, 0], # 起点设在地板上，更有空间感
    mode='lines',
    line=dict(color='black', width=3),
    name='收缩路径'
))

# 中心垂直线 (Z轴指示)
fig.add_trace(go.Scatter3d(
    x=[0, 0], y=[0, 0], z=[z_floor, max_z],
    mode='lines',
    line=dict(color='gray', width=1, dash='dot'),
    hoverinfo='skip'
))

# --- 布局设置 ---
fig.update_layout(
    scene=dict(
        xaxis_title='Beta 1',
        yaxis_title='Beta 2',
        zaxis_title='RSS',
        # 强制设置 Z 轴范围，包含“地板”
        zaxis=dict(range=[z_floor, max_z]),
        camera=dict(
            eye=dict(x=1.2, y=1.2, z=0.8)
        )
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=700,
    # 将图例放在左上角，不遮挡
    legend=dict(x=0, y=1, bgcolor='rgba(255,255,255,0.5)')
)

st.plotly_chart(fig, use_container_width=True)

# --- 说明 ---
st.info(f"""
**图解说明：**
1.  **彩色的底部投影**：这是 RSS 的等高线。请注意观察，它是**椭圆**。
    *   当你把上方 *特征相关性* 拉高时，这个底部的投影会变得非常扁平。
2.  **红色的底部圆圈**：这是岭回归的约束条件 ($\\|\\beta\\|_2 \\le t$) 在平面的投影。
3.  **立体结构**：
    *   半透明的红色圆柱体代表约束范围的“墙”。
    *   **红点 (Ridge解)** 就在这堵墙上，它是 RSS 曲面与墙体接触的最低点（切点）。
""")