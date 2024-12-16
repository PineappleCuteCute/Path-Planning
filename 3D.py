import plotly.graph_objects as go

import pandas as pd

# Đọc dữ liệu từ tệp CSV
z_data1 = pd.read_csv('land.csv')
z_data2 = pd.read_csv('sea3.csv')
dates = pd.read_csv("agentpos3.csv")

# Tạo một đối tượng Surface cho dữ liệu đất
fig = go.Figure(data=[go.Surface(z=z_data1.values, colorscale="algae")])

# Thêm một đối tượng Surface cho dữ liệu biển
fig.add_trace(go.Surface(z=z_data2.values, colorscale='Blues', opacity=0.9))

# Cập nhật các thuộc tính của các đường viền trên trục Z
fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                  highlightcolor="limegreen", project_z=True))

# Đặt các giá trị và nhãn cho các dấu tick trên trục Z
z_tickvals = [-6, -4, -2, 0, 2, 4]
z_ticktext = [-6, -4, -2, 0, 2, 4]

# Cập nhật bố cục của biểu đồ
fig.update_layout(
    scene=dict(
        xaxis=dict(showgrid=True, gridcolor='dimgray', gridwidth=2),
        yaxis=dict(
            showgrid=True, 
            gridcolor='dimgray', 
            gridwidth=2,
            ticks='outside',
        ),
        zaxis=dict(showgrid=True, gridcolor='dimgray', gridwidth=2, ticktext=z_ticktext, tickvals=z_tickvals)
    ),
    title='Mô phỏng đại dương', 
    autosize=True,
    scene_camera_eye=dict(x=2, y=0, z=2),
    width=1500, 
    height=1000,
    margin=dict(l=65, r=50, b=65, t=90)
)

# Thêm dữ liệu 3D của các vị trí tác nhân
fig.add_trace(go.Scatter3d(
    x=dates.y, y=dates.x, z=dates.z,
    marker=dict(
        size=4,
        colorscale='Viridis',
    ),
    line=dict(
        color='darkblue',
        width=5
    )
))

# Hiển thị biểu đồ
fig.show()
