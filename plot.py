import plotly.express as px
import pandas as pd

# Đọc dữ liệu từ tệp CSV
df1 = pd.read_csv('mean_reward_out.csv')  # Tự tay thay đổi mean_reward_out.csv thành steps_out.csv

# Vẽ đồ thị đường cho MeanReward
fig = px.line(df1 ,title='MeanReward')
fig.update_xaxes(
    type='linear',
    side='bottom',
    showgrid=False,
    linecolor='black',
    linewidth=3,
    gridwidth=2,
    title={'font': {'size': 18}, 'text': 'Episode', 'standoff': 10},
    automargin=True,
)

fig.update_yaxes(
    showline=True,
    linecolor='black',
    linewidth=3,
    gridwidth=2,
    title={'font': {'size': 18}, 'text': 'MeanReward', 'standoff': 10},
    automargin=True,
)

fig.show()
