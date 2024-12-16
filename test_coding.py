# Định nghĩa kích thước bản đồ và kích thước ban đầu của hình chữ nhật tìm kiếm
import plotly.graph_objects as go
map_size = 20
initial_rectangle_size = 3

# Khởi tạo danh sách điểm tìm kiếm
search_points = []

# Định nghĩa điểm trung tâm
center_point = (map_size // 2, map_size // 2)
search_points.append(center_point)

# Dần dần mở rộng hình chữ nhật tìm kiếm và ghi lại các điểm tìm kiếm
rectangle_size = initial_rectangle_size
while rectangle_size <= map_size:
    # Tính toán bốn điểm góc của hình chữ nhật tìm kiếm hiện tại
    top_left = (center_point[0] - rectangle_size // 2, center_point[1] - rectangle_size // 2)
    top_right = (center_point[0] + rectangle_size // 2, center_point[1] - rectangle_size // 2)
    bottom_right = (center_point[0] + rectangle_size // 2, center_point[1] + rectangle_size // 2)
    bottom_left = (center_point[0] - rectangle_size // 2, center_point[1] + rectangle_size // 2)
    
    # Thêm các điểm trên các cạnh của hình chữ nhật vào danh sách điểm tìm kiếm
    for x in range(top_left[0], bottom_right[0] + 1):
        search_points.append((x, top_left[1]))
        search_points.append((x, bottom_right[1]))
    for y in range(top_left[1], bottom_right[1] + 1):
        search_points.append((top_left[0], y))
        search_points.append((bottom_right[0], y))
    
    # Mở rộng kích thước hình chữ nhật tìm kiếm
    rectangle_size += 2

# In danh sách điểm tìm kiếm
x1 = []
y1 = []
for point in search_points: 
    x1.append(point[0])
    y1.append(point[1])
fig = go.Figure(data=[go.Scatter(x=x1, y=y1)])

fig.show()
