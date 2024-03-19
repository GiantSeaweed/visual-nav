import numpy as np
import cv2

depth_fn = 'data-full/depth/depth_1635452215.65.png'
depth = cv2.imread(depth_fn, cv2.IMREAD_UNCHANGED)

im_x = 1280
im_y = 720
fov_x = 90
fov_y = 60
focal_x = im_x / (2 * np.tan(fov_x * np.pi / 360))
focal_y = im_y / (2 * np.tan(fov_y * np.pi / 360))
cu = im_x / 2
cv = im_y / 2

intrinsics = [[focal_x, 0, cu], 
              [0, focal_y, cv], 
              [0,     0, 1]]
intrinsics = np.array(intrinsics)
print(intrinsics)

depth_data = np.array(depth)
print(depth_data.shape)
# norm depth_data to 0-1
depth_data = depth_data / 255.0

# transform depth to 3d world coordinates
world_coords = np.zeros((im_y, im_x))
w_pos_list = []
for u in range(im_x):
    for v in range(im_y):
        p_pos = np.array([v, u, 1])
        depth_val = depth_data[v, u]
        if depth_val == 0:
            continue
        w_pos = np.dot(np.linalg.inv(intrinsics), p_pos) * depth_val
        w_pos_list.append(w_pos)

w_pos_list = np.array(w_pos_list)
print(w_pos_list.shape)

# plot 3d world coordinates without z-axis
import matplotlib
matplotlib.use('PDF')
import matplotlib.pyplot as plt
# fig = plt.figure()

# plt.scatter(w_pos_list[:, 0], w_pos_list[:, 1], s=0.1)
# plt.axis('equal')
# plt.savefig('world_coords-0-1.png')
plt.close()

plt.scatter(-w_pos_list[:, 0], 1-w_pos_list[:, 2], s=0.1)
plt.axis('equal')
plt.title('BEV of Depth Image')
plt.savefig('world_coords-0-2.png')
plt.close()

# plt.scatter(w_pos_list[:, 1], w_pos_list[:, 2], s=0.1)
# plt.axis('equal')
# plt.savefig('world_coords-1-2.png')

# visualize 3d world coordinates in bird's eye view
