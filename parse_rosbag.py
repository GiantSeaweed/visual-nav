import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge      # sudo apt-get install ros-melodic-cv-bridge
import os

bag = rosbag.Bag('./ahg2library.bag')

print(bag)

info = bag.get_type_and_topic_info() # tuple : (msy_types, topics)
topics = info[1].keys()

# topic_list = [

# ]
img = None
rgb_topic = '/camera/rgb/image_raw/compressed'
depth_topic = '/camera/depth/image_raw/compressed'
# vis_topic = '/visualization'
ROOT_DIR = './data'
if not os.path.exists(ROOT_DIR):
    os.mkdir(ROOT_DIR)
    os.mkdir(ROOT_DIR + '/color')
    os.mkdir(ROOT_DIR + '/depth')

for i in range(2):
    if (i == 0):
        TOPIC = depth_topic
        DESCRIPTION = 'depth_'
    else:
        TOPIC = rgb_topic
        DESCRIPTION = 'color_'
    # image_topic = bag.read_messages(TOPIC)
    for topic, msg, t in bag.read_messages(topics=[TOPIC]):
        # print(topic, t, type(msg), len(msg.data))
        # print(t.to_sec())
        # assert 0
        bridge = CvBridge()
        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
        cv_image.astype(np.uint8)
        if (DESCRIPTION == 'depth_'):
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
            cv2.imwrite(ROOT_DIR + '/depth/' + DESCRIPTION + str(t.to_sec()) + '.png', cv_image)
        else:
            cv2.imwrite(ROOT_DIR + '/color/' + DESCRIPTION + str(t.to_sec()) + '.png', cv_image)
        print('saved: ' + DESCRIPTION + str(t.to_sec()) + '.png')

    print('PROCESS %s COMPLETE' % DESCRIPTION)
    # break
assert 0
for topic, msg, t in bag.read_messages(topics=[depth_topic]):
    # print(topic, t, type(msg), len(msg.data))
    # img = msg.data
    # bag.close()
    # break
    bridge = CvBridge()
    cv_image = bridge.compressed_imgmsg_to_cv2(msg)
    cv_image.astype(np.uint8)
    print(cv_image.shape)
    print(cv_image)
    # dump cv_image to file
    import json
    with open('depth.json', 'w') as f:
        json.dump(cv_image.tolist(), f)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(cv_image, alpha=0.03), cv2.COLORMAP_JET)
    # cv2.imwrite(ROOT_DIR + '/depth/' + DESCRIPTION + str(b.timestamp) + '.png', cv_image)
    cv2.imwrite('dep.png', cv_image)
    break

# plot img to be image
img = np.array(img)
import matplotlib.pyplot as plt
plt.savefig('img.png', img)

# TypesAndTopicsTuple(msg_types={'nmea_msgs/Sentence': '9f221efc5f4b3bac7ce4af102b32308b', 'amrl_msgs/Localization2DMsg': '4ce450daa8564e2fb59b919aebbbe26e', 'sensor_msgs/TimeReference': 'fded64a0265108ba86c3d38fb11c0c16', 'sensor_msgs/LaserScan': '90c7ef2dc6895d81024acba2ac42f369', 'sensor_msgs/NavSatFix': '2d3a8cd499b9b4a0249fb98fd05cfa48', 'sensor_msgs/Imu': '6a62c6daae103f4ff57a132d6f95cec2', 'sensor_msgs/CompressedImage': '8f7a12909da2c9d3332d540a0977563f', 'velodyne_msgs/VelodyneScan': '50804fc9533a0e579e6322c04ae70566', 'jackal_msgs/Status': 'c851ebcf9a6e20b196bc7894e285b4f6', 'nav_msgs/Odometry': 'cd5e73d190d741a2f92e81eda573aca7', 'geometry_msgs/TwistStamped': '98d34b0043a2093cf9d9345ab6eef12e', 'amrl_msgs/VisualizationMsg': '989408817adef9cc3922d1967821d49a', 'sensor_msgs/Joy': '5a9ea5f83505693b71e785041e67a8bb'}, 
#                     topics={'/bluetooth_teleop/joy': TopicTuple(msg_type='sensor_msgs/Joy', message_count=17193, connections=1, frequency=100.53220200858081), '/velodyne_2dscan': TopicTuple(msg_type='sensor_msgs/LaserScan', message_count=3478, connections=1, frequency=9.91441693791056), '/localization': TopicTuple(msg_type='amrl_msgs/Localization2DMsg', message_count=8747, connections=1, frequency=24.856386840226026), '/navsat/nmea_sentence': TopicTuple(msg_type='nmea_msgs/Sentence', message_count=15418, connections=1, frequency=166.34821924327755), '/status': TopicTuple(msg_type='jackal_msgs/Status', message_count=351, connections=1, frequency=0.998254133892258), '/navsat/time_reference': TopicTuple(msg_type='sensor_msgs/TimeReference', message_count=3508, connections=1, frequency=9.820219194631814), '/imu/data_raw': TopicTuple(msg_type='sensor_msgs/Imu', message_count=24717, connections=1, frequency=83.0275748757844), '/navsat/fix': TopicTuple(msg_type='sensor_msgs/NavSatFix', message_count=3508, connections=1, frequency=9.820357150382929), '/jackal_velocity_controller/odom': TopicTuple(msg_type='nav_msgs/Odometry', message_count=17532, connections=1, frequency=82.7246262474853), '/visualization': TopicTuple(msg_type='amrl_msgs/VisualizationMsg', message_count=35918, connections=2, frequency=28728.109589041094), '/navsat/vel': TopicTuple(msg_type='geometry_msgs/TwistStamped', message_count=6238, connections=1, frequency=11.88248692567893), '/camera/depth/image_raw/compressed': TopicTuple(msg_type='sensor_msgs/CompressedImage', message_count=10519, connections=1, frequency=30.627094324467404), '/velodyne_2dscan_high_beams': TopicTuple(msg_type='sensor_msgs/LaserScan', message_count=3478, connections=1, frequency=9.914440373478312), '/camera/rgb/image_raw/compressed': TopicTuple(msg_type='sensor_msgs/CompressedImage', message_count=10519, connections=1, frequency=30.604188252462606), '/velodyne_packets': TopicTuple(msg_type='velodyne_msgs/VelodyneScan', message_count=3478, connections=1, frequency=9.914838795082181)})