
import numpy as np
import cv2
from cv_bridge import CvBridge      # sudo apt-get install ros-melodic-cv-bridge
import os
import time
import pickle
import json


# print('filter:', filter_topics)
# for topic, msg, t in bag.read_messages(topics=filter_topics):
# for topic, msg, t in bag.read_messages(topics=['/imu/data_raw']):

def dump_msgs():
    import rosbag
    bag = rosbag.Bag('./ahg2library-new2.bag')

    print(bag)

    info = bag.get_type_and_topic_info() # tuple : (msy_types, topics)
    topics = info[1].keys()
    print(topics)

    filter_topics = [tp for tp in topics if ('velodyne' not in tp and 'camera' not in tp)]

    rgb_topic = '/camera/rgb/image_raw/compressed'
    depth_topic = '/camera/depth/image_raw/compressed'
    odom_topic = '/jackal_velocity_controller/odom'

    bridge = CvBridge()
    rgb_dict = {} # time -> msg
    depth_dict = {} # time -> msg
    odom_dict = {}

    LENGTH = 3000
    dump_dir = './msgs_dump'

    start_time = time.time()
    for topic, msg, t in bag.read_messages(topics=[rgb_topic]):
        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
        cv_image.astype(np.uint8)
        ts = str(t.to_sec())
        rgb_dict[ts] = cv_image
        if len(rgb_dict) % 100 == 0:
            print('RGB:', time.time() - start_time)
        if len(rgb_dict) >= LENGTH:
            break
    fn = dump_dir + '/rgb_dict' + str(LENGTH) + '.npy'
    with open(fn, 'wb') as f:
        np.save(f, rgb_dict)

    print('RGB:', len(rgb_dict))
    for k, v in rgb_dict.items():
        print(k, v.shape)
        break
    print('RGB:', time.time() - start_time)
    # assert 0

    start_time = time.time()
    for topic, msg, t in bag.read_messages(topics=[depth_topic]):
        cv_image = bridge.compressed_imgmsg_to_cv2(msg)
        cv_image.astype(np.uint8)
        # print(type(cv_image), cv_image.shape) # (720, 1280, 3)
        ts = str(t.to_sec())
        depth_dict[ts] = cv_image
        if len(depth_dict) % 100 == 0:
            print('Depth:', time.time() - start_time)
        if len(depth_dict) >= LENGTH:
            break
    print('Depth:', time.time() - start_time)
    fn = dump_dir + '/depth_dict' + str(LENGTH) + '.npy'
    with open(fn, 'wb') as f:
        np.save(f, depth_dict)


    for topic, msg, t in bag.read_messages(topics=[odom_topic]):
        pos = msg.pose.pose.position
        twist = msg.twist.twist
        t_str = "%.6f" % t.to_sec()
        odom_dict[t_str] = [pos.x, pos.y, pos.z, 
                                    twist.linear.x, twist.linear.y, twist.linear.z,
                                    twist.angular.x, twist.angular.y, twist.angular.z]
        if len(odom_dict) % 100 == 0:
            print("Odom:", len(odom_dict))
            print('Odom:', time.time() - start_time)
        if len(odom_dict) >= LENGTH:
            break
    fn = dump_dir + '/odom_dict' + str(LENGTH) + '.npy'
    print('Odom:', time.time() - start_time)
    with open(fn, 'wb') as f:
        np.save(f, odom_dict)


def align_obs_action(sample_rate=15):
    npy_dir = './msgs_dump'
    with open(npy_dir + '/rgb_dict3000.npy', 'rb') as f:
        rgb_dict = np.load(f, allow_pickle=True, encoding='bytes').item()
    print('RGB:', len(rgb_dict))
    with open(npy_dir + '/depth_dict3000.npy', 'rb') as f:
        depth_dict = np.load(f, allow_pickle=True, encoding='bytes').item()
    print('Depth:', len(depth_dict))
    with open(npy_dir + '/odom_dict3000.npy', 'rb') as f:
        odom_dict = np.load(f, allow_pickle=True, encoding='bytes').item()
    print('Odom:', len(odom_dict))
    # pass
    # sort the RGB and Depth dict by time
    rgb_keys = list(rgb_dict.keys())
    rgb_keys.sort()
    depth_keys = list(depth_dict.keys())
    depth_keys.sort()
    odom_keys = list(odom_dict.keys())
    odom_keys.sort()
    # print('RGB:', rgb_keys[0], rgb_keys[-1])
    # print('Depth:', depth_keys[0], depth_keys[-1])
    # print('Odom:', odom_keys[0], odom_keys[-1])

    rgbd_dict = {}
    rgbd_data = []
    odom_data = []
    for idx in range(0, len(rgb_keys), sample_rate):
        ts = rgb_keys[idx]
        rgb = rgb_dict[rgb_keys[idx]]
        depth = depth_dict[depth_keys[idx]]
        # extend the depth's dimension
        depth = np.expand_dims(depth, axis=2)
        rgbd = np.concatenate((rgb, depth), axis=2)
        rgbd_dict[rgb_keys[idx]] = rgbd

        # find the first odom after ts
        for i in range(len(odom_keys)):
            if float(odom_keys[i]) >= float(ts):
                odom_idx = i
                break
        if odom_idx == len(odom_keys):
            print('Odom not found for ts:', ts)
            assert 0
        odom_seq = []
        if odom_idx + 20 >= len(odom_keys):
            continue
        for k in range(odom_idx, odom_idx + 20):
            odom_seq.append(odom_dict[odom_keys[k]])
        odom_data.append(odom_seq)
        rgbd_data.append(rgbd)

        if idx % 100 == 0:
            print('RGBD:', idx)
            # break
    rgbd_data = np.array(rgbd_data)
    odom_data = np.array(odom_data)
    print('NP RGBD:', rgbd_data.shape)
    print('NP Odom:', odom_data.shape)
    dataset = {'rgbd': rgbd_data, 'odom': odom_data}
    # assert 0
    dataset_fn = npy_dir + '/dataset3000_sp' + str(sample_rate) + '.npz'
    with open(dataset_fn, 'wb') as f:
        np.savez_compressed(f, **dataset)

    # print('RGBD:', len(rgbd_dict))
    # sample the rgbd_dict with sample_rate
    # rgbd_keys = list(rgbd_dict.keys())
    # rgbd_keys.sort()
    # rgbd_dict_sampled = {}
    # for idx in range(0, len(rgbd_keys), sample_rate):
    #     rgbd_dict_sampled[rgbd_keys[idx]] = rgbd_dict[rgbd_keys[idx]]
    # print('RGBD Sampled:', len(rgbd_dict_sampled))
    # fn = npy_dir + '/rgbd_dict_sp' + str(sample_rate) + '.npy'
    # with open(fn, 'wb') as f:
    #     np.save(f, rgbd_dict)

import cv2
import os
def parse_npz():
    npy_dir = './msgs_dump'
    dataset_fn = npy_dir + '/dataset3000_sp1.npz'
    with open(dataset_fn, 'rb') as f:
        data = np.load(f)
        rgbd = data['rgbd']
        odom = data['odom']
        print('RGBD:', rgbd.shape)
        print('Odom:', odom.shape)
    rgbd_len = rgbd.shape[0]
    viz_dir = './viz'
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
    
    for idx in range(rgbd_len):
        rgb = rgbd[idx, :, :, :3]
        depth = rgbd[idx, :, :, 3]
        print(rgb.shape, depth.shape)
        cv2.imwrite(viz_dir+'/rgb' + str(idx) + '.png', rgb)
        cv2.imwrite(viz_dir+'/depth' + str(idx) + '.png', depth)

# dump_msgs()
# align_obs_action(sample_rate=1)
parse_npz()

   