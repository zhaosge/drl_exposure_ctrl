#!/home/scutracing/anaconda3/envs/stitch/bin/python
'''
Author: Shuyang Zhang
Date: 2024-06-06 21:20:32
LastEditors: ShuyangUni shuyang.zhang1995@gmail.com
LastEditTime: 2024-08-25 20:43:51
Description: ROS inference node for exposure control

Copyright (c) 2024 by Shuyang Zhang, All Rights Reserved. 
'''
import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from cv_bridge import CvBridge
import cv2
import numpy as np
from agent import Actor
import torch
import time
from collections import deque

class ExposureControlNode:
    def __init__(self):
        rospy.init_node('exposure_control_node', anonymous=True)
        
        self.params = {
            'device': torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
            'state_dim': (4, 84, 84),
            'action_dim': 1,
            'rwd_mode': "feat",
            'sac_hidden_dim': 512
        }
        
        # 初始化模型
        self.agent = Actor(self.params['state_dim'], self.params['action_dim'], self.params['sac_hidden_dim'])
        self.agent.load_state_dict(torch.load('/home/scutracing/25_ros_ws/src/drivers/camera_sua630/drl_exposure_ctrl/model/actor_drl_feat_10000.pth'))
        self.agent.eval()
        
        # 初始化状态缓冲区
        self.state_buffer = deque(maxlen=4)
        self.img_h, self.img_w = 84, 84
        
        # 当前曝光参数
        self.current_exposure = 100  # 初始曝光值
        self.expo_lb = 50
        self.expo_ub = 2000000
        
        # 快门速度和增益范围
        self.shutter_min = 25  # 最小快门速度 (us)
        self.shutter_max = 20000  # 最大快门速度 (us)
        self.gain_min = 4  # 最小增益
        self.gain_max = 36  # 最大增益
        
        # 性能统计
        self.inference_times = []
        
        # 频率控制 - 10Hz
        self.last_inference_time = 0
        self.inference_interval = 0.1  # 100ms间隔 (10Hz)
        self.enable_feature_debug = False  # 启用特征点调试显示
        
        # # ORB特征点检测器（用于84x84图像分析）
        self.orb = cv2.ORB_create(nfeatures=500)  # 最多检测500个特征点
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # 特征点统计
        self.feature_stats = {
            'keypoints_history': deque(maxlen=100),
            'matches_history': deque(maxlen=100),
            'last_keypoints': None,
            'last_descriptors': None,
            'last_84x84_image': None
        }
        
        # 调试选项
        
        # ROS组件
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('/perception/camera/mv_cam1', Image, self.image_callback)
        # 使用Vector3消息类型发布曝光参数：x=曝光值, y=快门速度(us), z=增益
        self.exposure_control_pub = rospy.Publisher('/perception/camera/drl_exposure_control', Vector3, queue_size=1)
        
        rospy.loginfo("Exposure control node initialized")
        rospy.loginfo("Publishing format: x=exposure_value, y=shutter_speed(us), z=gain")
        rospy.loginfo("Control frequency: 10Hz")
        rospy.loginfo(f"Shutter range: {self.shutter_min}-{self.shutter_max}us")
        rospy.loginfo(f"Gain range: {self.gain_min}-{self.gain_max}")

    def decompose_exposure(self, exposure_value):
        """
        将曝光值分解为快门速度和增益
        公式: e = t × 10^(g/20)
        """
        # 先尝试使用最小增益
        min_shutter = exposure_value / (10 ** (self.gain_min / 20.0))
        
        if self.shutter_min <= min_shutter <= self.shutter_max:
            # 快门速度在合理范围内，使用最小增益
            shutter_speed = int(np.clip(min_shutter, self.shutter_min, self.shutter_max))
            gain = self.gain_min
        
        elif min_shutter < self.shutter_min:
            # 目标曝光值太小，即使用最小快门速度也会过曝
            # 需要降低增益，但增益不能低于最小值
            shutter_speed = self.shutter_min
            required_gain = 20 * np.log10(exposure_value / shutter_speed)
            
            if required_gain < self.gain_min:
                # 即使用最小增益也会过曝，只能用最小增益并接受误差
                gain = self.gain_min
                rospy.logwarn(f"曝光值 {exposure_value} 太小，无法精确实现")
            else:
                gain = int(np.clip(required_gain, self.gain_min, self.gain_max))
        
        else:  # min_shutter > self.shutter_max
            # 需要更长的曝光时间，但快门速度已达上限
            # 需要提高增益来补偿
            shutter_speed = self.shutter_max
            required_gain = 20 * np.log10(exposure_value / shutter_speed)
            gain = int(np.clip(required_gain, self.gain_min, self.gain_max))
        
        return shutter_speed, gain
    
    def verify_exposure_decomposition(self, exposure_value, shutter_speed, gain):
        """验证分解结果是否正确"""
        calculated_exposure = shutter_speed * (10 ** (gain / 20.0))
        error_ratio = abs(calculated_exposure - exposure_value) / exposure_value
        return error_ratio < 0.1  # 允许10%的误差

    def detect_orb_features_84x84(self, processed_image_84x84):
        """
        对84x84的预处理图像进行ORB特征点检测
        这是agent实际"看到"的图像
        """
        # 将归一化的[0,1]图像转换回[0,255]用于ORB检测
        img_84x84_uint8 = (processed_image_84x84 * 255).astype(np.uint8)
        
        # ORB特征点检测
        keypoints, descriptors = self.orb.detectAndCompute(img_84x84_uint8, None)
        n_keypoints = len(keypoints)
        n_matches = 0
        
        # 计算与上一帧的匹配
        if (self.feature_stats['last_descriptors'] is not None and 
            descriptors is not None and len(descriptors) > 0):
            
            try:
                matches = self.bf.match(descriptors, self.feature_stats['last_descriptors'])
                
                if len(matches) > 4:
                    # 使用RANSAC过滤匹配点
                    pts1 = np.array([keypoints[m.queryIdx].pt for m in matches], dtype=np.float32)
                    pts0 = np.array([self.feature_stats['last_keypoints'][m.trainIdx].pt for m in matches], dtype=np.float32)
                    
                    if len(pts1) >= 4 and len(pts0) >= 4:
                        _, mask = cv2.findHomography(pts1, pts0, cv2.RANSAC, ransacReprojThreshold=3.0)
                        if mask is not None:
                            n_matches = int(np.sum(mask))
                        else:
                            n_matches = 0
                    else:
                        n_matches = len(matches)
                else:
                    n_matches = len(matches)
                        
            except Exception as e:
                rospy.logwarn(f"ORB feature matching error: {e}")
                n_matches = 0
        
        # 更新历史记录
        self.feature_stats['keypoints_history'].append(n_keypoints)
        self.feature_stats['matches_history'].append(n_matches)
        self.feature_stats['last_keypoints'] = keypoints
        self.feature_stats['last_descriptors'] = descriptors
        self.feature_stats['last_84x84_image'] = img_84x84_uint8.copy()
        
        # 计算特征点质量指标
        feature_density = n_keypoints / (84 * 84) * 1000  # 每1000像素的特征点数
        
        return n_keypoints, n_matches, feature_density, keypoints, img_84x84_uint8
        
    def preprocess_image(self, cv_image):
        """预处理图像：调整大小并归一化"""
        # 转换为灰度图像（如果是彩色）
        if len(cv_image.shape) == 3:
            gray_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = cv_image
            
        # 调整大小
        resized = cv2.resize(gray_image, (self.img_w, self.img_h))
        
        # 归一化到[0,1]
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
        
    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 预处理图像
            processed_image = self.preprocess_image(cv_image)
            
            # 对84x84图像进行ORB特征点检测
            orb_results = None
            if self.enable_feature_debug:
                orb_results = self.detect_orb_features_84x84(processed_image)
            
            # 添加到状态缓冲区
            self.state_buffer.append(processed_image)
            
            # 频率控制：检查是否应该进行推理（10Hz）
            current_time = time.time()
            if (len(self.state_buffer) == 4 and 
                current_time - self.last_inference_time >= self.inference_interval):
                self.perform_inference(orb_results)
                self.last_inference_time = current_time
                
            # 显示当前图像（可选，降低频率以减少计算负担）
            # if current_time - getattr(self, 'last_display_time', 0) >= 0.2:  # 5Hz显示
            #     cv2.imshow('Current Image', cv_image)
            #     cv2.imshow('Processed 84x84', processed_image)
                
            #     # 显示84x84图像的特征点
            #     if self.enable_feature_debug and orb_results is not None:
            #         n_keypoints, n_matches, feature_density, keypoints, img_84x84_uint8 = orb_results
                    
            #         # 绘制特征点
            #         feature_img = cv2.drawKeypoints(img_84x84_uint8, keypoints, None, 
            #                                        color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                    
            #         # 放大显示（从84x84放大到252x252）
            #         feature_img_large = cv2.resize(feature_img, (252, 252), interpolation=cv2.INTER_NEAREST)
                    
            #         # 添加文本信息
            #         cv2.putText(feature_img_large, f"Features: {n_keypoints}", (10, 25), 
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #         cv2.putText(feature_img_large, f"Matches: {n_matches}", (10, 50), 
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            #         cv2.putText(feature_img_large, f"Density: {feature_density:.1f}/1k", (10, 75), 
            #                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
            #         # 添加统计信息
            #         if len(self.feature_stats['keypoints_history']) > 1:
            #             avg_features = np.mean(list(self.feature_stats['keypoints_history'])[-10:])
            #             avg_matches = np.mean(list(self.feature_stats['matches_history'])[-10:])
            #             cv2.putText(feature_img_large, f"Avg F: {avg_features:.1f}", (10, 100), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            #             cv2.putText(feature_img_large, f"Avg M: {avg_matches:.1f}", (10, 120), 
            #                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                    
            #         cv2.imshow('ORB Features 84x84 (3x)', feature_img_large)
                
            #     cv2.waitKey(1)
            #     self.last_display_time = current_time
            
        except Exception as e:
            rospy.logerr(f"Error processing image: {e}")
            
    def perform_inference(self, orb_results=None):
        """使用当前状态进行推理并发布曝光控制命令"""
        try:
            # 准备输入状态
            state = np.stack(list(self.state_buffer), axis=0)  # (4, 84, 84)
            state_tensor = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), 0)  # (1, 4, 84, 84)
            
            # 开始计时
            t0 = time.time()
            
            # 推理
            with torch.no_grad():
                action, _ = self.agent(state_tensor, True, False)
                action_value = action.data.numpy().flatten()[0]
            
            # 结束计时
            t1 = time.time()
            inference_time = t1 - t0
            self.inference_times.append(inference_time)
            
            # 更新曝光值
            ev = action_value * 2
            new_exposure = self.current_exposure * np.power(2, ev)
            
            # 限制曝光范围
            new_exposure = np.clip(new_exposure, self.expo_lb, self.expo_ub)
            self.current_exposure = new_exposure
            
            # 分解曝光值为快门速度和增益
            shutter_speed, gain = self.decompose_exposure(new_exposure)
            
            # 验证分解结果
            is_valid = self.verify_exposure_decomposition(new_exposure, shutter_speed, gain)
            
            # 发布控制命令到单个话题
            exposure_control_msg = Vector3()
            exposure_control_msg.x = float(new_exposure)      # 曝光值
            exposure_control_msg.y = float(shutter_speed)     # 快门速度 (us)
            exposure_control_msg.z = float(gain)              # 增益
            self.exposure_control_pub.publish(exposure_control_msg)
            
            # 计算平均推理时间
            avg_inference_time = np.mean(self.inference_times[-100:])  # 最近100次的平均时间
            
            # 输出信息（包含ORB特征点信息）
            avg_intensity = np.mean(list(self.state_buffer)[-1])
            
            if self.enable_feature_debug and orb_results is not None:
                n_keypoints, n_matches, feature_density, _, _ = orb_results
                rospy.loginfo(f"[10Hz] Action: {action_value:.4f}, Exposure: {new_exposure:.1f}")
                rospy.loginfo(f"[10Hz] Shutter: {shutter_speed}us, Gain: {gain}, Valid: {is_valid}")
                rospy.loginfo(f"[10Hz] ORB 84x84: Features={n_keypoints}, Matches={n_matches}, Density={feature_density:.1f}/1k")
                rospy.loginfo(f"[10Hz] Avg Intensity: {avg_intensity:.4f}, Inference Time: {inference_time:.4f}s (avg: {avg_inference_time:.4f}s)")
            else:
                rospy.loginfo(f"[10Hz] Action: {action_value:.4f}, Exposure: {new_exposure:.1f}")
                rospy.loginfo(f"[10Hz] Shutter: {shutter_speed}us, Gain: {gain}, Valid: {is_valid}")
                rospy.loginfo(f"[10Hz] Avg Intensity: {avg_intensity:.4f}, Inference Time: {inference_time:.4f}s (avg: {avg_inference_time:.4f}s)")
            
        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")
            
    def run(self):
        """运行节点"""
        rospy.loginfo("Starting exposure control node at 10Hz...")
        try:
            rospy.spin()
        except KeyboardInterrupt:
            rospy.loginfo("Shutting down exposure control node")
        finally:
            cv2.destroyAllWindows()
            if self.inference_times:
                avg_time = np.mean(self.inference_times)
                total_inferences = len(self.inference_times)
                rospy.loginfo(f"Average inference time: {avg_time:.4f}s")
                rospy.loginfo(f"Total inferences: {total_inferences}")
                rospy.loginfo(f"Actual frequency: {total_inferences / (rospy.Time.now().to_sec() - rospy.Time.now().to_sec() + total_inferences * 0.1):.2f}Hz")

if __name__ == '__main__':
    try:
        node = ExposureControlNode()
        node.run()
    except rospy.ROSInterruptException:
        pass