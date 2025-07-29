class DetectionManager:
    """目标检测管理器"""
    def __init__(self, vision_system: VisionGuidanceSystem):
        self.vision_system = vision_system
        self.target_detected = False
        self.detection_enabled = True
        self.detected_positions = []  # 记录已检测过的目标位置
        self.position_tolerance = 1.0  # 位置容差（米）
        
    async def check_for_targets(self, current_position=None) -> bool:
        """检查是否有目标"""
        if not self.detection_enabled:
            return False
        
        # 如果提供了位置信息，检查是否在已检测位置附近
        if current_position is not None:
            if self._is_position_already_detected(current_position):
                return False
            
        frame, command = self.vision_system.process_frame()
        if frame is not None:
            # 直接调用图像识别函数, 更加鲁棒
            detections = self.vision_system.detector.detect_objects(frame)
            has_detections = len(detections) > 0
            
            if has_detections:
                if not self.target_detected:
                    print("🎯 检测到目标！")
                    self.target_detected = True
                    # 记录检测到目标的位置
                    if current_position is not None:
                        self._add_detected_position(current_position)
                        print(f"📍 记录目标位置: ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f})")
                return True
        
        if self.target_detected:
            print("目标丢失，继续飞行")
            self.target_detected = False
        return False
    
    def enable_detection(self):
        """启用目标检测"""
        self.detection_enabled = True
    
    def disable_detection(self):
        """禁用目标检测"""
        self.detection_enabled = False
    
    def _is_position_already_detected(self, current_position) -> bool:
        """检查当前位置是否已经检测过目标"""
        for detected_pos in self.detected_positions:
            # 和所有的已知点进行比较, 判断距离
            distance = ((current_position[0] - detected_pos[0]) ** 2 + 
                       (current_position[1] - detected_pos[1]) ** 2 + 
                       (current_position[2] - detected_pos[2]) ** 2) ** 0.5
            if distance < self.position_tolerance:
                print(f"⚠️ 位置 ({current_position[0]:.1f}, {current_position[1]:.1f}, {current_position[2]:.1f}) 已检测过目标，跳过")
                return True
        return False
    
    def _add_detected_position(self, position):
        """添加已检测位置到记录中"""
        self.detected_positions.append((position[0], position[1], position[2]))
    
    def clear_detected_positions(self):
        """清空已检测位置记录（可用于新任务开始时）"""
        self.detected_positions.clear()
        print("🗑️ 已清空检测位置记录")