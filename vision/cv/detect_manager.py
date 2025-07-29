class DetectionManager:
    """目标检测管理器"""
    def __init__(self, vision_system: VisionGuidanceSystem):
        self.vision_system = vision_system
        self.target_detected = False
        self.detection_enabled = True
        
    async def check_for_targets(self) -> bool:
        """检查是否有目标"""
        if not self.detection_enabled:
            return False
            
        frame, command = self.vision_system.process_frame()
        if frame is not None:
            # 简单检测：如果有command输出说明检测到目标
            if command is not None and (
                abs(command.velocity_forward) > 0.001 or 
                abs(command.velocity_right) > 0.001 or 
                abs(command.velocity_down) > 0.001
            ):
                if not self.target_detected:
                    print("🎯 检测到目标！")
                    self.target_detected = True
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