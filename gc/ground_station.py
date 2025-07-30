#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
地面站程序
包含PyQt UI界面，用于与无人机通信，显示巡查数据和发送控制指令
"""

import sys
import socket
import json
import threading
import time
from datetime import datetime
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

class GridWidget(QWidget):
    """地图网格显示控件"""
    def __init__(self):
        super().__init__()
        self.grid_data = {}  # 存储每个格子的动物数据
        self.no_fly_zones = []  # 禁飞区列表
        self.setMinimumSize(450, 350)
        
    def add_no_fly_zone(self, grid_code):
        """添加禁飞区"""
        if grid_code not in self.no_fly_zones:
            self.no_fly_zones.append(grid_code)
            self.update()
    
    def remove_no_fly_zone(self, grid_code):
        """移除禁飞区"""
        if grid_code in self.no_fly_zones:
            self.no_fly_zones.remove(grid_code)
            self.update()
    
    def update_grid_data(self, grid_code, animal_type, count):
        """更新格子数据"""
        if grid_code not in self.grid_data:
            self.grid_data[grid_code] = {}
        self.grid_data[grid_code][animal_type] = count
        self.update()
    
    def paintEvent(self, event):
        """绘制网格"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 计算格子大小
        cell_width = self.width() / 9
        cell_height = self.height() / 7
        
        # 绘制网格
        painter.setPen(QPen(Qt.gray, 1, Qt.DashLine))
        for i in range(10):
            x = int(i * cell_width)
            painter.drawLine(x, 0, x, self.height())
        for j in range(8):
            y = int(j * cell_height)
            painter.drawLine(0, y, self.width(), y)
        
        # 绘制格子内容
        for i in range(9):
            for j in range(7):
                x = int(i * cell_width)
                y = int(j * cell_height)
                grid_code = f"A{i+1}B{j+1}"
                
                # 绘制禁飞区
                if grid_code in self.no_fly_zones:
                    painter.fillRect(x, y, int(cell_width), int(cell_height), QColor(180, 180, 180))
                
                # 绘制动物数据
                if grid_code in self.grid_data:
                    painter.setPen(QPen(Qt.black, 1))
                    text = ""
                    for animal, count in self.grid_data[grid_code].items():
                        text += f"{animal}:{count}\n"
                    painter.drawText(QRectF(x+2, y+2, cell_width-4, cell_height-4),
                                   Qt.AlignLeft | Qt.AlignTop, text.strip())
        
        # 绘制格子标签
        painter.setPen(QPen(Qt.black, 1))
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        
        # 列标签 A1-A9
        for i in range(9):
            x = i * cell_width + cell_width/2
            painter.drawText(QRectF(x-10, self.height()+2, 20, 15),
                           Qt.AlignCenter, f"A{i+1}")
        
        # 行标签 B1-B7
        for j in range(7):
            y = j * cell_height + cell_height/2
            painter.drawText(QRectF(-20, y-7, 18, 15),
                           Qt.AlignCenter, f"B{j+1}")

class StatisticsDialog(QDialog):
    """统计信息对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("统计信息")
        self.setModal(False)  # 非模态对话框
        self.resize(400, 300)
        
        layout = QVBoxLayout(self)
        
        # 统计表格
        self.stats_table = QTableWidget(5, 2)
        self.stats_table.setHorizontalHeaderLabels(["动物类型", "数量"])
        self.stats_table.verticalHeader().setVisible(False)
        self.stats_table.horizontalHeader().setStretchLastSection(True)
        
        animals = ['象', '虎', '狼', '猴', '孔雀']
        for i, animal in enumerate(animals):
            self.stats_table.setItem(i, 0, QTableWidgetItem(animal))
            self.stats_table.setItem(i, 1, QTableWidgetItem("0"))
            self.stats_table.item(i, 0).setFlags(Qt.ItemIsEnabled)  # 动物类型不可编辑
            self.stats_table.item(i, 1).setFlags(Qt.ItemIsEnabled)  # 数量不可编辑
        
        layout.addWidget(QLabel("各种动物检测数量统计："))
        layout.addWidget(self.stats_table)
        
        # 刷新和关闭按钮
        button_layout = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新")
        self.close_btn = QPushButton("关闭")
        self.close_btn.clicked.connect(self.close)
        button_layout.addWidget(self.refresh_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
    
    def update_statistics(self, total_animals):
        """更新统计数据"""
        animals = ['象', '虎', '狼', '猴', '孔雀']
        for i, animal in enumerate(animals):
            count = total_animals.get(animal, 0)
            self.stats_table.item(i, 1).setText(str(count))

class HistoryDialog(QDialog):
    """检测历史对话框"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("检测历史")
        self.setModal(False)  # 非模态对话框
        self.resize(600, 400)
        
        layout = QVBoxLayout(self)
        
        # 历史记录列表
        self.history_list = QListWidget()
        layout.addWidget(QLabel("动物检测历史记录："))
        layout.addWidget(self.history_list)
        
        # 按钮布局
        button_layout = QHBoxLayout()
        self.clear_btn = QPushButton("清空历史")
        self.export_btn = QPushButton("导出")
        self.close_btn = QPushButton("关闭")
        
        self.clear_btn.clicked.connect(self.clear_history)
        self.export_btn.clicked.connect(self.export_history)
        self.close_btn.clicked.connect(self.close)
        
        button_layout.addWidget(self.clear_btn)
        button_layout.addWidget(self.export_btn)
        button_layout.addWidget(self.close_btn)
        layout.addLayout(button_layout)
    
    def add_record(self, record):
        """添加历史记录"""
        self.history_list.addItem(record)
        # 自动滚动到最新记录
        self.history_list.scrollToBottom()
    
    def clear_history(self):
        """清空历史记录"""
        reply = QMessageBox.question(self, '确认', '确定要清空所有历史记录吗？',
                                   QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.history_list.clear()
    
    def export_history(self):
        """导出历史记录"""
        if self.history_list.count() == 0:
            QMessageBox.information(self, '提示', '没有历史记录可导出')
            return
            
        filename, _ = QFileDialog.getSaveFileName(self, '导出历史记录', 
                                                f'detection_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt',
                                                'Text Files (*.txt)')
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"野生动物检测历史记录\n")
                    f.write(f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for i in range(self.history_list.count()):
                        f.write(self.history_list.item(i).text() + "\n")
                
                QMessageBox.information(self, '成功', f'历史记录已导出到：\n{filename}')
            except Exception as e:
                QMessageBox.critical(self, '错误', f'导出失败：{e}')

class GroundStation(QMainWindow):
    # 定义信号
    data_received = pyqtSignal(dict)
    connection_status_changed = pyqtSignal(bool)
    
    def __init__(self):
        super().__init__()
        self.server_socket = None
        self.client_socket = None
        self.running = False
        self.detection_history = []  # 检测历史
        self.total_animals = {'象': 0, '虎': 0, '狼': 0, '猴': 0, '孔雀': 0}
        
        # 子窗口
        self.stats_dialog = None
        self.history_dialog = None
        
        self.init_ui()
        self.init_signals()
        
    def init_ui(self):
        """初始化UI界面"""
        self.setWindowTitle("野生动物巡查系统 - 地面站")
        self.setGeometry(100, 100, 720, 576)
        
        # 中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局
        main_layout = QHBoxLayout(central_widget)
        
        # 左侧布局
        left_layout = QVBoxLayout()
        
        # 地图显示
        map_group = QGroupBox("巡查区域地图")
        map_layout = QVBoxLayout(map_group)
        self.grid_widget = GridWidget()
        map_layout.addWidget(self.grid_widget)
        left_layout.addWidget(map_group)
        
        # 禁飞区设置
        no_fly_group = QGroupBox("禁飞区设置")
        no_fly_layout = QHBoxLayout(no_fly_group)
        self.no_fly_input = QLineEdit()
        self.no_fly_input.setPlaceholderText("输入格子代码，如A3B5")
        self.add_no_fly_btn = QPushButton("添加禁飞区")
        self.clear_no_fly_btn = QPushButton("清除禁飞区")
        no_fly_layout.addWidget(self.no_fly_input)
        no_fly_layout.addWidget(self.add_no_fly_btn)
        no_fly_layout.addWidget(self.clear_no_fly_btn)
        left_layout.addWidget(no_fly_group)
        
        # 控制按钮
        control_group = QGroupBox("控制指令")
        control_layout = QGridLayout(control_group)
        self.start_btn = QPushButton("开始巡查")
        self.stop_btn = QPushButton("紧急停止")
        self.return_btn = QPushButton("返航")
        self.connect_btn = QPushButton("启动服务器")
        control_layout.addWidget(self.start_btn, 0, 0)
        control_layout.addWidget(self.stop_btn, 0, 1)
        control_layout.addWidget(self.return_btn, 1, 0)
        control_layout.addWidget(self.connect_btn, 1, 1)
        left_layout.addWidget(control_group)
        
        main_layout.addLayout(left_layout, 2)
        
        # 右侧布局
        right_layout = QVBoxLayout()
        
        # 连接状态
        status_group = QGroupBox("连接状态")
        status_layout = QHBoxLayout(status_group)
        self.status_label = QLabel("未连接")
        self.status_label.setStyleSheet("color: red; font-weight: bold;")
        status_layout.addWidget(self.status_label)
        right_layout.addWidget(status_group)
        
        # 实时数据显示
        data_group = QGroupBox("实时检测数据")
        data_layout = QVBoxLayout(data_group)
        self.data_text = QTextEdit()
        self.data_text.setReadOnly(True)
        data_layout.addWidget(self.data_text)
        right_layout.addWidget(data_group)
        
        # 数据查看按钮
        view_group = QGroupBox("数据查看")
        view_layout = QHBoxLayout(view_group)
        self.stats_btn = QPushButton("查看统计信息")
        self.history_btn = QPushButton("查看检测历史")
        view_layout.addWidget(self.stats_btn)
        view_layout.addWidget(self.history_btn)
        right_layout.addWidget(view_group)
        
        main_layout.addLayout(right_layout, 1)
        
        # 状态栏
        self.statusBar().showMessage("准备就绪")
        
    def init_signals(self):
        """初始化信号连接"""
        self.add_no_fly_btn.clicked.connect(self.add_no_fly_zone)
        self.clear_no_fly_btn.clicked.connect(self.clear_no_fly_zones)
        self.start_btn.clicked.connect(self.send_start_patrol)
        self.stop_btn.clicked.connect(self.send_emergency_stop)
        self.return_btn.clicked.connect(self.send_return_home)
        self.connect_btn.clicked.connect(self.toggle_server)
        self.stats_btn.clicked.connect(self.show_statistics)
        self.history_btn.clicked.connect(self.show_history)
        
        self.data_received.connect(self.handle_received_data)
        self.connection_status_changed.connect(self.update_connection_status)
        
    def show_statistics(self):
        """显示统计信息窗口"""
        if self.stats_dialog is None:
            self.stats_dialog = StatisticsDialog(self)
        
        self.stats_dialog.update_statistics(self.total_animals)
        self.stats_dialog.show()
        self.stats_dialog.raise_()
        self.stats_dialog.activateWindow()
    
    def show_history(self):
        """显示检测历史窗口"""
        if self.history_dialog is None:
            self.history_dialog = HistoryDialog(self)
        
        self.history_dialog.show()
        self.history_dialog.raise_()
        self.history_dialog.activateWindow()
        
    def add_no_fly_zone(self):
        """添加禁飞区"""
        grid_code = self.no_fly_input.text().strip().upper()
        if self.validate_grid_code(grid_code):
            self.grid_widget.add_no_fly_zone(grid_code)
            self.no_fly_input.clear()
            self.send_no_fly_zones()
        else:
            QMessageBox.warning(self, "警告", "无效的格子代码！")
    
    def clear_no_fly_zones(self):
        """清除所有禁飞区"""
        self.grid_widget.no_fly_zones.clear()
        self.grid_widget.update()
        self.send_no_fly_zones()
    
    def validate_grid_code(self, code):
        """验证格子代码是否有效"""
        if len(code) != 4:
            return False
        if code[0] != 'A' or code[2] != 'B':
            return False
        try:
            col = int(code[1])
            row = int(code[3])
            return 1 <= col <= 9 and 1 <= row <= 7
        except:
            return False
    
    def toggle_server(self):
        """切换服务器状态"""
        if not self.running:
            self.start_server()
        else:
            self.stop_server()
    
    def start_server(self):
        """启动服务器"""
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind(('0.0.0.0', 8888))
            self.server_socket.listen(1)
            self.running = True
            self.connect_btn.setText("停止服务器")
            
            # 启动接受连接的线程
            accept_thread = threading.Thread(target=self.accept_connections)
            accept_thread.daemon = True
            accept_thread.start()
            
            self.statusBar().showMessage("服务器已启动，等待无人机连接...")
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"启动服务器失败: {e}")
    
    def stop_server(self):
        """停止服务器"""
        self.running = False
        if self.client_socket:
            self.client_socket.close()
        if self.server_socket:
            self.server_socket.close()
        self.connect_btn.setText("启动服务器")
        self.connection_status_changed.emit(False)
        self.statusBar().showMessage("服务器已停止")
    
    def accept_connections(self):
        """接受连接"""
        while self.running:
            try:
                self.server_socket.settimeout(1.0)
                client_socket, addr = self.server_socket.accept()
                self.client_socket = client_socket
                self.connection_status_changed.emit(True)
                print(f"无人机已连接: {addr}")
                
                # 启动接收线程
                recv_thread = threading.Thread(target=self.receive_data)
                recv_thread.daemon = True
                recv_thread.start()
                
            except socket.timeout:
                continue
            except Exception as e:
                if self.running:
                    print(f"接受连接错误: {e}")
    
    def receive_data(self):
        """接收数据"""
        buffer = ""
        while self.running and self.client_socket:
            try:
                data = self.client_socket.recv(1024).decode('utf-8')
                if not data:
                    break
                
                buffer += data
                
                while len(buffer) >= 8:
                    msg_len = int(buffer[:8])
                    
                    if len(buffer) >= 8 + msg_len:
                        json_data = buffer[8:8+msg_len]
                        buffer = buffer[8+msg_len:]
                        
                        try:
                            msg = json.loads(json_data)
                            self.data_received.emit(msg)
                        except json.JSONDecodeError as e:
                            print(f"JSON解析错误: {e}")
                    else:
                        break
                        
            except Exception as e:
                print(f"接收错误: {e}")
                break
        
        self.connection_status_changed.emit(False)
    
    def handle_received_data(self, data):
        """处理接收到的数据"""
        msg_type = data.get('type')
        timestamp = data.get('timestamp', '')
        
        if msg_type == 'animal_detection':
            # 处理动物检测数据
            grid_code = data.get('grid_code')
            animal_type = data.get('animal_type')
            count = data.get('count')
            
            # 更新地图显示
            self.grid_widget.update_grid_data(grid_code, animal_type, count)
            
            # 更新统计
            if animal_type in self.total_animals:
                self.total_animals[animal_type] += count
                # 如果统计窗口打开，更新显示
                if self.stats_dialog and self.stats_dialog.isVisible():
                    self.stats_dialog.update_statistics(self.total_animals)
            
            # 添加到历史记录
            record = f"{timestamp} - {grid_code}: {animal_type} x{count}"
            self.detection_history.append(record)
            # 如果历史窗口打开，添加记录
            if self.history_dialog:
                self.history_dialog.add_record(record)
            
            # 显示实时数据
            self.data_text.append(f"[{timestamp}] 检测到动物:")
            self.data_text.append(f"  位置: {grid_code}")
            self.data_text.append(f"  类型: {animal_type}")
            self.data_text.append(f"  数量: {count}")
            self.data_text.append("")
            
        elif msg_type == 'status':
            # 处理状态信息
            status_type = data.get('status_type')
            value = data.get('value')
            self.statusBar().showMessage(f"{status_type}: {value}")
            
        elif msg_type == 'mission_complete':
            # 处理任务完成信息
            total = data.get('total_animals', {})
            self.data_text.append(f"[{timestamp}] 任务完成！")
            self.data_text.append("统计结果:")
            for animal, count in total.items():
                self.data_text.append(f"  {animal}: {count}")
            self.data_text.append("")
            
            # 弹出任务完成提示
            QMessageBox.information(self, "任务完成", "巡查任务已完成！\n请查看统计信息获取详细结果。")
    
    def update_connection_status(self, connected):
        """更新连接状态"""
        if connected:
            self.status_label.setText("已连接")
            self.status_label.setStyleSheet("color: green; font-weight: bold;")
            self.statusBar().showMessage("无人机已连接")
        else:
            self.status_label.setText("未连接")
            self.status_label.setStyleSheet("color: red; font-weight: bold;")
            self.statusBar().showMessage("无人机断开连接")
    
    def send_command(self, command):
        """发送指令到无人机"""
        if self.client_socket:
            try:
                json_data = json.dumps(command, ensure_ascii=False)
                msg = f"{len(json_data):08d}{json_data}"
                self.client_socket.sendall(msg.encode('utf-8'))
                self.data_text.append(f"[发送指令] {command['type']}")
            except Exception as e:
                QMessageBox.warning(self, "警告", f"发送指令失败: {e}")
        else:
            QMessageBox.warning(self, "警告", "无人机未连接！")
    
    def send_no_fly_zones(self):
        """发送禁飞区设置"""
        command = {
            'type': 'set_no_fly_zone',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'zones': self.grid_widget.no_fly_zones
        }
        self.send_command(command)
    
    def send_start_patrol(self):
        """发送开始巡查指令"""
        command = {
            'type': 'start_patrol',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.send_command(command)
    
    def send_emergency_stop(self):
        """发送紧急停止指令"""
        command = {
            'type': 'emergency_stop',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.send_command(command)
    
    def send_return_home(self):
        """发送返航指令"""
        command = {
            'type': 'return_home',
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.send_command(command)
    
    def closeEvent(self, event):
        """关闭事件"""
        self.stop_server()
        
        # 关闭子窗口
        if self.stats_dialog:
            self.stats_dialog.close()
        if self.history_dialog:
            self.history_dialog.close()
            
        event.accept()

def main():
    app = QApplication(sys.argv)
    station = GroundStation()
    station.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()