import React, { useState, useEffect, useRef } from 'react';
import { ArrowUp, ArrowDown, ArrowLeft, ArrowRight, RotateCcw, Copy, Check } from 'lucide-react';

export default function KeyboardCoordinateController() {
  const [x, setX] = useState(9);
  const [y, setY] = useState(1);
  const [history, setHistory] = useState(['A9B1']);
  const [copied, setCopied] = useState(false);
  const containerRef = useRef(null);

  // 生成当前坐标字符串
  const getCurrentCoordinate = (currentX, currentY) => `A${currentX}B${currentY}`;

  // 处理键盘事件
  const handleKeyPress = (event) => {
    event.preventDefault();
    let newX = x;
    let newY = y;
    let shouldUpdate = false;

    switch (event.key) {
      case 'ArrowLeft':
        newX = x - 1;
        shouldUpdate = true;
        break;
      case 'ArrowRight':
        newX = x + 1;
        shouldUpdate = true;
        break;
      case 'ArrowUp':
        newY = y + 1;
        shouldUpdate = true;
        break;
      case 'ArrowDown':
        newY = y - 1;
        shouldUpdate = true;
        break;
    }

    if (shouldUpdate) {
      setX(newX);
      setY(newY);
      const newCoordinate = getCurrentCoordinate(newX, newY);
      setHistory(prev => [...prev, newCoordinate]);
    }
  };

  // 重置功能
  const handleReset = () => {
    setX(9);
    setY(1);
    setHistory(['A9B1']);
    setCopied(false);
  };

  // 复制功能
  const handleCopy = () => {
    const pythonList = JSON.stringify(history);
    navigator.clipboard.writeText(pythonList).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  };

  // 监听键盘事件
  useEffect(() => {
    const container = containerRef.current;
    if (container) {
      container.addEventListener('keydown', handleKeyPress);
      container.focus();
      
      return () => {
        container.removeEventListener('keydown', handleKeyPress);
      };
    }
  }, [x, y]);

  return (
    <div 
      ref={containerRef}
      tabIndex={0}
      className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
      style={{ outline: 'none' }}
    >
      <div className="text-center mb-6">
        <h1 className="text-2xl font-bold text-gray-800 mb-2">键盘坐标控制器</h1>
        <p className="text-gray-600">使用方向键控制坐标，点击此区域后按键盘方向键</p>
      </div>

      {/* 当前坐标显示 */}
      <div className="text-center mb-6">
        <div className="inline-block bg-blue-100 px-6 py-3 rounded-lg">
          <span className="text-lg font-mono text-blue-800">当前位置: </span>
          <span className="text-2xl font-bold font-mono text-blue-900">
            {getCurrentCoordinate(x, y)}
          </span>
        </div>
      </div>

      {/* 控制说明 */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-2">键盘控制</h3>
          <div className="space-y-1 text-sm text-gray-600">
            <div className="flex items-center gap-2">
              <ArrowLeft size={16} /> 左键: A值减1 (x-1)
            </div>
            <div className="flex items-center gap-2">
              <ArrowRight size={16} /> 右键: A值加1 (x+1)
            </div>
            <div className="flex items-center gap-2">
              <ArrowUp size={16} /> 上键: B值加1 (y+1)
            </div>
            <div className="flex items-center gap-2">
              <ArrowDown size={16} /> 下键: B值减1 (y-1)
            </div>
          </div>
        </div>
        
        <div className="bg-gray-50 p-4 rounded-lg">
          <h3 className="font-semibold text-gray-700 mb-2">当前坐标</h3>
          <div className="text-sm text-gray-600 space-y-1">
            <div>A坐标: <span className="font-mono font-semibold">{x}</span></div>
            <div>B坐标: <span className="font-mono font-semibold">{y}</span></div>
            <div>总步数: <span className="font-semibold">{history.length}</span></div>
          </div>
        </div>
      </div>

      {/* 操作按钮 */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={handleReset}
          className="flex items-center gap-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
        >
          <RotateCcw size={16} />
          重置
        </button>
        
        <button
          onClick={handleCopy}
          className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-colors ${
            copied 
              ? 'bg-green-500 text-white' 
              : 'bg-blue-500 text-white hover:bg-blue-600'
          }`}
        >
          {copied ? <Check size={16} /> : <Copy size={16} />}
          {copied ? '已复制!' : '复制列表'}
        </button>
      </div>

      {/* 历史记录显示 */}
      <div className="bg-gray-50 p-4 rounded-lg">
        <h3 className="font-semibold text-gray-700 mb-3">
          轨迹历史 (Python列表格式)
        </h3>
        <div className="bg-white p-3 rounded border font-mono text-sm text-gray-800 max-h-40 overflow-y-auto">
          {JSON.stringify(history)}
        </div>
        
        {/* 历史步骤详细显示 */}
        <div className="mt-3">
          <details className="text-sm">
            <summary className="cursor-pointer text-gray-600 hover:text-gray-800">
              查看详细步骤 ({history.length} 步)
            </summary>
            <div className="mt-2 space-y-1 max-h-32 overflow-y-auto">
              {history.map((coord, index) => (
                <div key={index} className="flex items-center gap-2 text-gray-600">
                  <span className="w-8 text-right">{index + 1}.</span>
                  <span className="font-mono">{coord}</span>
                  {index === history.length - 1 && (
                    <span className="text-blue-500 text-xs">← 当前</span>
                  )}
                </div>
              ))}
            </div>
          </details>
        </div>
      </div>

      {/* 使用提示 */}
      <div className="mt-4 text-center text-sm text-gray-500">
        💡 点击此界面区域使其获得焦点，然后使用键盘方向键进行控制
      </div>
    </div>
  );
}