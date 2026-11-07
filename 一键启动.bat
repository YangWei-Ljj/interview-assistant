@echo off
chcp 65001 >nul
title 模拟面试个人知识库问答系统
cd /d "%~dp0"
echo ============================================
echo   模拟面试个人知识库问答系统 正在启动...
echo   首次启动需要初始化示例题库，请耐心等待
echo ============================================
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1"
echo.
echo 系统已退出，按任意键关闭窗口
pause >nul
