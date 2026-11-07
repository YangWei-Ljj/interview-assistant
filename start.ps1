# 模拟面试个人知识库问答系统 · 一键启动（Streamlit 单体，默认本地 Ollama）
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# 优先使用已验证的 conda 环境，找不到再用系统 python
$py = "python"
if (-not (Test-Path $py)) { $py = "python" }
Write-Host "使用 Python：$py" -ForegroundColor DarkGray

function Test-Port([int]$p) {
    [bool](Get-NetTCPConnection -State Listen -LocalPort $p -ErrorAction SilentlyContinue)
}

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "已从 .env.example 生成 .env（默认本地 Ollama）" -ForegroundColor Yellow
}

Write-Host "初始化数据库、示例题库与知识图谱（首次较慢）..." -ForegroundColor Cyan
& $py scripts\init_db.py

if (Test-Port 8516) {
    Write-Host "端口 8516 已在监听，系统可能已启动，请直接访问 http://localhost:8516" -ForegroundColor Yellow
    Read-Host "按回车退出"
    exit 0
}

Write-Host "启动系统，浏览器将自动打开；如未打开请手动访问 http://localhost:8516" -ForegroundColor Green
# 不使用 headless：让 Streamlit 自动调用默认浏览器打开页面，新手双击即用
& $py -m streamlit run app\Home.py --server.port 8516
