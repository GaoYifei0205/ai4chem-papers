#!/bin/bash
# 一键抓取论文并推送到 GitHub Pages
# 用法:
#   ./run.sh            # 抓取昨天的论文
#   ./run.sh 2026-03-27 # 抓取指定日期的论文

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 加载 AWS 环境变量（从 Cursor settings 读取）
export AWS_REGION="us-west-2"
export AWS_ACCESS_KEY_ID="$(python3 -c "
import json
with open(\"$HOME/Library/Application Support/Cursor/User/settings.json\") as f:
    d = json.load(f)
for v in d.get('claudeCode.environmentVariables', []):
    if v['name'] == 'AWS_ACCESS_KEY_ID':
        print(v['value']); break
")"
export AWS_SECRET_ACCESS_KEY="$(python3 -c "
import json
with open(\"$HOME/Library/Application Support/Cursor/User/settings.json\") as f:
    d = json.load(f)
for v in d.get('claudeCode.environmentVariables', []):
    if v['name'] == 'AWS_SECRET_ACCESS_KEY':
        print(v['value']); break
")"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="arn:aws:bedrock:us-west-2:027950631154:application-inference-profile/cn502xxhr6xk"

# 安装依赖（如未安装）
python3 -m pip install -q -r requirements.txt --break-system-packages 2>/dev/null || true

# 设置目标日期
if [ -n "$1" ]; then
  export TARGET_DATE="$1"
  echo "📅 抓取日期: $TARGET_DATE"
else
  echo "📅 抓取昨天的论文"
fi

# 运行抓取脚本
python3 scripts/fetch_papers.py

# 推送到 GitHub
echo "🚀 推送到 GitHub..."
git add data/papers.json index.html
git diff --cached --quiet && echo "✅ 无新内容" && exit 0
git commit -m "update: papers $(date +'%Y-%m-%d %H:%M')"
git push
echo "✅ 网页已更新: https://gaoyifei0205.github.io/ai4chem-papers"
