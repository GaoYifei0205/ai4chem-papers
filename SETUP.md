# 部署到 GitHub Pages 步骤

## 第一步：创建 GitHub 仓库

1. 打开 https://github.com/new
2. Repository name 填：`ai4chem-papers`
3. 选择 **Public**（公开，这样任何人都能访问）
4. 不要勾选任何初始化选项，直接点 **Create repository**

## 第二步：上传代码

在终端执行（把 `YOUR_USERNAME` 换成你的 GitHub 用户名）：

```bash
cd ~/Desktop/ai4chem-papers
git init
git add .
git commit -m "init: AI for Chemistry paper tracker"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/ai4chem-papers.git
git push -u origin main
```

## 第三步：开启 GitHub Pages

1. 进入仓库页面 → **Settings** → **Pages**
2. Source 选择 **Deploy from a branch**
3. Branch 选 **main**，目录选 **/ (root)**
4. 点 **Save**

等约 1 分钟，你的网页地址就是：
```
https://YOUR_USERNAME.github.io/ai4chem-papers
```

## 第四步（可选）：配置 Semantic Scholar API Key

免费注册获取 API key：https://www.semanticscholar.org/product/api

1. 仓库页面 → **Settings** → **Secrets and variables** → **Actions**
2. 点 **New repository secret**
3. Name: `S2_API_KEY`，Value: 粘贴你的 key

没有 key 也能正常运行，只是不会显示引用数。

## 自动更新

代码推送后，GitHub Actions 会：
- **每天北京时间 10:00** 自动运行，抓取前一天的论文，更新网页
- 你也可以在 Actions 页面手动触发，指定任意日期

## 手动触发（测试用）

1. 仓库页面 → **Actions** → **Update AI4Chem Papers**
2. 点 **Run workflow**
3. 可以在 `target_date` 填日期（如 `2026-03-27`），留空则抓昨天
