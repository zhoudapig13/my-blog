---
title: "资源库设计：让文件和链接不再散落"
category: "资源整理"
tags:
  - "PDF"
  - "文件"
  - "知识库"
date: "2026-05-20"
pdf: "resources/example.pdf"
---

# 资源库设计：让文件和链接不再散落

资源页可以放 PDF、代码、数据集和网页链接。当前版本以 GitHub 仓库文件作为数据源，适合部署到 GitHub Pages。

## 下一步

如果要长期维护，建议把文章放进 `content/posts`，资源放进 `resources`，再让 GitHub Actions 自动生成站点数据。
