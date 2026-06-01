# 个人博客网站

这是一个轻量的浅色个人博客原型，直接使用 HTML/CSS/JavaScript 构建，不需要安装依赖。内容数据以 GitHub 仓库文件为准：

- 文章：`content/posts/*.md`
- 个人介绍：`data/profile.json`
- 资源：`data/resources.json`
- 计划：`data/plans.json`
- 站点读取数据：`data/site.json`

## 功能

- 首页左上角个人介绍与照片栏
- 首页左下角分类目录与标签
- 首页右侧近期文章展示
- 顶部导航：首页、博客、资源、计划、写作台和搜索栏
- Markdown 文章渲染
- 文章可附 PDF 链接
- 资源页可添加 PDF、数据集、代码和外部链接
- 计划页可记录目标与完成进度
- 写作台支持 Markdown 文件导入、实时预览和草稿导出
- `/admin/` 提供 GitHub CMS 可视化内容管理入口
- 3 套浅色样式方案可切换

## 使用

先生成站点数据：

```powershell
npm run build
```

再启动静态服务器：

```powershell
npm run dev
```

然后访问：

```text
http://localhost:4173
```

## GitHub 发布

1. 新建一个 GitHub 仓库，把本项目推送到 `main` 分支。
2. `admin/config.yml` 已配置为 `repo: zhoudapig13/my-blog`。
3. 在 GitHub 仓库设置中启用 Pages，来源选择 GitHub Actions。
4. 推送后，`.github/workflows/pages.yml` 会自动运行 `node scripts/build-site-data.js` 并发布网站。

## 可视化后台

`admin/` 使用 Decap CMS。正式部署到 GitHub Pages 后，后台会通过 GitHub 登录，把文章、资源、计划和个人介绍提交回仓库。

本地调试后台需要 Decap CMS 的本地后端服务；如果只是写作，可以先用站内写作台导入 Markdown、实时预览并下载草稿文件，再提交到 `content/posts/`。

## 资源路径

PDF 或其他文件可以放在 `resources/` 目录中，然后在文章或资源页中填写类似路径：

```text
resources/example.pdf
```

当前版本不使用浏览器 LocalStorage 保存博客内容。内容更新应进入 GitHub 仓库，再由构建脚本生成 `data/site.json`。
