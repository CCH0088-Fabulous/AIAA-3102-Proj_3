# 关于多人合作提交 Git 的步骤
在服务器上配置 Git 身份 (解决"谁提交的"问题)

团队协作建议：如果是多人共用同一服务器账号，建议不设置全局配置，而是在各自的仓库目录下单独配置，避免提交记录混乱

```
git config user.name "Your Name"
git config user.email "your_email@example.com"
```
