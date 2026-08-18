### 1. 开发过程记录
- 每轮与 Kilo Code 的交互必须完整记录到 `records.jsonl` 文件
- 记录内容包括：本轮输入指令、代码修改 diff、对应 commit hash、修改时间
- 所有交互轮次不得遗漏、不得篡改

### 2. JSONL 格式要求
- 每轮交互生成一条 JSON 记录，每行一条，禁止合并
- 必填字段：
  - `round_id`：轮次序号（从1开始递增）
  - `prompt_content`：本轮输入给 Kilo Code 的完整自然语言指令
  - `modify_diff`：Kilo Code 本轮修改代码的完整 diff 内容
  - `commit_hash`：本轮修改对应的 Git commit 哈希值
  - `modify_time`：修改时间（格式：YYYY-MM-DD HH:MM:SS）
  - `agent_type`：固定为 "Kilo Code"
  - `dev_language`：本次开发使用的编程语言

### 3. Git 提交规范
- 每轮代码修改必须对应一次 Git commit
- commit 信息应清晰描述本轮修改内容
- 保留完整的提交历史，不得清空或删减
- JSONL 中的 `commit_hash` 必须与实际 Git 记录一一对应

### 4. 项目文档要求
- 必须维护 `README.md`，描述：
  - 选题说明
  - 提示词产生方法
  - JSONL 文件生成方法
  - 过程中遇到的问题和解决方法
