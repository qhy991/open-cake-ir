# ADR 0052：独立代理会话也可以审查编译器发布

[设计目录](README.md) · [英文原文与完整证据](../../adr/0052-independent-agent-release-review.md)

**原文状态：仓库拥有者政策已接受，2026-09-06；实现本身仍需独立审查。**

本决定取代 0030 的“只能人类”解释，保留作者不能自批的边界。人类或新的独立代理会话可审查；允许的精确模型名只由 `src/open_cake_ir/compiler/release.py` 的 `ALLOWED_REVIEW_MODELS` 负责，不在翻译里复制另一份名单。

流程是：作者准备隔离后继、完整 Gate、固定审查提交和作者会话身份；另开明确选择允许模型的审查会话，核对实际运行元数据；审查改动、合同、回归和完整 Gate，并独立测试相关公开边界。审查记录和临时结果放在仓库外。

审查者返回 approved、request-changes 或 rejected。只有 approved 才能由该审查者写精确 Gate 的 `compiler/release-approval.json`；不能修改实现、预期或发布 lock。源码或 Gate 变化需要新审查。作者再运行既有发布流程，失败退出 3 并保留先前 lock 与批准。

新格式 schema_version=2。代理 reviewer 写 kind、实际 model、session_id、author_session_id，身份非空无多余空格，忽略大小写也必须不同；人类写 kind 与 name。还需明确批准、非空依据和精确 Gate 绑定。原文 JSON 是格式例子，不能当批准使用。

填写字段不证明真实身份，要核对启动／会话记录；模型不可用或未验证时不能静默换模型，作者也不能假扮人类绕过规则。无需读凭据。旧 schema-v1 用原固定实现回放，新发布不能借旧格式跳过检查。

验收覆盖所有允许模型、错误／缺失模型、同会话、畸形 reviewer、人类、旧格式、否决、过期 Gate、失败保护和不变发布无操作，完整 Corpus Gate 仍必需。
