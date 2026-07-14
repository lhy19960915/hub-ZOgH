# 三方式对比结果（Function Call / MCP / CLI）

- LLM provider：`deepseek`
- 生成时间：本表由 `python compare.py` 实跑生成
- 问题数：4，方式数：4

## 对比表

| 问题 | 方式 | 工具调用 | 工具数 | LLM耗时 | 正确拒绝幻觉 | 答案摘要 |
|------|------|---------|:------:|:-------:|:------------:|---------|
| Transformer的自注意力机制原理是什么？ | Function Call | search_ai_knowledge, search_ai_knowledge | 2 | 14.1s | - | 以下是基于论文原文对 **Transformer 自注意力机制**原理的详细解释：  ---  ## 自注意力机制（Self-Attention）原理  ###... |
| Transformer的自注意力机制原理是什么？ | MCP | search_ai_knowledge, search_ai_knowledge | 2 | 18.0s | - | 好，根据检索到的论文原文，下面详细解释Transformer的自注意力机制原理。  ---  ## Transformer自注意力机制（Scaled Dot-P... |
| Transformer的自注意力机制原理是什么？ | CLI(named) | run_cli | 1 | 16.2s | - | 下面我结合检索到的内容，为你详细讲解 **Transformer 的自注意力机制（Self-Attention）原理**。  ---  ## 自注意力机制的核心... |
| Transformer的自注意力机制原理是什么？ | CLI(bash) | run_bash, run_bash, run_bash, run_bash | 4 | 67.5s | - | 好的，基于《动手学深度学习（PyTorch版）》中的内容，我来为你详细解答Transformer的自注意力机制原理。  ---  ## Transformer ... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | Function Call | search_ai_knowledge, get_weather, search_ai_knowledge | 3 | 18.9s | - | 好的，结合检索到的论文原文，下面是完整的回答：  ---  ## 一、Transformer 自注意力机制原理  根据《Attention Is All You... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | MCP | search_ai_knowledge, get_weather | 2 | 19.9s | - | 好的，以下分别回答您的两个问题。  ---  ## 一、Transformer的自注意力机制原理  根据《Attention Is All You Need》（... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | CLI(named) | run_cli, run_cli | 2 | 32.7s | - | 好的，以下是回答：  ---  ## 一、Transformer的自注意力机制原理  自注意力机制（Self-Attention，也称 intra-attent... |
| Transformer的自注意力机制原理是什么？另外北京的天气如何？ | CLI(bash) | run_bash, run_bash, run_bash, run_bash, run_bash | 5 | 44.6s | - | 好的，现在我已经获取了所有需要的信息，下面为您完整解答：  ---  ## 一、Transformer的自注意力机制原理  根据知识库中《动手学深度学习》(Py... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | Function Call | search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge | 4 | 15.6s | - | 根据检索到的资料，BERT和GPT-3的预训练方式有以下核心区别：  ---  ## BERT vs GPT-3 预训练方式对比  ### 1. 预训练任务不同... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | MCP | search_ai_knowledge, search_ai_knowledge, list_papers, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge, search_ai_knowledge | 7 | 20.6s | - | 现在我对两篇论文的预训练方式有了详细的原文依据，下面给你做完整对比。  ---  ## BERT vs GPT-3 预训练方式的核心差异  两者的预训练方式本质... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | CLI(named) | run_cli, run_cli | 2 | 22.7s | - | 根据检索到的资料，我来对比分析BERT和GPT-3的预训练方式的差异：  ---  ## BERT vs GPT-3 预训练方式对比  ### 1. **模型架... |
| 对比BERT和GPT-3的预训练方式有什么不同？ | CLI(bash) | run_bash, run_bash, run_bash, run_bash, run_bash, run_bash, run_bash, run_bash | 8 | 66.3s | - | 现在信息充足了，下面给出完整回答。  ---  ## BERT 与 GPT-3 预训练方式的对比  根据知识库中的资料，这两者的预训练方式有本质的区别，核心对比... |
| 区块链的共识算法是什么？ | Function Call | (无工具调用) | 0 | 4.0s | ✓ 拒绝 | 这个问题不在我的知识库覆盖范围内。我的知识库仅收录以下主题：  1. **Transformer架构**（论文：Attention Is All You Nee... |
| 区块链的共识算法是什么？ | MCP | (无工具调用) | 0 | 5.5s | ✓ 拒绝 | 您的问题是关于"区块链的共识算法"，这不在我的AI技术面试知识库收录范围内。  我的知识库目前只包含以下主题： - **Transformer架构**（Atte... |
| 区块链的共识算法是什么？ | CLI(named) | (无工具调用) | 0 | 3.4s | ✓ 拒绝 | 抱歉，关于"区块链的共识算法"这个主题，不在我的知识库范围内。我的知识库仅涵盖以下内容：  - **Transformer架构**（如自注意力机制、多头注意力等... |
| 区块链的共识算法是什么？ | CLI(bash) | (无工具调用) | 0 | 4.0s | ✓ 拒绝 | 我无法基于知识库回答区块链共识算法的问题，因为知识库仅包含以下 7 份论文/书籍：  1. **Transformer** – "Attention Is Al... |

## 解读

- **工具调用一致性**：四种方式对同一问题调用的工具与参数基本一致——说明底层能力相同，差异在『接入方式』而非『能力』。
- **接入成本**：Function Call 要手写 schema；MCP 要写 Server 但工具自动发现可跨产品复用；CLI(named) 写白名单；CLI(bash) 几乎零封装但需沙箱。
- **安全**：Function Call / MCP / CLI(named) 都走白名单，安全；CLI(bash) 依赖沙箱拦截，最危险。
- **跨模型复用**：MCP 工具可被任意支持 MCP 的 Host 复用；Function Call schema 各家 API 略有差异；CLI 与模型完全无关。
- **幻觉控制**：问区块链（不在知识库）时，看各方式是否正确拒绝而非编造数据。

## 各方式原始回答
