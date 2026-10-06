# Related work (checked 6 Oct 2026)

AdviceLine is **not** the first work on AI and financial advice. The open gap is narrower: an **open, UK-specific benchmark** of how far **general-purpose chatbots** move along the guidance → targeted support → personal recommendation scale, including **multi-turn drift**, with a **guardrail tested by A/B experiment**.

| Work | What it does | How AdviceLine differs |
|---|---|---|
| Saturn study (Sep 2026, UK) | 18 models, 121 UK money questions; 57% of answers wrong | Measures accuracy, not the advice boundary |
| Which? investigation (UK) | Chatbots gave risky or wrong money answers | Consumer accuracy test; no boundary measure or open benchmark |
| FCA research and the Mills Review (Jul 2026) | 26% of UK adults trust chatbots for financial advice; flags "advice-like support" | The regulator may publish its own evidence; cite it |
| Principle-Bench (arXiv, Aug 2026) | AI judges on 168 crypto financial-promotion scenarios under FCA principles | Promotions, not the advice boundary |
| FinGuard (arXiv, May 2026) | Detects regulatory non-compliance in AI conversations | Chinese regulations; single-turn |
| FinSafetyBench (ACL Findings 2026) | Do models refuse financial crime and misconduct requests? | Crime and ethics refusals |
| Fieberg et al. (2025); Lo & Ross (2024) | Is AI investment advice suitable or good? | Assume the AI is advising; don't ask whether it should |
| Aveni FinLLM (UK company) | Guardrails for the advice/guidance line in firms' own systems | Proprietary: no public benchmark |

**Search again before publishing.** Useful alerts: "advice-like" + LLM, "targeted support" + LLM, "advice guidance boundary" + LLM.

## Regulatory sources
- FCA PERG 8.28 (information vs advice): https://www.handbook.fca.org.uk/handbook/PERG/8/28.html
- FCA Glossary, "personal recommendation": https://handbook.fca.org.uk/handbook/glossary/G877.html
- FCA Advice Guidance Boundary Review: https://www.fca.org.uk/firms/advice-guidance-boundary-review
- PS25/22 targeted support, summaries: [KPMG](https://kpmg.com/xx/en/our-insights/regulatory-insights/targeted-support-final-rules.html) · [Linklaters](https://financialregulation.linklaters.com/post/102lxof/target-support-is-coming-fca-publishes-final-rules-and-guidance) · [PwC](https://www.pwc.co.uk/industries/financial-services/understanding-regulatory-developments/fca-fires-the-starting-gun-on-targeted-support.html)
- Mills Review summaries: [A&O Shearman](https://www.aoshearman.com/en/insights/report-issued-by-the-mills-review-the-future-of-ai-in-retail-financial-services) · [Freshfields](https://www.freshfields.com/en/our-thinking/briefings/2026/07/from-assistant-to-decision-maker-the-fcas-mills-review-charts-ais-path-through-retail-financial-services)
