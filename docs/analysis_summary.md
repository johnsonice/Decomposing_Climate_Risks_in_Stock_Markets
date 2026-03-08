# LLM vs Traditional Sentiment Models: Climate News Classification

## Motivation

A reviewer suggested comparing our LLM-based classification approach against traditional keyword/rule-based sentiment models to validate the value of using LLMs for climate news classification. We ran VADER (NLTK) and TextBlob alongside Llama-3.1-8B (few-shot chain-of-thought) on the same dataset of climate news paragraphs, classifying each as **favorable** or **unfavorable** to climate-friendly policies.

---

## Model Descriptions

| Model | Type | How It Works |
|-------|------|-------------|
| **Llama-3.1-8B** | LLM (few-shot CoT) | Reads the full paragraph with few-shot examples and chain-of-thought reasoning to determine whether the text supports or undermines climate-friendly policies |
| **VADER** | Lexicon + rule-based | Looks up ~7,500 pre-scored sentiment words, applies heuristic rules (negation, capitalization, degree modifiers), outputs a compound score in [-1, +1] |
| **TextBlob** | Pattern-based | Averages polarity scores from a dictionary of adjectives, outputs a polarity in [-1, +1] |

For VADER and TextBlob, positive scores are mapped to "favorable" and negative scores to "unfavorable" (threshold = 0.0).

---

## Performance Comparison

### Macro Metrics (Validation Set, 108 samples)

| Model | Accuracy | Precision | Recall | F1 (Macro) |
|-------|----------|-----------|--------|------------|
| **Llama-3.1-8B** | **0.7870** | **0.7843** | **0.7854** | **0.7848** |
| VADER | 0.5463 | 0.5231 | 0.5167 | 0.4931 |
| TextBlob | 0.5370 | 0.5063 | 0.5042 | 0.4716 |

### Per-Class F1 Scores (Validation Set)

| Model | F1 (favorable) | F1 (unfavorable) |
|-------|----------------|------------------|
| **Llama-3.1-8B** | **0.81** | **0.76** |
| VADER | 0.66 | 0.33 |
| TextBlob | 0.66 | 0.29 |

### Key Observation

Llama-3.1-8B outperforms both traditional models by 29-31 F1 points on the validation set, with VADER and TextBlob performing near random chance (~54% accuracy) and failing especially on the "unfavorable" class (F1 of 0.29-0.33 vs Llama's 0.76), because they systematically misclassify unfavorable paragraphs that contain positive-sounding climate vocabulary.

---

## Why Traditional Models Fail: Error Analysis

The fundamental problem is that VADER and TextBlob measure **emotional polarity** (positive/negative tone), while the classification task requires understanding **policy stance** (supports or undermines climate action). These two dimensions are often misaligned in climate news.

We identified three systematic failure patterns:

### Pattern 1: Negative Tone, but Favorable to Climate Action

Paragraphs that describe climate threats or use urgent language to *argue for* stronger climate policy. Traditional models see negative words and classify as "unfavorable."

**Example — International cooperation on climate (Val):**
> *"To this end, Ban Ki-moon, the UN secretary-general, has been working closely with French president Hollande, who will host critical climate negotiations in Paris... leaders quickly accepted the secretary-general's invitation."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | favorable | Recognizes proactive diplomatic action to address climate change |
| VADER | unfavorable | "critical" triggers negative sentiment |
| TextBlob | unfavorable | Same — negative keyword bias |

**Example — Proven technologies for climate action (Train):**
> *"Climate change can seem so complex and global that action by any one country or individual can seem futile. In reality, however, much can be done using known and proven technology. Energy use could be cut by at least 20%... proven technologies such as wind, solar and systems to convert waste into power could be deployed."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | favorable | Understands this is a strong argument *for* deploying renewable technology |
| VADER | unfavorable | Opening clause ("complex", "futile") poisons the overall score |
| TextBlob | unfavorable | Same — averages negative opening with positive body |

**Example — Growth of renewables (Val):**
> *"That started to change after the 1970s oil shock, when a spike in crude prices spurred interest in homegrown alternatives to imported oil, and climate change concerns began to drive the development of renewable energy sources."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | favorable | Narrative about the *growth* of renewable energy |
| VADER | unfavorable | "shock", "spike" trigger negative sentiment |
| TextBlob | unfavorable | Same |

### Pattern 2: Neutral/Positive Tone, but Unfavorable to Climate Policy

Paragraphs that use calm, matter-of-fact, or even positive language to *undermine* climate action. Traditional models see no negative words and classify as "favorable."

**Example — Climate skepticism (Val):**
> *"'The great European switch to diesel engines was a top-down decision as a direct result of exaggerated fears about climate change,' says Matt Ridley, a frequent critic of tougher global warming regulations."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Catches "exaggerated fears" as climate skepticism |
| VADER | favorable | No overtly negative sentiment words |
| TextBlob | favorable | "great" scores positively |

**Example — Exploiting COVID to weaken environmental rules (Val):**
> *"Ricardo Salles, Brazil's environmental minister, was caught on video saying the government should take advantage of the media's focus on the COVID-19 pandemic to 'change and simplify' environmental rules."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Understands the context: exploiting a crisis to *weaken* environmental protections |
| VADER | favorable | "take advantage", "simplify" — neutral/positive words |
| TextBlob | favorable | Same |

**Example — Trump's climate stance (Train):**
> *"Carbon dioxide, sulphur dioxide, methane and... Donald Trump? He has dismissed climate change as a hoax, threatened to ditch the Paris accord and promised to dismantle the Obama administration's clean energy programme."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Recognizes this describes *attacks* on climate policy |
| VADER | favorable | Ironic/literary tone scores as neutral-positive |
| TextBlob | favorable | Same |

### Pattern 3: Criticism of Insufficient Climate Action

Paragraphs that criticize governments for *not doing enough* on climate. The language may include positive-sounding climate vocabulary ("tackling climate change", "renewable energy"), but the actual message is frustration at policy failure.

**Example — UK readiness assessment (Val):**
> *"In its first assessment of the UK's readiness for climate change... the sub-committee found that the government had 'made some progress' in raising awareness but that 'very little tangible action' had taken place."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Sees this as criticism of *inadequate* government action |
| VADER | favorable | "some progress" + balanced language → positive score |
| TextBlob | favorable | Same |

**Example — Australia's inaction (Train):**
> *"'We don't see any signs that the national level decision makers are willing to take the steps required to ensure Australia plays its part in tackling climate change, protecting communities and unleashing the huge opportunities in renewable energy...'"*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Understands this expresses *frustration at government inaction* |
| VADER | favorable | "tackling climate change", "huge opportunities", "renewable energy" — all positive words |
| TextBlob | favorable | Same |

**Example — Fossil fuel lobbying (Train):**
> *"Unlike other cases of policy delay, the costs of delay on climate change are not just lost time but also lost opportunity... This makes the lobbying by the fossil fuel industries against control measures even more understandable. They are not just buying time; they are trying to burn through the targets."*

| Model | Prediction | Why |
|-------|-----------|-----|
| **Llama** | unfavorable | Recognizes this criticizes fossil fuel lobbying as undermining climate targets |
| VADER | favorable | "opportunity", "understandable" — positive-leaning words |
| TextBlob | favorable | Same |

---

## Conclusion

Traditional keyword-based sentiment models (VADER, TextBlob) are fundamentally unsuitable for climate news stance classification. They measure emotional tone, not policy stance — and these two dimensions are frequently misaligned in climate discourse. Specifically:

1. **Urgency does not mean opposition.** Paragraphs that describe climate threats to motivate stronger policy use negative language but hold a *favorable* stance.
2. **Calm language does not mean support.** Climate skepticism and policy rollbacks are often described in neutral, matter-of-fact tones.
3. **Mentioning climate action is not the same as supporting it.** Criticisms of insufficient policy use the same vocabulary as policy advocacy.

The LLM (Llama-3.1-8B) can parse argumentative structure, irony, and domain-specific context that keyword-based tools completely miss. This results in a **+20-30 point F1 improvement** over traditional approaches, with the gap widening on the harder "unfavorable" class. On the validation set, 25% of samples can *only* be correctly classified by the LLM.

These findings support the use of LLM-based approaches for nuanced financial and policy text classification tasks where surface-level sentiment diverges from underlying stance.
