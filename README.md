> **This is a mirror of the model card.** The weights are on the Hugging Face Hub, at
> [blazejewicz/functiongemma-270m-it-gemma-teacher](https://huggingface.co/blazejewicz/functiongemma-270m-it-gemma-teacher), behind an access gate. `FILES.md` lists each file
> with its size, its sha256 and a link to the pinned revision.

# FunctionGemma 270M — voice commands of a screen-free appliance (English & Polish) — v6

<img src="https://raw.githubusercontent.com/peterblazejewicz/functiongemma-270m-it-gemma-teacher/main/assets/gemma-teacher.jpg" alt="A learner at a desk speaks to Gemma Teacher, a round, screen-free speakerphone with a few keys; a Japanese notebook lies beside it" width="900">

A full fine-tune of **Google's FunctionGemma 270M**
([`google/functiongemma-270m-it`](https://huggingface.co/google/functiongemma-270m-it), revision
`39eccb091651513a5dfb56892d3714c1b5b8276c`) that reads what a person says to the settings menu of a
screen-free, hands-free voice appliance, in English or Polish. For each utterance it gives **one call** of a
small declared set of functions, or **no call** when the words are not a command.

> **A learning project.** This model is the result of a personal learning project: it explores how a small
> function-calling model can read spoken menu commands of a screen-free voice appliance. It is shared as study
> material, to show the method and the measurements. It is not a product, it is not intended for commercial use,
> and it must not control anything where a wrong command can cause harm. Its answers can be wrong (see
> "Limitations").

> **Gemma is provided under and subject to the Gemma Terms of Use found at
> [ai.google.dev/gemma/terms](https://ai.google.dev/gemma/terms).** This model is a modified version of
> FunctionGemma 270M: its weights were changed by the fine-tune that this card describes. The use restrictions of
> the [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy) apply to it and to every
> derivative of it. Copies of both documents are in this repository.

---

## Quick start

The files are gated: first, click **Agree and get access** on
[this model's page](https://huggingface.co/blazejewicz/functiongemma-270m-it-gemma-teacher), and make an
[access token](https://huggingface.co/settings/tokens) of the type **Read**.

### In the browser: Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peterblazejewicz/functiongemma-270m-it-gemma-teacher/blob/main/quick-start.ipynb)

The notebook loads the model with Transformers on the free CPU runtime, and you type what a person would say to
the appliance. Its first cell tells you how to give it your token.

### On your computer: Transformers

Log in first with your token: `hf auth login` (or set the `HF_TOKEN` environment variable).

```python
import json, re, torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

repo = "blazejewicz/functiongemma-270m-it-gemma-teacher"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
tools = json.load(open(hf_hub_download(repo, "tools.json"), encoding="utf-8"))

def propose(utterance):
    inputs = tokenizer.apply_chat_template([{"role": "user", "content": utterance}], tools=tools,
                                           add_generation_prompt=True, return_dict=True, return_tensors="pt")
    end = [tokenizer.convert_tokens_to_ids("<end_function_call>"), tokenizer.eos_token_id]
    output = model.generate(**inputs, max_new_tokens=64, do_sample=False, eos_token_id=end)
    text = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:])
    call = re.search(r"<start_function_call>call:(\w+)\{(.*?)\}<end_function_call>", text)
    if not call:
        return None                                   # no command
    return call.group(1), dict(re.findall(r"(\w+):<escape>(.*?)<escape>", call.group(2)))

print(propose("Ustaw długość wyjaśnień na szczegółową"))  # ('set_explanation_length', {'value': 'Detailed'})
print(propose("How do I say thank you in Japanese?"))      # None
```

- **One user turn, the declarations of `tools.json`, nothing else.** Add no system or developer message of your
  own (the chat template puts the declarations in its developer turn); the model was trained with exactly these
  11 declarations, in this order.
- **Stop at `<end_function_call>`.** FunctionGemma's format continues with a function response after a call; a
  caller that does not stop there reads text that the model invents.
- **No call** is the text `No function call is needed.`: treat it, and any text without a call, as "not a
  command".
- Greedy decoding.

### On your computer: llama.cpp

Start `llama-server` with a GGUF file of this repository, `--jinja`, and a context of at least 2,048 tokens (the
declarations take about 1,000). Send the declarations in the `tools` field of `/v1/chat/completions` with
`"temperature": 0` and no `stop`: the model ends after one call, and the call comes back in the message text with
its `<end_function_call>` tag. Read it with the regular expression above. (A `stop` of `<end_function_call>`
removes that tag from the text, and the expression then finds no call.)

---

## 1. Where the model sits

The appliance is a speakerphone with a few keys and no screen. A person changes its settings by voice: the
language of the menu, the teacher's language and speed, the length of the explanations, the level, the language that the person speaks, how
the menu is controlled, and a reset. The menu is a ring of *settings cards* that the appliance reads aloud.

This is an **eyes-free menu**, in the sense of earPod (Zhao et al., CHI 2007): a person must be able to name a
card or a value directly, because an audio menu costs time for each item that the person must hear before the one
they want. But people say one thing in many ways, mix the two languages, correct themselves ("brief... no,
detailed"), or say a setting word in a sentence that asks for nothing ("my wife speaks Polish").

<img src="https://raw.githubusercontent.com/peterblazejewicz/functiongemma-270m-it-gemma-teacher/main/assets/c4-context.png" alt="C4 system context: a person speaks to a screen-free voice appliance, which asks this model for one call or no call" width="900">

The appliance reads a command in **three tiers**, from the cheapest and strictest to the most flexible: a closed
grammar that hears the fixed command phrases, a table of command words, and a proposer that asks this model. The
proposer asks only when the table cannot decide, as in a cascade that asks a model only when a cheaper step
cannot (FrugalGPT). The orange element of each diagram is this model:

<img src="https://raw.githubusercontent.com/peterblazejewicz/functiongemma-270m-it-gemma-teacher/main/assets/c4-containers.png" alt="C4 containers: speech input, the command reader, this model, the keys, the menu host and speech output" width="900">

<img src="https://raw.githubusercontent.com/peterblazejewicz/functiongemma-270m-it-gemma-teacher/main/assets/c4-components.png" alt="C4 components: the closed grammar in speech input, then the table of command words, the proposer that asks this model, and the gate" width="900">

**The model answers; the appliance decides.** A gate checks each call against the menu state and the words that
were said, and a change away from a safe value still waits for the person's confirmation. The model's job is
narrow: one utterance to one declared call, or to no call, and above all **no call when the words are not a
command**. A refusal costs the person one more sentence; a wrong change costs more.

## 2. The functions

`tools.json` holds the 11 declarations, in the OpenAI form that `apply_chat_template(tools=...)` takes:

| Function | What it asks for |
| :--- | :--- |
| `set_operation_language` | The language of the menu (English, Polish) |
| `set_teacher_language` | The language the teacher explains in (English, Polish) |
| `set_teacher_speed` | The speed of the teacher's voice (Normal, Slower) |
| `set_explanation_length` | The length of the explanations (Normal, Brief, Detailed) |
| `set_learner_level` | The level (Beginner, Intermediate) |
| `set_learner_language` | The language the person speaks in lessons (English, Polish) |
| `set_menu_control` | Keys or speech for the menu (Keys, Speaking) |
| `set_reset` | Keep the settings, or reset them to the defaults when leaving the menu (Keep, ResetOnLeave) |
| `navigate` | The next card, the previous card, or the other value of this card |
| `query_card` | What a card holds now, or which values it offers |
| `session_action` | Repeat what was said last, or help |

Confirming, cancelling, leaving the menu and stopping are **not** functions of the model: the appliance takes them
only from its keys or its strict tiers, and the model was trained to give no call for them.

## 3. Evaluation

Measured on 2026-09-28; the model ran on one NVIDIA DGX Spark (GB10). Both question sets are authored, not recorded from people,
in English and Polish; neither is published.

| Measurement | Base FunctionGemma | **v6** |
| :--- | :---: | :---: |
| **The appliance's path** (137 utterances) | | |
| Correct accept: the expected change (46) | 22 | **42** |
| False accept: a change where the words ask for none (5 that can give one) | 0 | **0** |
| Time for one request of v6, median (Q8_0, `llama-server`, through the network) | | 78 ms |
| **The synthetic set** (292 utterances) | | |
| All rows correct | 32.5% | **95.5%** |
| False calls, rows that need no call (148) | 99 | **1** |
| Missed calls, rows that need a call (144) | 7 | 7 |

**Why v6.** On the appliance's path, v6 gives no false accept, where the earlier releases gave some (v3: 3; v4
and v5: 1 each, "keep going" taken as "keep the settings"), with as many correct accepts (42 of 46). One
utterance separates v6 from v4 and v5, so this is a small difference; but in a hands-free menu a change that
nobody asked for costs more than a command said twice.

- **The appliance's path** is its own test suite: 137 utterances that reach this tier, each sent through the
  appliance's real request and gate to `llama-server` (Q8_0, `--jinja`). Only 5 of its 91 no-change utterances can
  give a false accept (the gate refuses the rest whatever the model says), and the training data holds close
  paraphrases of 3 of those 5.
- **The base model is called as the appliance calls it**, without a stop sequence: it does not end after its
  first call, and an answer with several calls counts as none. This measures it in the appliance, not at its best.
- **The synthetic set** comes from the generator of the training data (no sentence shared), so it measures how
  well the model learned the data more than how it meets new speech.

## 4. Training

Full fine-tune of the base model (no adapter): bf16, AdamW, learning rate 5 × 10⁻⁵, 2 epochs, batch 16, seed 42,
on one NVIDIA DGX Spark (GB10): 414 steps in 1,350 s, training loss 0.043. The loss is on the answer only. The data (private) are 3,300 synthetic rows in English and Polish, 30%
with no call: every setting value, navigation, queries, repeat and help, mixed-language commands,
self-corrections, negations, requests that are not functions of the model, small talk, and setting words without
a request. Every Polish sentence was reviewed by three judge models (Bielik-11B v3.0, Qwen3.6-35B-A3B,
Gemma-4-26B-A4B); where they disagreed, a separate model (Claude) decided, and two more such models ruled on
each repeated pattern. No native speaker reviewed the data.

## 5. Limitations

- **A closed world.** It knows only the 11 declarations and only English and Polish.
- **It must be checked.** A wrong call is possible; validate every call and every value against `tools.json` and
  the state of what it controls, and confirm a change that matters.
- **Authored sentences, not people's speech.** Recognition errors, accents and real phrasing are not measured.
- **Degree words are weak.** "make it harder" gives the wrong level.
- **Not a chat model**, and not tested for safety beyond its task.

## 6. Versions and files

- **v6** (this version): every Polish training sentence reviewed by three judge models; plain descriptions in
  `tools.json` (after *Adapt Tool Schemas to the Models*, section 7). **v5**: rewritten Polish templates. **v4**: sentences that name a setting word without a request.
- Report a problem in the [Community tab of the model on the Hugging Face Hub](https://huggingface.co/blazejewicz/functiongemma-270m-it-gemma-teacher/discussions).

| File | What it is |
| :--- | :--- |
| `model.safetensors` | The fine-tuned weights, BF16; sha256 `d716f244439bdaf0dd63de9db891579435ee0d45a3310be371f325b612330f4f` |
| `config.json`, `generation_config.json` | The configuration of the fine-tuned model |
| tokenizer files, `chat_template.jinja` | The tokenizer and chat template of the base model, unchanged |
| `tools.json` | The 11 function declarations |
| `functiongemma-270m-it-gemma-teacher-v6-f16.gguf` | GGUF F16; sha256 `81a982dc615efefd3baf76854cc5a9edbd0748657784df3aec29de538df1d01b` |
| `functiongemma-270m-it-gemma-teacher-v6-q8_0.gguf` | GGUF Q8_0; sha256 `81ca79d3afffc41a5732057e241a8329ac0098e970dd32529ecf5af112e37510` |
| `quick-start.ipynb` | The Colab notebook of the Quick start |
| `NOTICE`, `GEMMA_TERMS_OF_USE.md`, `GEMMA_PROHIBITED_USE_POLICY.md` | The Gemma notice and copies of the Gemma terms |
| `assets/` | The image of the appliance and the three diagrams of section 1 (the card shows them from the GitHub mirror) |

## 7. References

Models used, at the exact revision:

- [google/functiongemma-270m-it](https://huggingface.co/google/functiongemma-270m-it) at
  `39eccb091651513a5dfb56892d3714c1b5b8276c`: the base model of the fine-tune.
- [speakleash/Bielik-11B-v3.0-Instruct](https://huggingface.co/speakleash/Bielik-11B-v3.0-Instruct) at
  `735bfee1125fe8b497ac2769de94822a11f77167` (BF16): a judge of the Polish training sentences.
- [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) at
  `1355db6a052410cfd62085d94b58866fd0f2c3c5`, an NVFP4 quantization of
  [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B): a judge of the Polish sentences.
- [nvidia/Gemma-4-26B-A4B-NVFP4](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) at
  `a19cfe00be84568a6867111c9a68c9c44fdcffe6`, an NVFP4 quantization of
  [google/gemma-4-26B-A4B-it](https://huggingface.co/google/gemma-4-26B-A4B-it): a judge of the Polish sentences.
- Claude (Anthropic, `claude-fable-5-1`; not on the Hub): decided the ties of the judges and ruled on the
  repeated patterns.
- Software: the judges ran in vLLM (`nvcr.io/nvidia/vllm:26.05.post1-py3`), the fine-tune in
  [Transformers](https://github.com/huggingface/transformers), and the GGUF conversion and the evaluation in
  [llama.cpp](https://github.com/ggml-org/llama.cpp) build 9949.

Sources:

- S. Zhao, P. Dragicevic, M. Chignell, R. Balakrishnan, P. Baudisch. *earPod: Eyes-free Menu Selection using Touch
  Input and Reactive Audio Feedback.* CHI 2007. [doi:10.1145/1240624.1240836](https://doi.org/10.1145/1240624.1240836)
- L. Chen, M. Zaharia, J. Zou. *FrugalGPT* — a cascade that asks a model only when a cheaper step cannot
  decide. [arXiv:2305.05176](https://arxiv.org/abs/2305.05176)
- *Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models.*
  [arXiv:2510.07248](https://arxiv.org/abs/2510.07248)
- M. Mitchell et al. *Model Cards for Model Reporting.* [arXiv:1810.03993](https://arxiv.org/abs/1810.03993)
