#### Instruction

You will be given a prompt involving negation and a list of attention output tokens from different layers of a language model.

#### Prompt format

The prompt will be of the form **"A that are not B"**.

Example prompt:

- Here is a list of materials that are not biodegradable:

In this prompt:
* **A** = "materials"
* **B** = "biodegradable"

#### What to look for

Your task is to identify **evidence that the model represents the semantic concept "not B"**, rather than merely attending to unrelated or noisy tokens.

Valid evidence should fall into at least one of the following categories:

* **Direct antonyms or negations of B**
* **Properties or categories logically incompatible with B**
* **Concepts that strongly imply not B (via world knowledge)**

Do **NOT** count:

* Formatting artifacts, punctuation, or multilingual noise
* Tokens whose connection to not B is speculative or weak
* General topical tokens unrelated to B

If evidence is ambiguous or uncertain, **do not include it**.

#### Example attention outputs

[10_attn_out]
['…\n', 'enger', '…', 'ants', ' …', ' …\n', ' examples', '…”', 'oda', ' —']

[11_attn_out]
['alf', 'alten', ',', ' fruits', 'пример', 'addock', '司', ' names', 'itz', ';']

[12_attn_out]
['нт', 'roke', 'agen', 'urat', ' muse', 'igli', '_FLUSH', 'oise', 'ш', 'خوان']

[13_attn_out]
['…\n', ' …\n', '…', ' […]\n', '…”', '….', '……', '…..', '-present', '.mvp']

[14_attn_out]
['utz', ' Guys', 'lington', ' commercial', 'utherford', 'ucz', ' Remaining', '-elect', 'ęż', '451']

[15_attn_out]
[' Spieler', 'ixa', '…\n', 'antz', 'R', 'K', ' Eddie', 'iff', '…', ' Nash']

[16_attn_out]
[' …\n', ' green', ' fri', 'าย', 'juven', 'inq', ' Alive', 'jr', ' Mother', ' Howe']

[17_attn_out]
['…\n', ' …\n', '…', ' […]\n', '….', '…..', '……', 'odka', '”\n', ',…']

[18_attn_out]
[' آنها', ' ones', ' them', '…\n', '它们', ' they', 'erken', ' These', 'ods', ' chamber']

#### Example reasoning chain

- the negation part in the prompt is "not biodegradable"
- going through the attention outputs, `14_attn_out` contains " commercial" and " Remaining". " commercial" could be related to unnatural. " Remaining" could be related to not decomposing.

Side note: Other examples contain easier cases. For example:
- " solid" relates to "not gas"
- " inland" relates to "not located near the ocean"
- " American" relates to "not in Asia"

#### Expected output

Output a JSON list. Each entry should include:

* the layer index
* the evidence tokens
* a brief justification explaining why they indicate "not B"

Example:

```json
[
    {
        "layer": 14,
        "tokens": [" commercial", " Remaining"],
        "justification": "These tokens suggest non-natural, persistent materials, which are incompatible with biodegradability."
    }
]
```

If **no convincing evidence exists**, output:

```json
[]
```