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

Your task is to identify **evidence that the model represents the semantic concept "B"**, rather than merely attending to unrelated or noisy tokens.

Valid evidence should fall into at least one of the following categories:

* **Directly related to B**
* **Properties or categories logically compatible with B**
* **Concepts that strongly imply B (via world knowledge)**

Do **NOT** count:

* Formatting artifacts, punctuation, or multilingual noise
* Tokens whose connection to B is speculative or weak
* General topical tokens unrelated to B

If evidence is ambiguous or uncertain, **do not include it**.

#### Expected output

Output a JSON list. Each entry should include:

* the layer index
* the evidence tokens
* a brief justification explaining why they indicate "B"

Example:

```json
[
    {
        "layer": 14,
        "tokens": [" natural", " Green"],
        "justification": "These tokens suggest natural, green materials, which are compatible with biodegradability."
    }
]
```

If **no convincing evidence exists**, output:

```json
[]
```