# AI Consensus Framework
A multi-model framework for evaluating LLM classification consistency with mental-health-related text samples, following a process with independent classications, cross-model debate, and consensus analysis

## Overview
With AI becoming a larger part of human life, the usage of LLMs in mental health cases has been explored. When asked to classify a text sample as low, medium, or high risk for depression, LLMs tend to have different classification methods and produce different outputs. This framework, therefore, provides a reproducible method of testing whether classifications become more consistent when seeing other model's responses and given the chance to reconsider their original decision.

## How It Works
This process runs in a total of 4 steps. 
1) Starting off, the prompt and the datasets are given to each of the four models and each make their own independent classifications based on their own training. They are not allowed to see each other's classifications. Each sentence goes through 3 trials, and each model cannot look at it's previous classifications for classifying in the next trial.
2) After the trials are made, the models are allowed to look at each other's classifications and the models have a "debate" and based on the model's individual reasons for picking a classification, the other models either change or don't change their classification.
3) The models make their decisions, and then go for another round of debate, seeing if there was anything they missed in the previous debate.
4) Finally, after the two rounds of debate, the models make their final decisions in their classifications and continue on to their analysis of the results, provided in the 2 csvs and json inside result's folder of the project.

A couple things I kept in mind:  
**Models are not given category definitions or an external scoring rubric.**  
**Models are shown anonymized responses from other models rather than provider names.**

This means that this project is solely to test for the consistency of the model's results and eliminates some of the bias that the models might be susceptible too if they saw the provider names. 
## Models
The models used in this framework are as follows:
| Model | Provider | Role |
| --- | --- | --- |
| GPT 4.0-mini | OpenAI | Classification + debate |
| Sonnet 4.6 | Anthropic | Classification + debate |
| Deepseek V-3 | Deepseek | Classification + debate |
| Qwen2.5 | Ollama | Classification + debate |

They are configurable through individual API keys and separate models can be run individually with the command --models

## Dataset
The experiment used two different datasets from two different sites. The dataset with the depression proxy is from Zenodo and the dataset with casual conversations is from Kaggle. 
  
When building this framework, I could have either made it so that the files could be read in different ways, due to the fact that they have different columns and labeling, or I could change one file and code the rest so that my Python code could process the files in the same way. I chose to change one file, the casual conversations file, using ChatGPT.
  
To run a small scale version of the experiment, use the command --limit N where N represents the entries per dataset. 
## Metrics
The framework evaluates inter-model agreement, consensus stability, and classification changes during debate.

* **Unanimous Agreement Rate:** Percentage of items for which all models agree.
* **Majority Agreement Rate:** Percentage of items for which three of four models agree.
* **Tie and No-Majority Rates:** Measure how often models disagree without a clear majority.
* **Fleiss' Kappa:** Measures inter-model agreement while accounting for chance agreement.
* **Krippendorff's Alpha:** An additional chance-adjusted measure of inter-model agreement.
* **Per-Model Agreement with Consensus:** Measures how often each model matches the group's consensus.
* **Classification Change Rate:** Tracks how often models change their classifications between debate rounds.
* **Cross-Trial Stability:** Measures consistency across repeated experimental trials.

The framework also records agreement by dataset category and experiment metadata to support analysis and reproducibility. Some metrics are metrics I used for evaluation in my research paper.

**Note:** Agreement does not necessarily indicate accuracy, and classification changes do not necessarily indicate improvement. Cross-trial stability requires multiple trials.
## Output Files
For every time the program is run, these three files get rewritten:
| File | Purpose |
| -------------------- | ------------------------------------ |
| `raw_predictions.csv`| Every model response from every round |
| `consensus_results.csv`| Final-round classifications and consensus |
| `summary.json`| Aggregate statistics + reproducibility metadata |
## Installation
**CLONING:**  
git clone ...
cd consensus-model-test

pip install -r requirements.txt

**ENV VARIABLES (Fill with Own API Keys):**   
OPENAI_API_KEY=  
DEEPSEEK_API_KEY=  
ANTHROPIC_API_KEY=  
OLLAMA_BASE_URL=  
OLLAMA_MODEL=  

## Usage
To run the experiment, use the following command:
python -m src.run

Sample test:
python -m src.run --limit 10 --trials 1 --debate-rounds 2 \
  --models openai deepseek claude qwen

## Example Results
From information in summary.json, here is an **example result**:  

A preliminary test of 10 items across four LLMs achieved 50% unanimous agreement (Fleiss’ κ = 0.582), while classification change rates during debate ranged from 0% to 25% across models.
## Limitations
Some things that are important to remember are that this project does not make any claims or provide any results related to accuracy. There is no ground truth to back up any claims related to accuracy, so this is only for testing consistency within the models and between models.
  
Another limitation is that the datasets that were used only pulled data from Reddit, therefore, this data is biased towards Reddit users. Furthermore, this project can only be used with individual API keys with a user having to put in their own keys and download the Ollama model on their end, causing a slight financial bias, as there is cost involved.
  
Most importantly, using the results from this project to generalize or draw far-fetched conclusions is dangerous because this project deals in mental health, a sensitive issue for people. Drawing overly generalized conclusions may lead to a misunderstanding that could indirectly deteriorate the mental health of a user's clients or patients.
## Future Work
To continue this project, there are multiple angles in which I could take this project.
  
1) **Adding more models:** Scaling up the project with more classifications and more perspective from more models could provide even more consistent results.
2) **Add a Ground Truth:** Currently, this project only measures consistency in labelling, but adding a ground truth would make a clearer picture of how the models are interpreting different mental health signals and if it would help patients.
3) **Experiment with additional frameworks:** This project uses a consensus framework in which the models converse with each other, decide which reasoning is stronger, and adjust accordingly. However, adding more frameworks in which models talk to each other, not just in a debate setting but in different ways like a committee, may show results that can help us understand effective ways to view mental health.

## Connection to Research
This framework was developed as an extension of research investigating inter-model agreement in LLM-based depression-risk classification. While the original study focused on measuring agreement, this project explores whether structured cross-model interaction can affect classification consistency.

## License
This project is licensed under the MIT License. See the LICENSE file for details.