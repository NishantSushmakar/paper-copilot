from copilot.llm.llm_client import LLMResponse, chat_with_llm
from evals.adapters.qasper import QASPERQuestion
from pydantic import BaseModel
from typing import Optional
import re



class SystemOutputModel(BaseModel): 

    prediction:str
    response:Optional[LLMResponse]=None


def oracle_copy(question:QASPERQuestion):

    response = {
        "prediction": question.answers[0].text
        }

    return SystemOutputModel.model_validate(response)


def always_unanswerable(question:QASPERQuestion):

    response = {
        "prediction": "unanswerable"
    }

    return SystemOutputModel.model_validate(response)

def closed_book(question:QASPERQuestion):

    system_prompt = """
            You are a researcher in NLP field. 
            The user will ask you a question about a scientific paper you can't see. 
            For every question, make your best guess when you don't have enough context. 
            If it is a Yes or No question, just answer yes or no. Keep your answers short, 
            give a list when the question asks for it, 
            and give no explanation of the thinking behind the answer. 
            Given below are some examples of the kind of questions and how to answer them.

            For example:
            1. What datasets did they use?
            Answer: News Articles, Twitter
            2. How do they measure the size of models?
            Answer: Direct comparison of model parameters
            3. Are the models compared to some baseline models?
            Answer: yes


            """

    
    user_prompt = question.question

    messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user",   "content": user_prompt}
      ]

    answer = chat_with_llm(slot=system_settings["closed_book"]["slot"],
    messages=messages,temperature=system_settings["closed_book"]["temperature"],
    max_tokens=system_settings["closed_book"]["max_tokens"],thinking=system_settings["closed_book"]["thinking"]
    )

    text = re.sub(r"^answer:", "", answer.text.strip(), flags=re.IGNORECASE)
    

    response = {
        "prediction": text,
        "response": answer
    }

    return SystemOutputModel.model_validate(response)


def get_evidences(question:QASPERQuestion):
    evidence_set = ()

    for answer in question.answers: 
        for evidence in answer.evidence:
            evidence_set = evidence_set + (evidence,)

    if len(evidence_set)==0: 
        return ""

    evidence_str = ""

    unique = tuple(dict.fromkeys(evidence_set))

    for i,evidence in enumerate(unique):
        evidence_str += f"[{i+1}] {evidence}\n"

    
    evidence_str = evidence_str.strip()

    return evidence_str




def oracle(question:QASPERQuestion):

    system_prompt = r"""You are a researcher in NLP field.
The user will give you evidence passages from a scientific paper, followed by a question about that paper.
Answer the question using the evidence, and copy the words from the evidence wherever possible.
When you don't have enough context from the evidence, just answer "Unanswerable".
If it is a Yes or No question, just answer yes or no.
Keep your answers short, give a list when the question asks for it, and give no explanation of the thinking behind the answer.
Given below are some examples of the kind of questions and how to answer them.

Example 1:
[Evidence]
[1] Evaluation Framework ::: Datasets ::: News Articles
[2] Our dataset source of news articles is described in BIBREF2. This dataset was built from two different sources, for the trusted news (real news) they sampled news articles from the English Gigaword corpus. For the false news, they collected articles from seven different unreliable news sites. These news articles include satires, hoaxes, and propagandas but not clickbaits. Since we are interested also in analyzing clickbaits, we slice a sample from an available clickbait dataset BIBREF33 that was originally collected from two sources: Wikinews articles' headlines and other online sites that are known to publish clickbaits. The satire, hoax, and propaganda news articles are considerably long (some of them reach the length of 5,000 words). This length could affect the quality of the analysis as we mentioned before. We focus on analyzing the initial part of the article. Our intuition is that it is where emotion-bearing words will be more frequent. Therefore, we shorten long news articles into a maximum length of N words (N=300). We choose the value of N based on the length of the shortest articles. Moreover, we process the dataset by removing very short articles, redundant articles or articles that do not have a textual content.
[3] With the complicated political and economic situations in many countries, some agendas are publishing suspicious news to affect public opinions regarding specific issues BIBREF0. The spreading of this phenomenon is increasing recently with the large usage of social media and online news sources. Many anonymous accounts in social media platforms start to appear, as well as new online news agencies without presenting a clear identity of the owner. Twitter has recently detected a campaign organized by agencies from two different countries to affect the results of the last U.S. presidential elections of 2016. The initial disclosures by Twitter have included 3,841 accounts. A similar attempt was done by Facebook, as they detected coordinated efforts to influence U.S. politics ahead of the 2018 midterm elections.
[4] For this dataset, we rely on a list of several Twitter accounts for each type of false information from BIBREF6. This list was created based on public resources that annotated suspicious Twitter accounts. The authors in BIBREF6 have built a dataset by collecting tweets from these accounts and they made it available. For the real news, we merge this list with another 32 Twitter accounts from BIBREF34. In this work we could not use the previous dataset and we decide to collect tweets again. For each of these accounts, we collected the last M tweets posted (M=1000). By investigating these accounts manually, we found that many tweets just contain links without textual news. Therefore, to ensure of the quality of the crawled data, we chose a high value for M (also to have enough data). After the collecting process, we processed these tweets by removing duplicated, very short tweets, and tweets without textual content. Table TABREF35 shows a summary for both datasets.
[End of evidence]
Question: What datasets did they use?
Answer: News Articles, Twitter

Example 2:
[Evidence]
[1] Traditional text-to-speech (TTS) systems are composed of complex pipelines BIBREF0 , these often include accoustic frontends, duration model, acoustic prediction model and vocoder models. The complexity of the TTS problem coupled with the requirement for deep domain expertise means these systems are often brittle in design and results in un-natural synthesized speech.
[2] Neural text-to-speech systems have garnered large research interest in the past 2 years. The first to fully explore this avenue of research was Google's tacotron BIBREF1 system. Their architecture based off the original Seq2Seq framework. In addition to encoder/decoder RNNs from the original Seq2Seq , they also included a bottleneck prenet module termed CBHG, which is composed of sets of 1-D convolution networks followed by highway residual layers. The attention mechanism follows the original Seq2Seq BIBREF7 mechanism (often termed Bahdanau attention). This is the first work to propose training a Seq2Seq model to convert text to mel spectrogram, which can then be converted to audio wav via iterative algorithms such as Griffin Lim BIBREF8 .
[3] The architecture of our model utilizes RNN-based Seq2Seq model for generating mel spectrogram from text. The architecture is similar to that of Tacotron 2 BIBREF4 . The generated mel spetrogram can either be inverted via iterative algorithms such as Griffin Lim, or through more complicated neural vocoder networks such as a mel spectrogram conditioned Wavenet BIBREF11 .
[4] Direct comparison of model parameters between ours and the open-source tacotron 2, our model contains 4.5 million parameters, whereas the Tacotron 2 contains around 13 million parameters with default setting. By helping our model learn attention alignment faster, we can afford to use a smaller overall model to achieve similar quality speech quality.
[End of evidence]
Question: How do they measure the size of models?
Answer: Direct comparison of model parameters

Example 3:
[Evidence]
[1] Our baseline is a GRU network for each of the three tasks. We use the same network architecture across the 3 tasks. For each network, the network contains a layer unidirectional GRU, with 500 units and an output linear layer. The network is trained end-to-end. Our input embedding layer is initialized with a standard normal distribution, with $\mu =0$, and $\sigma =1$, i.e., $W \sim N(0,1)$. We use a maximum sequence length of 50 tokens, and choose an arbitrary vocabulary size of 100,000 types, where we use the 100,000 most frequent words in TRAIN. To avoid over-fitting, we use dropout BIBREF2 with a rate of 0.5 on the hidden layer. For the training, we use the Adam BIBREF3 optimizer with a fixed learning rate of $1e-3$. We employ batch training with a batch size of 32 for this model. We train the network for 15 epochs and save the model at the end of each epoch, choosing the model that performs highest accuracy on DEV as our best model. We present our best result on DEV in Table TABREF7. We report all our results using accuracy. Our best model obtains 42.48% for age, 37.50% for dialect, and 57.81% for gender. All models obtains best results with 2 epochs.
[End of evidence]
Question: Are the models compared to some baseline models?
Answer: yes
""".strip()


    evidence_str = get_evidences(question) 

    if len(evidence_str)==0:
        evidence_str = "No evidence found"



    user_prompt = f"""[Evidence]\n{evidence_str}\n[End of evidence]\nQuestion: {question.question}""".strip()

    messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
            ]

    answer = chat_with_llm(slot=system_settings["oracle"]["slot"],
    messages=messages,temperature=system_settings["oracle"]["temperature"],
    max_tokens=system_settings["oracle"]["max_tokens"],thinking=system_settings["oracle"]["thinking"]
    )


    text = re.sub(r"^answer:", "", answer.text.strip(), flags=re.IGNORECASE)
    

    response = {
        "prediction": text,
        "response": answer
    }

    return SystemOutputModel.model_validate(response)





system_dict = {
    "oracle_copy": oracle_copy,
    "always_unanswerable": always_unanswerable,
    "closed_book": closed_book,
    "oracle": oracle

}


system_settings = {
    "closed_book": {
        "slot":"MEDIUM",
        "temperature":0.6,
        "thinking":True,
        "max_tokens":4000
    },
    "oracle": {
        "slot":"MEDIUM",
        "temperature":0.6,
        "thinking":True,
        "max_tokens":4000
    }
}

