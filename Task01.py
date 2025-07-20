# text_summarizer.py

# ---------- Required Imports ----------
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import nltk

# ---------- Download NLTK tokenizer (only once) ----------
nltk.download("punkt")

# ---------- Input Text (Longer is better) ----------
text = """
The 79-year-old's victory lap came a day after Republicans fell into line and passed the sprawling mega-bill, 
allowing him to sign it as he had hoped on the Fourth of July holiday. 
The bill honours many of Trump's campaign promises: extending tax cuts from his first term, 
boosting military spending, and providing massive new funding for Trump's migrant deportation drive. 
Pilots who carried out the bombing on Iran were among those invited to the White House event, 
which included a picnic for military families on the South Lawn.
"""

# ---------- Summarization ----------
parser = PlaintextParser.from_string(text, Tokenizer("english"))
summarizer = LsaSummarizer()
summary = summarizer(parser.document, 1)  # Change number for longer summary

# ---------- Output ----------
print("Summary:\n")
for sentence in summary:
    print(sentence)
