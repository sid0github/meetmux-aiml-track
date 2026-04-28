import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import string

nltk.download('punkt_tab') 
nltk.download('stopwords') 
nltk.download('wordnet') 

def clean_text(text):
    text= text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation ))

    tokens = word_tokenize(text)

    lemmatizer =WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]

    stop_words = set(stopwords.words('english')) 
    filtered_text = [w for w in tokens if not w in stop_words] 

    return ' '.join(filtered_text)

sample_bio = "I love Hiking in the mountains and Coding late at night!" 
print(f"Cleaned Bio: {clean_text(sample_bio)}")